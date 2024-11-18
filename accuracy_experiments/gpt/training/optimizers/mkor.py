import math

import torch
import torch.optim as optim
from .utils.timing import Timer
from .utils.factors import ComputeI, ComputeG, ComputeCovA, ComputeCovG, sm_inverse
from .utils.backend import EmptyBackend, TorchBackend


class MKOR(optim.Optimizer):
    def __init__(self,
                 param_groups,
                 model,
                 stat_decay=0.95,
                 inv_freq=10,
                 stabilization_factor=0.1,
                 measure_time=False,
                 backend=TorchBackend(),
                 half_precision=True,
                 grad_accum_steps=1,
                 sgd_layers=[],
                 optimizer='adamw',
                 grad_scale=1.0,
                 clipping_value=100.0,
                 warmup_steps=0,
                 **kwargs):
        """
        MKOR optimizer.
        :param model: the model
        :param stat_decay: the decay factor for the statistics
        :param inv_freq: the frequency of the preconditioning
        :param stabilization_factor: the stabilization factor for the preconditioning
        :param measure_time: if True, the time will be measured
        :param backend: the backend for the distributed training
        :param half_precision: if True, the optimizer will use half precision
        :param grad_accum_steps: the number of steps to accumulate the gradients
        :param sgd_layers: the layers that don't need to be preconditioned
        :param optimizer: the optimizer to be used
        :param grad_scale: the scale of the gradients
        :param clipping_value: the value to clip the gradients
        :param warmup_steps: the number of warmup steps
        :param kwargs: the arguments for the optimizer
        """
        defaults = dict()
        super(MKOR, self).__init__(param_groups, defaults)

        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.AHandler = ComputeI()
        self.GHandler = ComputeG()

        self.backend = backend
        self.verbose = self.backend.rank() == 0

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []

        self.model = model
        self.device = self.model.device

        self._prepare_model(sgd_layers=sgd_layers)

        self.steps = 0

        self.grad_outputs = [0] * len(self.modules)

        self.a_reset_factor = [True] * len(self.modules)
        self.g_reset_factor = [True] * len(self.modules)

        self.m_aa, self.m_gg = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_inv, self.GG_inv = [0] * len(self.modules), [0] * len(self.modules)
        self.AA, self.GG = [0] * len(self.modules), [0] * len(self.modules)
        self.stat_decay = stat_decay

        self.inv_freq = inv_freq

        # Timing Variables
        self.timer = Timer(measure=measure_time)

        self.inputs = [0] * len(self.modules)
        self.input_shapes = [0] * len(self.modules)
        self.inputs_reduced = False
        self.grads = [0] * len(self.modules)
        self.grad_shapes = [0] * len(self.modules)

        self.data_type = torch.float16 if half_precision else torch.float32

        self.stabilization_factor = stabilization_factor

        self.error_average_list = []
        self.error_svd_list = []

        self.grad_accum_steps = grad_accum_steps
        self.accumulated_steps = [0] * len(self.modules)

        self.clipping_value = clipping_value

        self.sgd_layers = sgd_layers

        self.warmup_steps = warmup_steps

        self._set_optimizer(optimizer, **kwargs)

        self.grad_scale = grad_scale

    def _set_optimizer(self, optimizer, **kwargs):
        """
        Set the optimizer for the MKOR optimizer.
        :param optimizer: str or torch.optim.Optimizer
        """
        if type(optimizer) == str:
            if optimizer == 'sgd':
                self.optimizer = optim.SGD(self.param_groups, **kwargs)
            elif optimizer == 'adam':
                self.optimizer = optim.Adam(self.param_groups, **kwargs)
            elif optimizer == 'adamw':
                self.optimizer = optim.AdamW(self.param_groups, **kwargs)
            else:
                raise ValueError("Invalid optimizer: {}".format(optimizer))
        else:
            self.optimizer = optimizer

        self.param_groups = self.optimizer.param_groups

    def _save_input(self, module, input):
        """
        Hook for saving the inputs of the module for computing the Fisher matrix later.
        :param input: tuple of input tensors
        """
        if torch.is_grad_enabled() and (self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps):
            # Convert to float32 to avoid overflow during accumulation
            a = torch.clamp(self.AHandler(input[0].data, module).to(torch.float32) / self.grad_accum_steps,
                            -self.clipping_value, self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.inputs[self.index(module)] = a.to(self.data_type)
            else:
                self.inputs[self.index(module)] += a.to(self.data_type)

    def _save_grad_output(self, module, grad_input, grad_output):
        """
        Hook for saving the grad_output of the module for computing the Fisher matrix later.
        :param grad_input: tuple of gradients w.r.t the inputs of the module
        :param grad_output: tuple of gradients w.r.t the outputs of the module
        """
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            # If it's the first module that runs backward pass, we can start synchronizing the inputs
            if not self.inputs_reduced and \
                    self.accumulated_steps[self.index(module)] % self.grad_accum_steps == self.grad_accum_steps - 1:
                self.inputs = self.low_rank_approx(self.inputs)
                self.reduced_inputs, self.input_handles, self.input_shapes = self._reduce_data(self.inputs)
                self.inputs_reduced = True
            g, _ = self.GHandler(grad_output[0].data, module)
            # Convert to float32 to avoid overflow during accumulation
            g = torch.clamp(g.to(torch.float32) / (self.grad_accum_steps * self.grad_scale), -self.clipping_value,
                            self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.grads[self.index(module)] = g.to(self.data_type)
            else:
                self.grads[self.index(module)] += g.to(self.data_type)
            self.accumulated_steps[self.index(module)] += 1

    def _prepare_model(self, sgd_layers=[]):
        """Store the layers that we want to compute the Fisher matrix for.
        :param sgd_layers: List of layers that don't need to be preconditioned."""
        self.apply_sgd = []
        self.index_dict = {}
        count = 0
        if self.verbose:
            print("=> We keep following layers in MKOR. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in MKOR. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                self.index_dict[module] = count
                self.apply_sgd.append(module in sgd_layers)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                if self.verbose:
                    print('(%s): %s' % (count, module))
                count += 1

    def index(self, module):
        """
        Returns the corresponding index to the module.
        :param module: the layer
        :return: the index of the layer
        """
        return self.index_dict[module]

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _dense_precondition(self, module, grad_mat):
        """
        Preconditions the gradient using the Fisher matrix.
        :param module: the layer
        :param grad_mat: the gradients in matrix form
        """
        return self.GG_inv[self.index(module)].to(torch.float32) @ grad_mat @ self.AA_inv[self.index(module)].to(
            torch.float32)

    def _get_natural_grad(self, m, p_grad_mat, identity=False):
        """
        Preconditions the gradients of the weights and biases for the layer.
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        if identity:
            v = p_grad_mat
        else:
            v = self._dense_precondition(m, p_grad_mat)
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _update_grad(self, updates):
        """
        Update the gradients for all the layers by preconditioning them and resets the norm of the preconditioned
        gradients to their initial value.
        :param updates: the preconditioned gradients
        """
        # Reset Norms
        norm_fixer = {}
        for m in self.modules:
            grad_norm = m.weight.grad.data.norm(2)
            update_norm = updates[self.index(m)][0].norm(2)
            if m.bias is not None:
                grad_norm += m.bias.grad.data.norm(2)
                update_norm += updates[self.index(m)][1].norm(2)
            norm_fixer[self.index(m)] = grad_norm / update_norm
            # If gradient is zero, we skip the update
            if torch.isnan(norm_fixer[self.index(m)]):
                continue
            v = updates[self.index(m)]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(norm_fixer[self.index(m)])
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(norm_fixer[self.index(m)])

    def _reduce_data(self, data_dict):
        """
        Reduce the data across the workers.
        :param data_dict: the data to be reduced
        :return: the reduced data, the handles for the allreduce operation, and the shapes of the data
        """
        if self.backend.size() == 1:
            return None, None, None
        data_list = []
        data_shapes = []
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            # Convert the data to a vector and append them to a list to utilize the bandwidth better
            data_list.append(data_dict[self.index(module)].reshape(1, -1))
            data_shapes.append(data_dict[self.index(module)].shape)
        reduced_data = torch.cat(data_list, dim=1)
        del data_list
        handles = []
        handles.append(self.backend.allreduce(reduced_data, async_op=True, average=True))
        return reduced_data, handles, data_shapes

    def low_rank_approx(self, data_dict):
        """
        Compute the low rank approximation of the data by averaging through the batch dimension.
        :param data_dict: the data to be approximated
        """
        for module in self.modules:
            a = data_dict[self.index(module)]
            a = torch.mean(a, dim=0, keepdim=True)
            data_dict[self.index(module)] = a
        return data_dict

    def sync_data(self, synchronized_data, data_dict, data_shapes, handles):
        """
        Synchronize the data across the workers.
        :param synchronized_data: the synchronized data
        :param data_dict: the original data
        :param data_shapes: the shapes of the original data
        :param handles: the handles for the allreduce operation
        """
        if self.backend.size() == 1:
            return

        self.backend.sync(handles)
        offset = 0
        i = 0
        # Convert the synchronized data to the original shape
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            data_shape = data_shapes[i]
            data_numel = torch.prod(torch.tensor(data_shapes[i])).item()
            data_dict[self.index(module)] = synchronized_data[0, offset:offset + data_numel].reshape(data_shape)
            offset += data_numel
            i += 1

    def _update_inv_factors(self, rank_1_dict, factor_dict, reset_factor_dict):
        """
        Update the inverse factors for the preconditioning by resetting or incorporating low-rank approximations.
        :param rank_1_dict: the rank 1 matrices
        :param factor_dict: the factors
        :param reset_factor_dict: the reset flags for the factors
        :param original_factor_dict: the original factors
        """
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            a = rank_1_dict[self.index(module)]
            v = a.t()
            if reset_factor_dict[self.index(module)]:
                self._reset_factors(module, factor_dict, a.size(1), reset_factor_dict)
            else:
                self._invert_factor(module, factor_dict, v)


    def _invert_factor(self, module, factor_dict, v):
        """
        Invert the factor for the preconditioning using the Sherman-Morrison formula and set the reset flags.
        :param module: the layer
        :param factor_dict: the factors
        :param v: the rank 1 matrix
        """
        shape = factor_dict[self.index(module)].shape
        factor_dict[self.index(module)] = sm_inverse(
            0.95 * factor_dict[self.index(module)] + 0.05 * torch.eye(shape[0], device=self.device,
                                                                      dtype=self.data_type),
            v)
        self._set_reset_factor_flags(module, factor_dict)

    def _reset_factors(self, module, factor_dict, dim, reset_factor_dict):
        """
        Reset the factors for the preconditioning and set the reset flags.
        :param module: the layer
        :param factor_dict: the factors
        :param dim: the dimension of the factor
        :param reset_factor_dict: the reset flags
        """
        if module not in factor_dict:
            factor_dict[self.index(module)] = torch.eye(dim, device=self.device, dtype=self.data_type)
        else:
            factor_dict[self.index(module)] = factor_dict[self.index(module)] * (
                    1 - self.stabilization_factor) + self.stabilization_factor * torch.eye(
                dim, device=self.device, dtype=self.data_type)
        reset_factor_dict[self.index(module)] = False

    def _set_reset_factor_flags(self, module, factor_dict):
        """
        Set the reset flags for the factors.
        :param module: the layer
        :param factor_dict: the factors
        """
        if torch.max(torch.abs(factor_dict[self.index(module)].flatten())) > 2:
            self.a_reset_factor[self.index(module)] = True
            self.g_reset_factor[self.index(module)] = True

    def _reduce_and_update_factors(self):
        """
        Reduce the inputs and gradients and update the inverse factors for the preconditioning.
        """
        self.grads = self.low_rank_approx(self.grads)

        # To overlap computation and communication, the input reduction has already started
        # We first reduce the gradients and update the inverse input factors for the preconditioning
        # Then we invert gradient factors for the preconditioning
        self.reduced_grads, self.grad_handles, self.grad_shapes = self._reduce_data(self.grads)
        self.sync_data(self.reduced_inputs, self.inputs, self.input_shapes, self.input_handles)
        self.inputs_reduced = False
        self._update_inv_factors(self.inputs, self.AA_inv, self.a_reset_factor)

        self.sync_data(self.reduced_grads, self.grads, self.grad_shapes, self.grad_handles)

        self._update_inv_factors(self.grads, self.GG_inv, self.g_reset_factor)

    def step(self, closure=None):
        """
        Perform a single optimization step.
        :param closure: A closure that reevaluates the model and returns the loss.
        """
        updates = {}
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            self.timer("reduce_and_update_factors", self._reduce_and_update_factors)
        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self.timer("precondition", self._get_matrix_form_grad, m=m, classname=classname)
            v = self.timer("precondition", self._get_natural_grad, m=m, p_grad_mat=p_grad_mat,
                           identity=self.apply_sgd[self.index(m)])
            updates[self.index(m)] = v

        self.timer("update_weights", self._update_grad, updates=updates)

        self.timer("update_weights", self.optimizer.step, closure=closure)
        self.steps += 1

    def update_grad_scale(self, scaler):
        """
        Update the gradient scale.
        :param scaler: the new gradient scale
        """
        self.grad_scale = scaler

    def state_dict(self):
        """
        Returns the state of the optimizer as a :class:`dict`.
        """
        return {
            "grad_outputs": self.grad_outputs,
            "a_reset_factor": self.a_reset_factor,
            "g_reset_factor": self.g_reset_factor,
            "steps": self.steps,
            "AA_inv": self.AA_inv,
            "GG_inv": self.GG_inv,
            "AA": self.AA,
            "GG": self.GG,
            "stat_decay": self.stat_decay,
            "inv_freq": self.inv_freq,
            "inputs": self.inputs,
            "input_shapes": self.input_shapes,
            "grads": self.grads,
            "grad_shapes": self.grad_shapes,
            "data_type": self.data_type,
            "stabilization_factor": self.stabilization_factor,
            "grad_accum_steps": self.grad_accum_steps,
            "accumulated_steps": self.accumulated_steps,
            "clipping_value": self.clipping_value,
            "sgd_layers": self.sgd_layers,
            "warmup_steps": self.warmup_steps,
            "optimizer": self.optimizer.state_dict(),
            "grad_scale": self.grad_scale,
            "apply_sgd": self.apply_sgd,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        :param state_dict: the state of the optimizer
        """
        if len(state_dict) == 2:
            return
        for key in state_dict:
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict[key])
                self.param_groups = self.optimizer.param_groups
            else:
                if type(state_dict[key]) == list:
                    for i in range(len(state_dict[key])):
                        if type(state_dict[key][i]) == torch.Tensor:
                            state_dict[key][i] = state_dict[key][i].to(self.device)
                setattr(self, key, state_dict[key])