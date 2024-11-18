import math

import torch
import torch.optim as optim
import numpy as np

from .utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.timing import Timer
from .utils.factors import ComputeI, ComputeG
from .utils.hylo_utils import EmptyBackend


class MKOROptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 inv_freq=10,
                 measure_time=False,
                 svd=False,
                 backend=EmptyBackend(),
                 half_precision=True,
                 grad_accum_steps=1,
                 sgd_layers=[],
                 distribute_factorizatoin=False,
                 distribute_preconditioning=False,
                 optimizer='sgd',
                 grad_scale=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): MKOR optimizer now only support model as input
        super(MKOROptimizer, self).__init__(model.parameters(), defaults)
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
        self.Q_a, self.Q_g = [0] * len(self.modules), [0] * len(self.modules)
        self.d_a, self.d_g = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_inv, self.GG_inv = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_Us, self.AA_Vt = [0] * len(self.modules), [0] * len(self.modules)
        self.GG_Us, self.GG_Vt = [0] * len(self.modules), [0] * len(self.modules)
        self.AA_sparse_factor, self.GG_sparse_factor = [0] * len(self.modules), [0] * len(self.modules)
        self.AA, self.GG = [0] * len(self.modules), [0] * len(self.modules)
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.inv_freq = inv_freq

        # Timing Variables
        self.timer = Timer(measure=measure_time)

        self.svd = svd

        self.reset_factors_freq = 10

        self.inputs = [0] * len(self.modules)
        self.input_shapes = [0] * len(self.modules)
        self.inputs_reduced = False
        self.grads = [0] * len(self.modules)
        self.grad_shapes = [0] * len(self.modules)

        self.data_type = torch.float16 if half_precision else torch.float32

        self.manual_reset_factors = False

        if self.manual_reset_factors:
            self.reset_weight = 0.8
        else:
            self.reset_weight = 0.1

        self.error_average_list = []
        self.error_svd_list = []

        self.rank = 1
        self.sparse = False
        self.sparse_threshold = 5e-3
        self.sparse_AA, self.sparse_GG = [0] * len(self.modules), [0] * len(self.modules)

        self.dummy_timer_start = torch.cuda.Event(enable_timing=True)
        self.dummy_timer_end = torch.cuda.Event(enable_timing=True)
        self.grad_accum_steps = grad_accum_steps
        self.accumulated_steps = [0] * len(self.modules)

        if self.sparse:
            self.compute_sparse_preconditioning_costs()

        self.distribute_factorization = distribute_factorizatoin
        # if distribute_factorizatoin:
        #     self.compute_factorization_costs()

        self.distribute_preconditioning = distribute_preconditioning
        # if distribute_preconditioning:
        #     self.compute_distributed_preconditioning_costs()

        if self.distribute_factorization and self.distribute_preconditioning:
            self.compute_distribution_costs()

        self.clipping_value = 100.0

        self.method = 'approx'

        self.sgd_layers = sgd_layers

        self.warmup_steps = 0

        self.set_optimizer(optimizer)

        self.grad_scale = grad_scale

    def set_optimizer(self, optimizer):
        if type(optimizer) == str:
            if optimizer == 'sgd':
                self.optimizer = optim.SGD(self.param_groups, lr=self.defaults['lr'], weight_decay=self.defaults['weight_decay'], momentum=self.defaults['momentum'])
            elif optimizer == 'adam':
                self.optimizer = optim.Adam(self.param_groups, lr=self.defaults['lr'], weight_decay=self.defaults['weight_decay'])
            else:
                raise ValueError("Invalid optimizer: {}".format(optimizer))
        else:
            self.optimizer = optimizer

        self.param_groups = self.optimizer.param_groups

    def compute_factorization_costs(self):
        worker_costs = [0] * self.backend.size()
        self.input_factorization_worker_assignments = {}
        self.grad_factorization_worker_assignments = {}
        self.worker_preconditioning_assignments = {}
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            input_size = self.get_input_size(module)
            grad_size = self.get_grad_size(module)
            input_factorizatoin_cost, grad_factorization_cost = input_size ** 2, grad_size ** 2
            preconditioning_cost = input_size * grad_size * (input_size + grad_size)
            min_idx = np.argmin(worker_costs)
            worker_costs[min_idx] += input_factorizatoin_cost + grad_factorization_cost + preconditioning_cost
            self.input_factorization_worker_assignments[self.index(module)] = min_idx
            self.grad_factorization_worker_assignments[self.index(module)] = min_idx
            self.worker_preconditioning_assignments[self.index(module)] = min_idx
        if self.verbose:
            # print(self.worker_factorization_assignments)
            print(worker_costs)

    def compute_factorization_costs(self):
        worker_costs = [0] * self.backend.size()
        self.input_factorization_worker_assignments = {}
        self.grad_factorization_worker_assignments = {}
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            input_size = self.get_input_size(module)
            grad_size = self.get_grad_size(module)
            input_cost, grad_cost = input_size ** 2, grad_size ** 2
            min_idx = np.argmin(worker_costs)
            worker_costs[min_idx] += input_cost
            self.input_factorization_worker_assignments[self.index(module)] = min_idx
            min_idx = np.argmin(worker_costs)
            worker_costs[min_idx] += grad_cost
            self.grad_factorization_worker_assignments[self.index(module)] = min_idx
        if self.verbose:
            # print(self.worker_factorization_assignments)
            print(worker_costs)

    def compute_distributed_preconditioning_costs(self):
        worker_costs = [0] * self.backend.size()
        self.worker_preconditioning_assignments = {}
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            input_size = self.get_input_size(module)
            grad_size = self.get_grad_size(module)
            cost, grad_cost = input_size * grad_size * (input_size + grad_size)
            min_idx = np.argmin(worker_costs)
            worker_costs[min_idx] += cost
            self.worker_preconditioning_assignments[(module, 'input')] = min_idx
        if self.verbose:
            # print(self.worker_preconditioning_assignments)
            print(worker_costs)

    def inverse(self, prev_inv, rank_1):
        tmp1 = (prev_inv @ rank_1)
        tmp2 = (rank_1.t() @ prev_inv)
        return prev_inv - 1 / (1 + tmp2 @ rank_1) * tmp1 @ tmp2

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and (self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps):
            a = torch.clamp(self.AHandler(input[0].data, module).to(torch.float32) / self.grad_accum_steps, -self.clipping_value, self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.inputs[self.index(module)] = a.to(self.data_type)
            else:
                self.inputs[self.index(module)] += a.to(self.data_type)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            if not self.inputs_reduced and \
                    self.accumulated_steps[self.index(module)] % self.grad_accum_steps == self.grad_accum_steps - 1:
                self.inputs = self.low_rank_approx(self.inputs)
                self.reduced_inputs, self.input_handles, self.input_shapes = self.reduce_data(
                    self.inputs, self.input_factorization_worker_assignments if self.distribute_factorization else None)
                self.inputs_reduced = True
            g, _ = self.GHandler(grad_output[0].data, module)
            g = torch.clamp(g.to(torch.float32) / (self.grad_accum_steps * self.grad_scale), -self.clipping_value, self.clipping_value)
            if self.accumulated_steps[self.index(module)] % self.grad_accum_steps == 0:
                self.grads[self.index(module)] = g.to(self.data_type)
            else:
                self.grads[self.index(module)] += g.to(self.data_type)
            self.accumulated_steps[self.index(module)] += 1

    def _prepare_model(self, sgd_layers=[]):
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
        return self.index_dict[module]

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[self.index(m)], self.Q_a[self.index(m)] = torch.linalg.eigh(
            self.m_aa[self.index(m)] + eps / 10 * torch.eye(self.m_aa[self.index(m)].shape[0], device=self.m_aa[self.index(m)].device))
        self.d_g[self.index(m)], self.Q_g[self.index(m)] = torch.linalg.eigh(
            self.m_gg[self.index(m)] + eps / 10 * torch.eye(self.m_gg[self.index(m)].shape[0], device=self.m_gg[self.index(m)].device))

        # print(min(torch.min(self.d_a[self.index(m)]), torch.min(self.d_g[self.index(m)])))

        self.d_a[self.index(m)].mul_((self.d_a[self.index(m)] > eps).float())
        self.d_g[self.index(m)].mul_((self.d_g[self.index(m)] > eps).float())

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

    def sparse_precondition(self, module, grad_mat):
        if self.GG_sparse_factor[self.index(module)] is None and self.AA_sparse_factor[self.index(module)] is None:
            return grad_mat
        elif self.GG_sparse_factor[self.index(module)] is None:
            return grad_mat @ self.AA_sparse_factor[self.index(module)]
        elif self.AA_sparse_factor[self.index(module)] is None:
            return self.GG_sparse_factor[self.index(module)] @ grad_mat
        else:
            return self.GG_sparse_factor[self.index(module)] @ grad_mat @ self.AA_sparse_factor[self.index(module)]

    def dense_precondition(self, module, grad_mat):
        return self.GG_inv[self.index(module)].to(torch.float32) @ grad_mat @ self.AA_inv[self.index(module)].to(torch.float32)

    def _get_natural_grad(self, m, p_grad_mat, identity=False):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        # v1 = self.Q_g[self.index(m)].t() @ p_grad_mat @ self.Q_a[self.index(m)]
        # v2 = v1 / (self.d_g[self.index(m)].unsqueeze(1) * self.d_a[self.index(m)].unsqueeze(0) + damping)
        # v = self.Q_g[self.index(m)] @ v2 @ self.Q_a[self.index(m)].t()
        if identity:
            v = p_grad_mat
        else:
            # self.dummy_timer_start.record()
            if self.rank == 1:
                # print(p_grad_mat.shape, m)
                if self.sparse:
                    v = self.sparse_precondition(m, p_grad_mat)
                else:
                    v = self.dense_precondition(m, p_grad_mat)
            else:
                v = self.rank * (self.GG_Us[self.index(m)] @ (self.GG_Vt[self.index(m)] @ p_grad_mat @ self.AA_Us[self.index(m)]) @ self.AA_Vt[self.index(m)]) + (
                            1 - self.rank) * p_grad_mat
            # self.dummy_timer_end.record()
            # torch.cuda.synchronize()
            # print('dummy time: ', self.dummy_timer_start.elapsed_time(self.dummy_timer_end))
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # # do kl clip
        # vg_sum = 0
        # for m in self.modules:
        #     v = updates[self.index(m)]
        #     vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
        #     if m.bias is not None:
        #         vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        # for m in self.modules:
        #     v = updates[self.index(m)]
        #     m.weight.grad.data.copy_(v[0])
        #     m.weight.grad.data.mul_(nu)
        #     if m.bias is not None:
        #         m.bias.grad.data.copy_(v[1])
        #         m.bias.grad.data.mul_(nu)

        # Reset Norms
        norm_fixer = {}
        for m in self.modules:
            grad_norm = m.weight.grad.data.norm(2)
            update_norm = updates[self.index(m)][0].norm(2)
            if m.bias is not None:
                grad_norm += m.bias.grad.data.norm(2)
                update_norm += updates[self.index(m)][1].norm(2)
            norm_fixer[self.index(m)] = grad_norm / update_norm
            if torch.isnan(norm_fixer[self.index(m)]): #Gradient is zero
                # if self.verbose:
                #     print("Gradient is zero for module: ", m)
                continue
            v = updates[self.index(m)]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(norm_fixer[self.index(m)])
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(norm_fixer[self.index(m)])

    def reduce_data(self, data_dict, distributed=False, distribution_dict=None):
        if self.backend.size() == 1:
            return None, None, None
        if distributed:
            data_list = [[] for _ in range(self.backend.size())]
            data_shapes = [[] for _ in range(self.backend.size())]
            for module in self.modules:
                if self.apply_sgd[self.index(module)]:
                    continue
                data_list[distribution_dict[self.index(module)]].append(data_dict[self.index(module)].reshape(1, -1))
                data_shapes[distribution_dict[self.index(module)]].append(data_dict[self.index(module)].shape)
            reduced_data = [torch.cat(data_list[i], dim=1) for i in range(self.backend.size())]
            del data_list
            handles = []
            for i in range(self.backend.size()):
                handles.extend(self.backend.reduce(reduced_data[i], dst=i, async_op=True, average=True))
            return reduced_data, handles, data_shapes
        else:
            data_list = []
            data_shapes = []
            for module in self.modules:
                if self.apply_sgd[self.index(module)]:
                    continue
                data_list.append(data_dict[self.index(module)].reshape(1, -1))
                data_shapes.append(data_dict[self.index(module)].shape)
            reduced_data = torch.cat(data_list, dim=1)
            del data_list
            handles = []
            handles.append(self.backend.allreduce(reduced_data, async_op=True, average=True))
            return reduced_data, handles, data_shapes

    def low_rank_approx(self, data_dict):
        for module in self.modules:
            a = data_dict[self.index(module)]
            if self.svd:
                U, S, V = torch.linalg.svd(a, full_matrices=False)
                a = (V[0, :].reshape(-1, 1) * S[0] * torch.sum(U[:, 0] ** 2)).t() / torch.sqrt(torch.tensor(a.shape[0]))

            else:
                # if self.verbose:
                #     print("Before Averaging:", a.abs().max())
                a = torch.mean(a, dim=0, keepdim=True)
                # if self.verbose:
                #     print("After Averaging:", a.abs().max())

            data_dict[self.index(module)] = a
        return data_dict

    def sync_data(self, synchronized_data, data_dict, data_shapes, handles, distributed_computation=False,
                  distribution_dict=None):
        if self.backend.size() == 1:
            return

        self.backend.sync(handles)

        if distributed_computation:
            rank = self.backend.rank()
            offset = 0
            i = 0
            for module in self.modules:
                if self.apply_sgd[self.index(module)] or rank != distribution_dict[self.index(module)]:
                    continue
                data_shape = data_shapes[rank][i]
                data_numel = torch.prod(torch.tensor(data_shape)).item()
                data_dict[self.index(module)] = synchronized_data[0, offset:offset + data_numel].reshape(data_shape)
                offset += data_numel
        else:
            offset = 0
            i = 0
            for module in self.modules:
                if self.apply_sgd[self.index(module)]:
                    continue
                data_shape = data_shapes[i]
                data_numel = torch.prod(torch.tensor(data_shapes[i])).item()
                data_dict[self.index(module)] = synchronized_data[0, offset:offset + data_numel].reshape(data_shape)
                offset += data_numel
                i += 1

    def update_inv_factors(self, rank_1_dict, factor_dict, reset_factor_dict, original_factor_dict, low_rank_Us_dict,
                           low_rank_Vt_dict, sparsity_dict, sparse_factor_dict, distributed_rank_dict=None):
        rank = self.backend.rank()
        for module in self.modules:
            if self.apply_sgd[self.index(module)] or (self.distribute_factorization and self.distribute_preconditioning
                                          and distributed_rank_dict[self.index(module)] != rank):
                continue
            a = rank_1_dict[self.index(module)]
            v = a.t()
            if reset_factor_dict[self.index(module)]:
                self.reset_factors(module, factor_dict, a.size(1), reset_factor_dict)

            else:
                self.invert_factor(module, original_factor_dict, factor_dict, a, v)
                if self.rank != 1:
                    low_rank_Us_dict[self.index(module)], low_rank_Vt_dict[self.index(module)] = self.randomized_svd(
                        factor_dict[self.index(module)].to(torch.float32), self.rank)
                else:
                    if self.sparse:
                        self.sparsify_factor(module, factor_dict, sparsity_dict, sparse_factor_dict)

    def invert_factor(self, module, original_factor_dict, factor_dict, a, v):
        if self.method == 'exact':
            factor_dict[self.index(module)] = torch.inverse(
                a.t() @ a * (1 - self.stat_decay) + self.stat_decay * original_factor_dict[self.index(module)])
        elif self.method == 'low_rank':
            factor_dict[self.index(module)] = torch.inverse(
                v @ v.t() * (1 - self.stat_decay) + self.stat_decay * original_factor_dict[self.index(module)])
        elif self.method == 'approx':
            shape = factor_dict[self.index(module)].shape
            factor_dict[self.index(module)] = self.inverse(0.95 * factor_dict[self.index(module)] + 0.05 * torch.eye(shape[0], device=self.device, dtype=self.data_type),
                                                           v)
            self.set_reset_factor_flags(module, factor_dict)

    def reset_factors(self, module, factor_dict, dim, reset_factor_dict):
        if module not in factor_dict:
            factor_dict[self.index(module)] = torch.eye(dim, device=self.device, dtype=self.data_type)
        else:
            factor_dict[self.index(module)] = factor_dict[self.index(module)] * (1 - self.reset_weight) + self.reset_weight * torch.eye(
                dim, device=self.device, dtype=self.data_type)
        reset_factor_dict[self.index(module)] = False

    def set_reset_factor_flags(self, module, factor_dict):
        if self.manual_reset_factors:
            self.a_reset_factor[self.index(module)] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
            self.g_reset_factor[self.index(module)] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
        else:
            if torch.max(torch.abs(factor_dict[self.index(module)].flatten())) > 2:
                self.a_reset_factor[self.index(module)] = True
                self.g_reset_factor[self.index(module)] = True

    def sparsify_factor(self, module, factor_dict, sparsity_dict, sparse_factor_dict):
        if self.sparsify[self.index(module)]:
            mask = (factor_dict[self.index(module)].abs() > self.sparse_threshold).to(torch.int)
            density_ratio = mask.sum() / mask.numel()
            sparsity_dict[self.index(module)] = True
            if density_ratio < 2 / factor_dict[self.index(module)].size(0):
                sparse_factor_dict[self.index(module)] = None
            elif density_ratio < self.right_thresholds[mask.shape[0]]:
                sparse_factor_dict[self.index(module)] = (factor_dict[self.index(module)] * mask).to(
                    torch.float32).to_sparse_csr().to(factor_dict[self.index(module)].device)
            else:
                sparse_factor_dict[self.index(module)] = factor_dict[self.index(module)].to(torch.float32)
        else:
            sparsity_dict[self.index(module)] = False
            sparse_factor_dict[self.index(module)] = factor_dict[self.index(module)].to(torch.float32)

    def reduce_and_update_factors(self):
        self.grads = self.low_rank_approx(self.grads)

        self.reduced_grads, self.grad_handles, self.grad_shapes = self.reduce_data(
            self.grads, self.grad_factorization_worker_assignments if self.distribute_factorization else None)
        self.sync_data(self.reduced_inputs, self.inputs, self.input_shapes, self.input_handles)
        self.inputs_reduced = False
        self.update_inv_factors(self.inputs, self.AA_inv, self.a_reset_factor, self.AA, self.AA_Us, self.AA_Vt,
                           self.sparse_AA, self.AA_sparse_factor)

        self.sync_data(self.reduced_grads, self.grads, self.grad_shapes, self.grad_handles)

        self.update_inv_factors(self.grads, self.GG_inv, self.g_reset_factor, self.GG, self.GG_Us, self.GG_Vt,
                                self.sparse_GG, self.GG_sparse_factor)

    def compute_min_eigenvals(self):
        self.a_min_eigenvals = {}
        self.g_min_eigenvals = {}
        self.a_rank = {}
        self.g_rank = {}
        density_ratios_a = 0
        density_ratios_g = 0
        average_a_rank = 0
        average_g_rank = 0
        num_mkor_layers = 0
        for module in self.modules:
            if self.apply_sgd[self.index(module)]:
                continue
            num_mkor_layers += 1
            norm = torch.norm(self.AA_inv[self.index(module)] - torch.eye(self.AA_inv[self.index(module)].size(0)).cuda()) / torch.numel(
                self.AA_inv[self.index(module)])
            print(norm)
            if norm < 1e-6:
                inv_factor = self.AA_inv[self.index(module)].to(torch.float32)
            else:
                inv_factor = (self.AA_inv[self.index(module)] - torch.eye(self.AA_inv[self.index(module)].size(0)).cuda()).to(torch.float64)
            density_ratios_a += torch.sum(
                torch.abs(self.AA_inv[self.index(module)] - torch.eye(self.AA_inv[self.index(module)].size(0)).cuda()) > 5e-2) / torch.numel(
                self.AA_inv[self.index(module)])
            density_ratios_g += torch.sum(
                torch.abs(self.GG_inv[self.index(module)] - torch.eye(self.GG_inv[self.index(module)].size(0)).cuda()) > 5e-2) / torch.numel(
                self.GG_inv[self.index(module)])
            # if torch.all(inv_factor == torch.zeros(inv_factor.size()).cuda()):
            #     inv_factor = self.AA_inv[self.index(module)].to(torch.float32)
            d_a, Q_a = torch.linalg.eigh(inv_factor)
            d_g, Q_g = torch.linalg.eigh(self.GG_inv[self.index(module)].to(torch.float32))

            self.a_min_eigenvals[self.index(module)] = torch.min(d_a)
            self.g_min_eigenvals[self.index(module)] = torch.min(d_g)

            d_a_sorted, _ = torch.sort(d_a)
            d_g_sorted, _ = torch.sort(d_g)
            a_energy = torch.cumsum(d_a_sorted, dim=0)
            g_energy = torch.cumsum(d_g_sorted, dim=0)
            self.a_rank[self.index(module)] = torch.argmax((a_energy > 0.7 * a_energy[-1]).to(torch.float32)) + 1
            self.g_rank[self.index(module)] = torch.argmax((g_energy > 0.7 * g_energy[-1]).to(torch.float32)) + 1

            average_a_rank += self.a_rank[self.index(module)].item() / len(d_a)
            average_g_rank += self.g_rank[self.index(module)].item() / len(d_g)

            if self.verbose:
                import os
                if not os.path.exists("./results/rank.csv"):
                    with open("./results/rank.csv", "w") as f:
                        f.write("a_rank,a_dim,g_rank,g_dim\n")
                with open("./results/rank.csv", "a") as f:
                    f.write(str(self.a_rank[self.index(module)].item()) + "," + str(len(d_a)) + "," + str(
                        self.g_rank[self.index(module)].item()) + "," + str(len(d_g)) + "\n")
        average_a_rank /= num_mkor_layers
        average_g_rank /= num_mkor_layers
        density_ratios_a /= num_mkor_layers
        density_ratios_g /= num_mkor_layers
        if self.verbose:
            print("Average a rank:", average_a_rank)
            print("Average g rank:", average_g_rank)
            print("Average a density ratio:", density_ratios_a)
            print("Average g density ratio:", density_ratios_g)

            # if self.verbose:
            #     import os
            #     if not os.path.exists("a_eigen.csv"):
            #         with open("a_eigen.csv", "w") as f:
            #             f.write("min_a_eigen,min_a_abs_eigen,max_a_eigen,max_a_abs_eigen,condition_number\n")
            #     with open("a_eigen.csv", "a") as f:
            #         f.write(str(torch.min(d_a).item()) + "," + str(torch.min(torch.abs(d_a)).item()) + "," + str(torch.max(d_a).item()) + "," + str(torch.max(torch.abs(d_a)).item()) + "," + str(torch.max(torch.abs(d_a)).item() / torch.min(torch.abs(d_a)).item()) + "\n")
            #     if not os.path.exists("g_eigen.csv"):
            #         with open("g_eigen.csv", "w") as f:
            #             f.write("min_g_eigen,min_g_abs_eigen,max_g_eigen,max_g_abs_eigen,condition_number\n")
            #     with open("g_eigen.csv", "a") as f:
            #         f.write(str(torch.min(d_g).item()) + "," + str(torch.min(torch.abs(d_g)).item()) + "," + str(torch.max(d_g).item()) + "," + str(torch.max(torch.abs(d_g)).item()) + "," + str(torch.max(torch.abs(d_g)).item() / torch.min(torch.abs(d_g)).item()) + "\n")

            # print(min(torch.min(self.d_a[self.index(m)]), torch.min(self.d_g[self.index(m)])))

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        updates = {}
        if self.steps % self.inv_freq == 0 or self.steps < self.warmup_steps:
            self.timer("reduce_and_update_factors", self.reduce_and_update_factors)
            # self.compute_min_eigenvals()
        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self.timer("precondition", self._get_matrix_form_grad, m=m, classname=classname)
            v = self.timer("precondition", self._get_natural_grad, m=m, p_grad_mat=p_grad_mat,
                           identity=self.apply_sgd[self.index(m)])
            updates[self.index(m)] = v

        self.timer("update_weights", self._kl_clip_and_update_grad, updates=updates, lr=lr)

        self.timer("update_weights", self.optimizer.step, closure=closure)
        #for param_group in self.param_groups:
        #    if not 'step' in param_group:
        #        param_group['step'] = 0
        #    else:
        #        param_group['step'] += 1
        self.steps += 1

    def randomized_svd(self, B, rank):
        if rank < 1:
            rank = int(rank * min(B.size()))
        m, n = B.size()
        rand_matrix = torch.rand((n, rank)).to(B.device)  # short side by k
        Q, _ = torch.linalg.qr(B @ rand_matrix)  # long side by k
        smaller_matrix = (Q.transpose(0, 1) @ B)  # k by short side
        U_hat, s, V = torch.svd(smaller_matrix, False)
        U = (Q @ U_hat)
        Us = U @ torch.diag(s)
        Vt = V[:, :rank].t()
        return Us, Vt

    def compute_sparse_preconditioning_costs(self):
        costs = []
        self.sparsify = {}
        self.left_thresholds = {}
        self.right_thresholds = {}
        for m in self.modules:
            if self.apply_sgd[self.index(m)]:
                costs.append(0)
                self.sparsify[self.index(m)] = False
                continue
            input_size = self.get_input_size(m)
            grad_size = self.get_grad_size(m)
            costs.append(input_size * grad_size * (input_size + grad_size))
            self.sparsify[self.index(m)] = False
        costs = torch.tensor(costs)
        self.costs = costs / torch.sum(costs)
        # copy_costs = self.costs.clone().detach()
        sum = 0
        while sum < 0.7:
            max, idx = torch.max(self.costs, 0)
            module = self.modules[idx]
            self.costs[idx] = 0
            sum += max
            self.sparsify[self.index(module)] = True
            input_size = int(self.get_input_size(module))
            grad_size = int(self.get_grad_size(module))
            # print("Added: ", self.modules[idx], input_size, grad_size)
            self.right_thresholds[input_size] = 0
            self.left_thresholds[grad_size] = 0

        device = self.model.device

        if self.verbose:
            left_thresholds = torch.zeros(len(self.left_thresholds.keys()), 2).to(device)
            right_thresholds = torch.zeros(len(self.right_thresholds.keys()), 2).to(device)
            i = 0
            for size in self.right_thresholds:
                right_thresholds[i, 0] = size
                right_thresholds[i, 1] = self.compute_threshold(size, left=False)
                i += 1
            i = 0
            for size in self.left_thresholds:
                left_thresholds[i, 0] = size
                left_thresholds[i, 1] = self.compute_threshold(size, left=True)
                i += 1
        else:
            left_thresholds = torch.empty(len(self.left_thresholds.keys()), 2).to(device)
            right_thresholds = torch.empty(len(self.right_thresholds.keys()), 2).to(device)

        self.backend.broadcast(left_thresholds, src=0, async_op=False)
        self.backend.broadcast(right_thresholds, src=0, async_op=False)

        for i in range(left_thresholds.size(0)):
            self.left_thresholds[int(left_thresholds[i, 0])] = left_thresholds[i, 1]
        for i in range(right_thresholds.size(0)):
            self.right_thresholds[int(right_thresholds[i, 0])] = right_thresholds[i, 1]

        # print(self.left_thresholds)
        # print(self.right_thresholds)

    def compute_threshold(self, size, left=True):
        device = self.model.device
        num_experiments = 10
        densities = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        threshold = 0

        dense_time = 0
        dense_start = torch.cuda.Event(enable_timing=True)
        dense_end = torch.cuda.Event(enable_timing=True)
        for i in range(num_experiments):
            dense_mat = torch.rand(size, size).to(device)
            sparse_mat = torch.rand(size, size).to(device)
            dense_start.record()
            result = sparse_mat @ dense_mat
            dense_end.record()
            torch.cuda.synchronize()
            if i > 0:
                dense_time += dense_start.elapsed_time(dense_end)
        for density in densities:
            sparse_start = torch.cuda.Event(enable_timing=True)
            sparse_end = torch.cuda.Event(enable_timing=True)
            sparse_time = 0

            for i in range(num_experiments):
                sparse_mat = (
                        (torch.rand(size, size).to(device) < density).to(torch.float) * torch.rand(size, size).to(
                    device)).to_sparse_csr()
                # print(sparse_mat.shape, sparse_mat._nnz())
                dense_mat = torch.rand(size, size).to(device)
                sparse_start.record()
                if left:
                    result = sparse_mat @ dense_mat
                else:
                    result = dense_mat @ sparse_mat
                sparse_end.record()
                torch.cuda.synchronize()
                if i > 0:
                    sparse_time += sparse_start.elapsed_time(sparse_end)

            print(f"Matrix Dim: {size}, Density: {density}, Sparse Time: {sparse_time}, Dense Time: {dense_time}")
            if sparse_time < dense_time:
                threshold = density
            else:
                break
        print(f"Optimal Threshold: {threshold}")
        return threshold

        # for m in self.modules:
        #     if self.apply_sgd[self.index(m)]:
        #         continue
        #     print(self.inputs[self.index(m)].shape, self.grads[self.index(m)].shape, copy_costs[self.modules.index(m)], self.sparsify[self.index(m)])
        # exit()

    def get_input_size(self, module):
        if isinstance(module, torch.nn.Conv2d):
            return (module.in_channels * torch.prod(torch.tensor(module.kernel_size))).data + int(
                module.bias is not None)
        elif isinstance(module, torch.nn.Linear):
            return module.in_features + int(module.bias is not None)
        else:
            raise NotImplementedError

    def get_grad_size(self, module):
        if isinstance(module, torch.nn.Conv2d):
            return module.in_channels
        elif isinstance(module, torch.nn.Linear):
            return module.out_features
        else:
            raise NotImplementedError

    def update_grad_scale(self, scaler):
        self.grad_scale = scaler

    def state_dict(self):
        return {
            "grad_outputs":self.grad_outputs,
            "a_reset_factor":self.a_reset_factor,
            "g_reset_factor":self.g_reset_factor,
            "steps":self.steps,
            "AA_inv":self.AA_inv,
            "GG_inv":self.GG_inv,
            "AA_Us":self.AA_Us,
            "AA_Vt":self.AA_Vt,
            "GG_Us":self.GG_Us,
            "GG_Vt":self.GG_Vt,
            "AA_sparse_factor":self.AA_sparse_factor,
            "GG_sparse_factor":self.GG_sparse_factor,
            "AA":self.AA,
            "GG":self.GG,
            "stat_decay":self.stat_decay,
            "inv_freq":self.inv_freq,
            "svd":self.svd,
            "reset_factors_freq":self.reset_factors_freq,
            "inputs":self.inputs,
            "input_shapes":self.input_shapes,
            "grads":self.grads,
            "grad_shapes":self.grad_shapes,
            "data_type":self.data_type,
            "manual_reset_factors":self.manual_reset_factors,
            "reset_weight":self.reset_weight,
            "rank":self.rank,
            "sparse":self.sparse,
            "sparse_threshold":self.sparse_threshold,
            "sparse_AA":self.sparse_AA,
            "sparse_GG":self.sparse_GG,
            "grad_accum_steps":self.grad_accum_steps,
            "accumulated_steps":self.accumulated_steps,
            "distribute_factorization":self.distribute_factorization,
            "distribute_preconditioning":self.distribute_preconditioning,
            "clipping_value":self.clipping_value,
            "method":self.method,
            "sgd_layers":self.sgd_layers,
            "warmup_steps":self.warmup_steps,
            "optimizer":self.optimizer.state_dict(),
            "grad_scale":self.grad_scale,
            "apply_sgd":self.apply_sgd,
        }
    
    def load_state_dict(self, state_dict):
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
        print("Checkpoint Loaded")
