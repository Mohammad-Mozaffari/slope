import math

import torch
import torch.optim as optim

from .utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.timing import Timer
from .utils.factors import ComputeI, ComputeG
from .utils.hylo_utils import EmptyBackend


def randomized_svd(B, rank):
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


class HKOROptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 inv_freq=10,
                 batch_averaged=True,
                 measure_time=False,
                 svd=False,
                 backend=EmptyBackend(),
                 half_precision=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(HKOROptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.AHandler = ComputeI()
        self.GHandler = ComputeG()
        self.batch_averaged = batch_averaged

        self.backend = backend
        self.verbose = self.backend.rank() == 0

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model

        self.a_reset_factor = {}
        self.g_reset_factor = {}

        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.AA_inv, self.GG_inv = {}, {}
        self.AA_Us, self.AA_Vt = {}, {}
        self.GG_Us, self.GG_Vt = {}, {}
        self.AA_sparse_factor, self.GG_sparse_factor = {}, {}
        self.AA, self.GG = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.inv_freq = inv_freq

        # Timing Variables
        self.timer = Timer(measure=measure_time)

        self.svd = svd

        self.reset_factors_freq = 10

        self.inputs = {}
        self.input_shapes = {}
        self.inputs_reduced = False
        self.grads = {}
        self.grad_shapes = {}

        self.data_type = torch.float16 if half_precision else torch.float32

        self.manual_reset_factors = False

        if self.manual_reset_factors:
            self.reset_weight = 0.8
        else:
            self.reset_weight = 0.1

        self.error_average_list = []
        self.error_svd_list = []

        self.sgd = False

        self.rank = 1
        self.sparse = False
        self.sparse_threshold = 1e-2
        self.sparse_AA, self.sparse_GG = {}, {}

        self.dummy_timer_start = torch.cuda.Event(enable_timing=True)
        self.dummy_timer_end = torch.cuda.Event(enable_timing=True)

    def inverse(self, prev_inv, rank_1):
        tmp1 = (prev_inv @ rank_1)
        tmp2 = (rank_1.t() @ prev_inv)
        return prev_inv - 1 / (1 + tmp2 @ rank_1) * tmp1 @ tmp2

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and (self.steps % self.inv_freq == 0 or self.steps < 10) and not self.sgd:
            a = self.AHandler(input[0].data, module)
            self.inputs[module] = a.to(self.data_type)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if (self.steps % self.inv_freq == 0 or self.steps < 10) and not self.sgd:
            if not self.inputs_reduced:
                self.reduce_inputs()
                self.inputs_reduced = True
            g, _ = self.GHandler(grad_output[0].data, module)
            self.grads[module] = g.to(self.data_type)

    def _prepare_model(self):
        count = 0
        if self.verbose:
            print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                self.a_reset_factor[module] = True
                self.g_reset_factor[module] = True
                if self.verbose:
                    print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        if torch.any(torch.isnan(self.m_aa[m])) or torch.any(torch.isnan(self.m_gg[m])):
            raise ValueError("NaN detected in m_aa or m_gg")
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
            self.m_aa[m] + eps / 10 * torch.eye(self.m_aa[m].shape[0], device=self.m_aa[m].device))
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
            self.m_gg[m] + eps / 10 * torch.eye(self.m_gg[m].shape[0], device=self.m_gg[m].device))

        # print(min(torch.min(self.d_a[m]), torch.min(self.d_g[m])))

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

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

    def _get_natural_grad(self, m, p_grad_mat, damping, identity=False):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        # v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        # v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        # v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if identity:
            v = p_grad_mat
        else:
            # self.dummy_timer_start.record()
            if self.rank == 1:
                if self.sparse:
                    v = self.GG_sparse_factor[m] @ p_grad_mat @ self.AA_sparse_factor[m]
                else:
                    v = self.GG_inv[m].to(torch.float32) @ p_grad_mat @ self.AA_inv[m].to(torch.float32)
            else:
                v = self.rank * (self.GG_Us[m] @ (self.GG_Vt[m] @ p_grad_mat @ self.AA_Us[m]) @ self.AA_Vt[m]) + (1 - self.rank) * p_grad_mat
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
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.inv_freq:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)
                if torch.isnan(p.data).any():
                    print('nan')
                    exit()

    def reduce_inputs(self):

        inputs = []
        # self.error_average_list = []
        # self.error_svd_list = []
        for module in self.modules:
            a = self.inputs[module]
            if self.svd:
                U, S, V = torch.linalg.svd(a, full_matrices=False)
                # average = torch.mean(a, dim=0, keepdim=True)
                # exact_mat = a.t() @ a / torch.tensor(a.shape[0])
                a = (V[0, :].reshape(-1, 1) * S[0] * torch.sum(U[:, 0] ** 2)).t() / torch.sqrt(torch.tensor(a.shape[0]))
                # error_average = torch.norm((average.t() @ average) - exact_mat)
                # error_svd = torch.norm((a.t() @ a) - exact_mat)
                # self.error_average_list.append(error_average)
                # self.error_svd_list.append(error_svd)

            else:
                # exact_mat = a.t() @ a / torch.tensor(a.shape[0])
                a = torch.mean(a, dim=0, keepdim=True)
                # error_average = torch.norm((a.t() @ a) - exact_mat) / torch.norm(exact_mat)
                # self.error_average_list.append(error_average)
                # self.error_svd_list.append(torch.zeros(1))

            self.inputs[module] = a

            if self.backend.size() != 1:
                inputs.append(self.inputs[module].reshape(1, -1))
        # import os
        # if not os.path.exists('error.csv'):
        #     with open('error.csv', 'w') as f:
        #         f.write('error_svd,error_mean\n')
        # with open('error.csv', 'a') as f:
        #     f.write(f'{torch.mean(torch.tensor(self.error_svd_list))},{torch.mean(torch.tensor(self.error_average_list))}\n')
        if self.backend.size() == 1:
            return

        self.reduced_inputs = torch.cat(inputs, dim=1)
        self.input_handles = []
        self.input_handles.append(self.backend.allreduce(self.reduced_inputs, async_op=True, average=True))

    def reduce_grads(self):
        grads = []
        for module in self.modules:
            g = self.grads[module]
            if self.svd:
                U, S, V = torch.linalg.svd(g, full_matrices=False)
                g = (V[0, :].reshape(-1, 1) * S[0] * torch.sum(U[:, 0] ** 2)).t()
            else:
                g = torch.mean(g, dim=0, keepdim=True)
            self.grads[module] = g

            if self.backend.size() != 1:
                grads.append(self.grads[module].reshape(1, -1))

        if self.backend.size() == 1:
            return

        self.reduced_grads = torch.cat(grads, dim=1)
        self.grad_handles = []
        self.grad_handles.append(self.backend.allreduce(self.reduced_grads, async_op=True, average=True))

    def sync_inputs(self):
        self.inputs_reduced = False
        if self.backend.size() == 1:
            return
        self.backend.sync(self.input_handles)
        offset = 0
        for module in self.modules:
            input_shape = self.inputs[module].shape
            input_numel = self.inputs[module].numel()
            self.inputs[module] = self.reduced_inputs[0, offset:offset + input_numel].reshape(input_shape)
            offset += input_numel

    def sync_grad(self):
        if self.backend.size() == 1:
            return
        self.backend.sync(self.grad_handles)
        offset = 0
        for module in self.modules:
            grad_shape = self.grads[module].shape
            grad_numel = self.grads[module].numel()
            self.grads[module] = self.reduced_grads[0, offset:offset + grad_numel].reshape(grad_shape)
            offset += grad_numel

    def update_factors(self):
        self.sync_inputs()

        for module in self.modules:
            if module == self.modules[-1]:
                continue
            a = self.inputs[module]
            v = a.t()
            # print("Forward Error", torch.norm(a.t() @ a - v @ v.t()) / torch.norm(a.t() @ a), S / torch.sum(S))
            if self.a_reset_factor[module]:
                if module not in self.AA_inv:
                    # self.AA[module] = torch.eye(a.size(1)).to(a.device)
                    self.AA_inv[module] = torch.eye(a.size(1), device=a.device, dtype=self.data_type)
                else:
                    self.AA_inv[module] = self.AA_inv[module] * (1 - self.reset_weight) + self.reset_weight * torch.eye(
                        a.size(1), device=a.device, dtype=self.data_type)
                self.a_reset_factor[module] = False

            self.method = 'approx'
            if self.method == 'exact':
                self.AA_inv[module] = torch.inverse(
                    a.t() @ a * (1 - self.stat_decay) + self.stat_decay * self.AA[module])
            elif self.method == 'low_rank':
                self.AA_inv[module] = torch.inverse(
                    v @ v.t() * (1 - self.stat_decay) + self.stat_decay * self.AA[module])
            elif self.method == 'approx':
                self.AA_inv[module] = self.inverse(self.AA_inv[module] / self.stat_decay,
                                                   v * math.sqrt(1 - self.stat_decay))
                # if self.verbose:
                #     print(torch.max(torch.abs(self.AA_inv[module].flatten())), torch.max(torch.abs(v)))
                if self.manual_reset_factors:
                    self.a_reset_factor[module] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
                    self.g_reset_factor[module] = self.steps % (self.inv_freq * self.reset_factors_freq) == 0
                else:
                    if torch.max(torch.abs(self.AA_inv[module].flatten())) > 2:
                        self.a_reset_factor[module] = True
                        self.g_reset_factor[module] = True
            if self.rank != 1:
                self.AA_Us[module], self.AA_Vt[module] = randomized_svd(self.AA_inv[module].to(torch.float32), self.rank)
            else:
                if self.sparse:
                    if self.sparsify[module]:
                        mask = (self.AA_inv[module].abs() > self.sparse_threshold).to(torch.int)
                        self.sparse_AA[module] = True
                        self.AA_sparse_factor[module] = (self.AA_inv[module] * mask).to(torch.float32).to_sparse_csr().to(self.AA_inv[module].device)
                        # print(mask.sum() / mask.numel(), mask.shape)
                    else:
                        self.sparse_AA[module] = False
                        self.AA_sparse_factor[module] = self.AA_inv[module].to(torch.float32)

        self.sync_grad()

        for module in self.modules:
            if module == self.modules[-1]:
                continue
            g = self.grads[module]
            v = g.t()
            # print("Backward Error", torch.norm(g.t() @ g - v @ v.t()) / torch.norm(g.t() @ g), S / torch.sum(S))
            if self.g_reset_factor[module]:
                if module not in self.GG_inv:
                    self.GG_inv[module] = torch.eye(g.size(1), device=g.device, dtype=self.data_type)
                else:
                    self.GG_inv[module] = self.GG_inv[module] * (1 - self.reset_weight) + self.reset_weight * torch.eye(
                        g.size(1), device=g.device, dtype=self.data_type)
                self.g_reset_factor[module] = False
            # GG = self.GG[module]
            if self.method == 'exact':
                self.GG_inv[module] = torch.inverse(
                    g.t() @ g * (1 - self.stat_decay) + self.stat_decay * self.GG[module])
            elif self.method == 'low_rank':
                self.GG_inv[module] = torch.inverse(
                    v @ v.t() * (1 - self.stat_decay) + self.stat_decay * self.GG[module])
            elif self.method == 'approx':
                self.GG_inv[module] = self.inverse(self.GG_inv[module] / self.stat_decay, v * (1 - self.stat_decay))
            if self.rank != 1:
                self.GG_Us[module], self.GG_Vt[module] = randomized_svd(self.GG_inv[module].to(torch.float32), self.rank)
            else:
                if self.sparse:
                    if self.sparsify[module]:
                        mask = (self.GG_inv[module].abs() > self.sparse_threshold).to(torch.int)
                        self.sparse_GG[module] = True
                        self.GG_sparse_factor[module] = (self.GG_inv[module] * mask).to(torch.float32).to_sparse_csr().to(self.GG_inv[module].device)
                        # print(mask.sum() / mask.numel(), mask.shape)
                    else:
                        self.sparse_GG[module] = False
                        self.GG_sparse_factor[module] = self.GG_inv[module].to(torch.float32)

    def reduce_and_update_factors(self):
        self.reduce_grads()
        self.update_factors()

    def compute_min_eigenvals(self):
        self.a_min_eigenvals = {}
        self.g_min_eigenvals = {}
        for module in self.modules[:-1]:
            d_a, Q_a = torch.linalg.eigh(self.AA_inv[module].to(torch.float32))
            d_g, Q_g = torch.linalg.eigh(self.GG_inv[module].to(torch.float32))

            self.a_min_eigenvals[module] = torch.min(d_a)
            self.g_min_eigenvals[module] = torch.min(d_g)

            if self.verbose:
                import os
                if not os.path.exists("a_eigen.csv"):
                    with open("a_eigen.csv", "w") as f:
                        f.write("min_a_eigen,min_a_abs_eigen,max_a_eigen,max_a_abs_eigen,condition_number\n")
                with open("a_eigen.csv", "a") as f:
                    f.write(str(torch.min(d_a).item()) + "," + str(torch.min(torch.abs(d_a)).item()) + "," + str(
                        torch.max(d_a).item()) + "," + str(torch.max(torch.abs(d_a)).item()) + "," + str(
                        torch.max(torch.abs(d_a)).item() / torch.min(torch.abs(d_a)).item()) + "\n")
                if not os.path.exists("g_eigen.csv"):
                    with open("g_eigen.csv", "w") as f:
                        f.write("min_g_eigen,min_g_abs_eigen,max_g_eigen,max_g_abs_eigen,condition_number\n")
                with open("g_eigen.csv", "a") as f:
                    f.write(str(torch.min(d_g).item()) + "," + str(torch.min(torch.abs(d_g)).item()) + "," + str(
                        torch.max(d_g).item()) + "," + str(torch.max(torch.abs(d_g)).item()) + "," + str(
                        torch.max(torch.abs(d_g)).item() / torch.min(torch.abs(d_g)).item()) + "\n")

            # print(min(torch.min(self.d_a[m]), torch.min(self.d_g[m])))

    def step_mkor(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        if self.steps == 0:
            self.reset_factors()
            self.compute_preconditioning_costs()
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        if self.steps % self.inv_freq == 0 or self.steps < 10:
            self.timer("reduce_and_update_factors", self.reduce_and_update_factors)
            # self.compute_min_eigenvals()
        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self.timer("precondition", self._get_matrix_form_grad, m=m, classname=classname)
            v = self.timer("precondition", self._get_natural_grad, m=m, p_grad_mat=p_grad_mat, damping=damping,
                           identity=m == self.modules[-1])
            updates[m] = v

        self.timer("update_weights", self._kl_clip_and_update_grad, updates=updates, lr=lr)

        self.timer("update_weights", self._step, closure=closure)
        self.steps += 1

    def step_sgd(self, closure=None):
        vg_sum = 0
        lr_squared = self.param_groups[0]['lr'] ** 2
        for m in self.modules:
            vg_sum += ((m.weight.grad.data) ** 2).sum().item() * lr_squared
            if m.bias is not None:
                vg_sum += ((m.bias.grad.data) ** 2).sum().item() * lr_squared
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.mul_(nu)
        self._step(closure=closure)

    def step(self, closure=None):
        if self.sgd:
            self.step_sgd(closure)
            self.steps = 0
        else:
            self.step_mkor(closure)

    def reset_factors(self):
        for m in self.GG_inv:
            self.GG_inv[m] = torch.eye(self.GG_inv[m].size(0), device=self.GG_inv[m].device, dtype=self.data_type)
            self.AA_inv[m] = torch.eye(self.AA_inv[m].size(0), device=self.AA_inv[m].device, dtype=self.data_type)

    def compute_preconditioning_costs(self):
        costs = []
        self.sparsify = {}
        for m in self.modules[:-1]:
            costs.append(self.inputs[m].size(1) * self.grads[m].size(1) * (self.inputs[m].size(1) + self.grads[m].size(1)))
            self.sparsify[m] = False
        costs = torch.tensor(costs)
        self.costs = costs / torch.sum(costs)
        sum = 0
        while sum < 0.7:
            max, idx = torch.max(self.costs, 0)
            self.costs[idx] = 0
            sum += max
            self.sparsify[self.modules[idx]] = True
