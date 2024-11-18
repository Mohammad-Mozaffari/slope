import math

import torch
import torch.optim as optim

from .utils.kfac_utils import (ComputeCovA, ComputeCovG)
from .utils.kfac_utils import update_running_stat
from utils.timing import Timer
from .utils.hylo_utils import EmptyBackend


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 measure_time=False,
                 backend=EmptyBackend()):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged


        self.backend = backend
        self.verbose = self.backend.rank() == 0

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.aa, self.gg = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

        # Timing Variables
        self.timer = Timer(measure=measure_time)


    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0 and module != self.modules[-1]:
            self.aa[module] = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(self.aa[module].new(self.aa[module].size(0)).fill_(1))
            

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.steps % self.TCov == 0 and module != self.modules[-1]:
            self.gg[module] = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(self.gg[module].new(self.gg[module].size(0)).fill_(1))

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

        # if self.verbose:
        #     import os
        #     if not os.path.exists("eigen.csv"):
        #         with open("eigen.csv", "w") as f:
        #             f.write("min_r_eigen,min_r_abs_eigen,max_r_eigen,max_r_abs_eigen,r_condition_number,min_l_eigen,min_l_abs_eigen,max_l_eigen,max_l_abs_eigen,l_condition_number\n")
        #     with open("eigen.csv", "a") as f:
        #         f.write(str(torch.min(self.d_a[m]).item()) + "," + str(torch.min(torch.abs(self.d_a[m])).item()) + "," + str(torch.max(self.d_a[m]).item()) + "," + str(torch.max(torch.abs(self.d_a[m])).item()) + "," + str(torch.max(torch.abs(self.d_a[m])).item() / torch.min(torch.abs(self.d_a[m])).item()) + str(torch.min(self.d_g[m]).item()) + "," + str(torch.min(torch.abs(self.d_g[m])).item()) + "," + str(torch.max(self.d_g[m]).item()) + "," + str(torch.max(torch.abs(self.d_g[m])).item()) + "," + str(torch.max(torch.abs(self.d_g[m])).item() / torch.min(torch.abs(self.d_g[m])).item()) + "\n")
        #         print(self.d_a[m].min(), self.d_a[m].max(), self.d_g[m].min(), self.d_g[m].max())
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
        if identity:
            v = p_grad_mat
        else:
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
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
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
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


    def reduce_input_covs(self):
        if self.backend.size() == 1:
            return
        aa = []
        for module in self.modules[0:-1]:
            aa.append(self.aa[module].reshape(1, -1))
        
        self.reduced_aa = torch.cat(aa, dim=1)
        self.input_handles = []
        self.input_handles.append(self.backend.allreduce(self.reduced_aa, async_op=True, average=True))

    
    def reduce_grad_covs(self):
        if self.backend.size() == 1:
            return
        gg = []
        for module in self.modules[0:-1]:
            gg.append(self.gg[module].reshape(1, -1))
        
        self.reduced_gg = torch.cat(gg, dim=1)
        self.grad_handles = []
        self.grad_handles.append(self.backend.allreduce(self.reduced_gg, async_op=True, average=True))

    
    def sync_input_covs(self):
        if self.backend.size() == 1:
            for module in self.modules[0:-1]:
                update_running_stat(self.aa[module], self.m_aa[module], self.stat_decay)
            return
        self.backend.sync(self.input_handles)
        offset = 0
        for module in self.modules[0:-1]:
            input_shape = self.aa[module].shape
            input_numel = self.aa[module].numel()
            self.aa[module] = self.reduced_aa[0, offset:offset+input_numel].reshape(input_shape)
            update_running_stat(self.aa[module], self.m_aa[module], self.stat_decay)
            offset += input_numel
        
    
    def sync_grad_covs(self):
        if self.backend.size() == 1:
            for module in self.modules[0:-1]:
                update_running_stat(self.gg[module], self.m_gg[module], self.stat_decay)
            return
        self.backend.sync(self.grad_handles)
        offset = 0
        for module in self.modules[0:-1]:
            grad_shape = self.gg[module].shape
            grad_numel = self.gg[module].numel()
            self.gg[module] = self.reduced_gg[0, offset:offset+grad_numel].reshape(grad_shape)
            update_running_stat(self.gg[module], self.m_gg[module], self.stat_decay)
            offset += grad_numel

    
    def reduce_covs(self):
        self.reduce_input_covs()
        self.reduce_grad_covs()
        self.sync_input_covs()
        self.sync_grad_covs()
        

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        self.timer("reduce_factors", self.reduce_covs)
        for m in self.modules:
            classname = m.__class__.__name__
            if m != self.modules[-1]:
                if self.steps % self.TInv == 0:
                    self.timer("update_inv", self._update_inv, m=m)
            p_grad_mat = self.timer("precondition", self._get_matrix_form_grad, m=m, classname=classname)
            v = self.timer("precondition", self._get_natural_grad, m=m, p_grad_mat=p_grad_mat, damping=damping, identity=m==self.modules[-1])
            updates[m] = v
            
        self.timer("apply_updates", self._kl_clip_and_update_grad, updates=updates, lr=lr)

        self.timer("apply_updates", self._step, closure=closure)
        self.steps += 1
