import torch
from slope.utils import prune_column_wise
from pruning_kernels.sparse_backend import pruner
from torch.sparse import to_sparse_semi_structured


class ADAMWSparse():
    r"""Implements AdamW algorithm with sparse weight updates.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_ and `Decoupled Weight Decay Regularization`_.

    Arguments:
        grad_dict (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        mask_change_freq (int): the frequency to change the backward pass mask
    """

    def __init__(
            self,
            model,
            grad_dict,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            mask_change_freq=100,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_dict = grad_dict
        self.model = model
        self.mask_change_freq = mask_change_freq
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # super(ADAMWSparse, self).__init__([], defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        """
        loss = None
        if closure is not None:
            loss = closure()
        for _, grad_dict in self.grad_dict.items():
            mask = grad_dict['mask']
            grad_weight = grad_dict['grad']
            weight = grad_dict['weight']
            weight_t = grad_dict['weight_t']

            if grad_weight is None:
                continue

            if self.weight_decay != 0:
                if type(weight) == list:
                    for w in weight:
                        w.data.values().add_(w.values(), alpha=-self.weight_decay*self.lr)
                else:
                    weight.data.values().add_(weight.values(), alpha=-self.weight_decay*self.lr)

            if not 'state' in grad_dict:
                state = {
                    'step': 0,
                    'exp_avg': torch.zeros(grad_weight.shape[0], grad_weight.shape[1], device=grad_weight.device),
                    'exp_avg_sq': torch.zeros(grad_weight.shape[0], grad_weight.shape[1], device=grad_weight.device),
                }
                grad_dict['state'] = state

            state = grad_dict['state']
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.mul_(self.betas[0]).add_(grad_weight, alpha=1 - self.betas[0])
            exp_avg_sq.mul_(self.betas[1]).addcmul_(grad_weight, grad_weight, value=1 - self.betas[1])

            denom = exp_avg_sq.sqrt().add_(self.eps)

            bias_correction1 = 1 - self.betas[0] ** state['step']
            bias_correction2 = 1 - self.betas[1] ** state['step']
            step_size = self.lr * (bias_correction2 ** 0.5) / bias_correction1


            if type(weight) == list:
                start_idx = 0
                for w in weight:
                    end_idx = start_idx + w.shape[0]
                    w.data.values().addcdiv_(exp_avg[start_idx:end_idx, :], denom[start_idx:end_idx, :], value=-step_size)
                    start_idx = end_idx
            else:
                weight.data.values().addcdiv_(exp_avg, denom, value=-step_size)


            dtype = weight.dtype if type(weight) != list else weight[0].dtype
            dense_weight = torch.zeros_like(mask, dtype=dtype)
            # The mask values for W are equal to 2 or 3
            if type(weight) == list:
                start = 0
                for w in weight:
                    mask_ = mask[start:start+w.shape[0], :] > 1
                    dense_weight[start:start+w.shape[0], :][mask_] = w.values().flatten()
                    start += w.shape[0]
            else:
                dense_weight[mask > 1] = weight.values().flatten()


            if type(weight_t) == list:
                start = 0
                # The mask values for W_t are equal to 1 or 3, so we set them to a number larger than 1 for pruning
                for wt in weight_t:
                    end = start + wt.shape[0]
                    if state['step'] % self.mask_change_freq != self.mask_change_freq - 1:
                        mask_ = mask[:, start:end].t()
                        mask_ = (torch.logical_or((mask_ == 1), (mask_ == 3)).to(torch.int8) * 2).contiguous()
                        wt.values().data = pruner.prune_and_compress(
                            dense_weight[:, start:end].t().contiguous(),
                            mask_)
                    else:
                        wt_new, _ = prune_column_wise(dense_weight[:, start:end], transpose=True, n=2, m=4)
                        wt_new = to_sparse_semi_structured(wt_new)
                        wt.values().data = wt_new.values().data
                        wt.indices().data = wt_new.indices().data
                    start = end
            else:
                if state['step'] % self.mask_change_freq != self.mask_change_freq - 1:
                    mask_ = mask.t()
                    mask_ = (torch.logical_or((mask_ == 1), (mask_ == 3)).to(torch.int8) * 2).contiguous()
                    weight_t.values().data = pruner.prune_and_compress(
                        dense_weight.t().contiguous(),
                        mask_)
                else:
                    wt, _ = prune_column_wise(dense_weight, transpose=True, n=2, m=4)
                    wt = to_sparse_semi_structured(wt)
                    weight_t.values().data = wt.values().data
                    weight_t.indices().data = wt.indices().data
        return loss

    def zero_grad(self):
        for _, grad_dict in self.grad_dict.items():
            grad_dict['grad'] = None


    def clip_grad_norm(self, max_norm, norm_type=2):
        r"""Clip the norm of the gradients for all parameters to a maximum value.

        Arguments:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be 'inf' for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        if norm_type == 'inf':
            total_norm = max(p['grad'].abs().max() for _, p in self.grad_dict.items())
            for param in self.model.parameters():
                if param.grad is not None:
                    total_norm = max(total_norm, param.grad.data.abs().max())
        else:
            param_stack = [p['grad'].norm(norm_type) for _, p in self.grad_dict.items()]

            for param in self.model.parameters():
                if param.grad is not None:
                    param_stack.append(param.grad.norm(norm_type))

            total_norm = torch.norm(torch.stack(param_stack), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for _, p in self.grad_dict.items():
                p['grad'].mul_(clip_coef)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        return total_norm



class ADAMW(ADAMWSparse):
    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        """
        loss = None
        if closure is not None:
            loss = closure()
        for _, grad_dict in self.grad_dict.items():
            grad_weight = grad_dict['grad']
            weight = grad_dict['weight']

            if grad_weight is None:
                continue

            if self.weight_decay != 0:
                    weight.data.add_(weight, alpha=-self.weight_decay * self.lr)

            if not 'state' in grad_dict:
                state = {
                    'step': 0,
                    'exp_avg': torch.zeros(grad_weight.shape[0], grad_weight.shape[1], device=grad_weight.device),
                    'exp_avg_sq': torch.zeros(grad_weight.shape[0], grad_weight.shape[1], device=grad_weight.device),
                }
                grad_dict['state'] = state

            state = grad_dict['state']
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.mul_(self.betas[0]).add_(grad_weight, alpha=1 - self.betas[0])
            exp_avg_sq.mul_(self.betas[1]).addcmul_(grad_weight, grad_weight, value=1 - self.betas[1])

            denom = exp_avg_sq.sqrt().add_(self.eps)

            bias_correction1 = 1 - self.betas[0] ** state['step']
            bias_correction2 = 1 - self.betas[1] ** state['step']
            step_size = self.lr * (bias_correction2 ** 0.5) / bias_correction1


            weight.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss