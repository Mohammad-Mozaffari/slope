import torch
from torch.optim import Optimizer
from compression.ops import grad_dict
from compression.pruning_kernels.sparse_backend import pruner


class SparseAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Implements AdamW algorithm with weight decay.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
        """
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SparseAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    if p.shape == torch.Size([1]):
                        if p.data.item() not in grad_dict:
                            continue
                    else:
                        continue
                grad = p.grad.data if p.grad is not None else grad_dict[p.data.item()]['grad']
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if p.grad is not None:
                    # Update parameters with weight decay
                    p.data.addcdiv_(-group['lr'], exp_avg, denom)
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
                else:
                    mask = grad_dict[p.data.item()]['mask']
                    fwd_sparse_idx = p.data.to(torch.int)
                    bwd_sparse_idx = grad_dict[p.data.item()]['bwd_sparse_idx']
                    new_weight = pruner.sparse_add(exp_avg / denom,
                                                   fwd_sparse_idx,
                                                   torch.tensor(-group['lr']),
                                                   torch.tensor(1 - group['lr'] * group['weight_decay']))
                    pruner.update_sparse_matrix(new_weight, fwd_sparse_idx)
                    dense_weight = torch.zeros(exp_avg.size(0), 2 * exp_avg.size(1), dtype=torch.float16, device=exp_avg.device)

                    dense_weight[mask] = new_weight.flatten()
                    compressed_transposed_grad = pruner.prune_and_compress(dense_weight.t().contiguous(), mask.t().contiguous()).t().contiguous()
                    pruner.update_sparse_matrix(compressed_transposed_grad, bwd_sparse_idx)




        return loss
