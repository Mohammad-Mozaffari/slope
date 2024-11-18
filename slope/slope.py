import torch

from slope.utils import prune_row_wise, prune_column_wise
from slope.ops.addmm import addmm
from pruning_kernels.sparse_backend import pruner
from types import MethodType
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import numpy as np
from slope.ops.matvec_add import matvec_add

grad_dict = {}


def sync_grads():
    handles = []
    operator = torch.distributed.ReduceOp.SUM
    for key in grad_dict:
        if grad_dict[key]['grad'] is not None:
            handles.append(torch.distributed.all_reduce(grad_dict[key]['grad'], async_op=True, op=operator))
    for handle in handles:
        handle.wait()



# @torch.compile
def to_sparse_semi_structured_compiled(tensor):
    return to_sparse_semi_structured(tensor)


class LoRAMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, input, weight, lora_right, lora_rank, bias, d_out):
        if type(weight) == list:
            if input.dim() == 3:
                xwl= torch.empty(input.shape[0],
                                 input.shape[1],
                                 d_out + lora_rank,
                                 device=input.device,
                                 dtype=input.dtype)
                start_idx = 0
                for i in range(len(weight)):
                    w = weight[i]
                    if w is not None:
                        xwl[:, :, start_idx:start_idx + w.shape[0]] = torch.matmul(input, w.t())
                        start_idx += w.shape[0]
            else:
                xwl = torch.empty(input.shape[0], d_out + lora_rank, device=input.device, dtype=input.dtype)
                start_idx = 0
                for w in weight:
                    if w is not None:
                        xwl[:, start_idx:start_idx + w.shape[0]] = torch.matmul(input, w.t())
                        start_idx += w.shape[0]
        else:
            xwl= torch.matmul(input, weight.t())

        if xwl.dim() == 3:
            bs, seq, _ = xwl.shape
            xwl = xwl.view(bs * seq, -1)
        else:
            seq = -1
        result = addmm(xwl[:, -lora_rank:], lora_right, xwl[:, :-lora_rank])
        if seq != -1:
            result = result.view(bs, seq, -1)
        if bias is not None:
            result = matvec_add(result, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("LoRAMatmul backward not implemented")


class SLoPeMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, input, weight, weight_t, mask, bias, d_in, d_out):
        dtype = weight.dtype if type(weight) != list else weight[0].dtype
        input = input.to(dtype)
        if type(weight) == list:
            if input.dim() == 3:
                result = torch.empty(input.shape[0], input.shape[1], d_out, device=input.device, dtype=input.dtype)
                start_idx = 0
                weight1 = weight[0]
                for i in range(len(weight)):
                    w = weight[i]
                    if w is not None:
                        result[:, :, start_idx:start_idx + w.shape[0]] = torch.matmul(input, w.t())
                        start_idx += w.shape[0]
            else:
                result = torch.empty(input.shape[0], d_out, device=input.device, dtype=input.dtype)
                start_idx = 0
                weight1 = weight[0]
                for w in weight:
                    if w is not None:
                        result[:, start_idx:start_idx + w.shape[0]] = torch.matmul(input, w.t())
                        start_idx += w.shape[0]
        else:
            weight1 = weight
            result = torch.matmul(input, weight.t())
        if bias is not None:
            result = matvec_add(result, bias)

        if type(weight_t) == list:
            weight_t1, weight_t2, weight_t3, weight_t4 = weight_t
        else:
            weight_t1, weight_t2, weight_t3, weight_t4 = weight_t, None, None, None
        ctx.save_for_backward(input, weight1, weight_t1, weight_t2, weight_t3, weight_t4, mask, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_t1, weight_t2, weight_t3, weight_t4, mask, bias = ctx.saved_tensors
        if weight in grad_dict:
            grad_weight = torch.matmul(grad_output.view(-1, grad_output.shape[-1]).t(), input.view(-1, input.shape[-1]))
            grad_weight = pruner.prune_and_compress(grad_weight, mask)

            if grad_dict[weight]['grad'] is None:
                grad_dict[weight]['grad'] = grad_weight
            else:
                grad_dict[weight]['grad'] += grad_weight
        weight_t = [weight_t1, weight_t2, weight_t3, weight_t4] if weight_t2 is not None else weight_t1
        if type(weight_t) == list:
            d_in = input.shape[-1]
            if input.dim() == 3:
                bs = input.shape[0] * input.shape[1]
            else:
                bs = input.shape[0]

            grad_input = torch.empty(bs, d_in, device=input.device, dtype=input.dtype)
            start_idx = 0
            for i in range(len(weight_t)):
                w = weight_t[i]
                if w is not None:
                    grad_output_reshaped = grad_output.view(-1, grad_output.shape[-1]).t()
                    grad_input[:, start_idx:start_idx + w.shape[0]] = torch.matmul(w, grad_output_reshaped).t()
                    start_idx += w.shape[0]
            grad_input = grad_input.view(input.shape)
        else:
            grad_input = torch.matmul(weight_t, grad_output.view(-1, grad_output.shape[-1]).t()).t().view(input.shape)

        if bias is not None:
            if grad_output.dim() == 3:
                grad_bias = grad_output.sum(0).sum(0)
            else:
                grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        return grad_input, None, None, None, grad_bias, None, None


def slope_linear_forward(module, input):
    if hasattr(module, "lora_right"):
        if module.merge_lora:
            output = LoRAMatmul.apply(
                input,
                module.weight,
                module.lora_right,
                module.lora_rank,
                module.bias,
                module.d_out,
            )
            return output

    output = SLoPeMatmul.apply(input, module.weight, module.weight_t, module.mask, module.bias, module.d_in,
                               module.d_out)
    if hasattr(module, "lora_right"):
        output += torch.matmul(torch.matmul(input, module.lora_left), module.lora_right) / module.lora_rank
    return output


class DenseMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, input, weight):
        input = input.to(weight.dtype)
        ctx.save_for_backward(input, weight)
        result = torch.matmul(input, weight.t())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.view(-1, grad_output.shape[-1]).t(), input.view(-1, input.shape[-1]))

        mask = weight == 0.
        grad_weight[mask] = 0.
        grad_weight, _ = prune_column_wise(grad_weight, n=2, m=4)
        if weight in grad_dict:
            if grad_dict[weight]['grad'] is None:
                grad_dict[weight]['grad'] = grad_weight
            else:
                grad_dict[weight]['grad'] += grad_weight
            grad_weight = None
        return grad_input, grad_weight


def dense_linear_forward(module, input):
    output = DenseMatmul.apply(input, module.weight)
    if module.bias is not None:
        output += module.bias
    return output


def prune_model(model, compress=True, tiling=True, add_lora=False, lora_rank=0, merge_lora=False, backend='cusparselt'):
    if backend == 'cusparselt':
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    elif backend == 'cutlass':
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    else:
        raise ValueError("Backend must be one of cusparselt, cutlass")
    known_modules = {"Linear"}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in known_modules:
            module.d_out, module.d_in = module.weight.shape
            aspect_ratio = max(module.weight.shape) / min(module.weight.shape)
            if aspect_ratio > 5:
                print("Skipping module with size ", module.weight.shape)
                continue
            w = module.weight.clone().detach()
            if add_lora:
                assert lora_rank > 0, "LoRA rank must be greater than 0"
                module.lora_rank = lora_rank
                module.merge_lora = merge_lora
                module.lora_right = torch.nn.Parameter(torch.zeros(lora_rank,
                                                                   w.shape[0],
                                                                   device=w.device,
                                                                   dtype=w.dtype),
                                                       requires_grad=True)
                lora_left = torch.nn.Parameter(torch.randn(w.shape[1],
                                                           lora_rank,
                                                           device=w.device,
                                                           dtype=w.dtype),
                                               requires_grad=True)
                if merge_lora:
                    w = torch.cat([w, lora_left.t()], dim=0)
                else:
                    module.lora_left = lora_left
            w, mask = prune_row_wise(w, n=2, m=4)
            w_double_pruned, mask_double_pruned = prune_column_wise(w, n=2, m=4)
            module.mask = 2 * mask.to(torch.int8) + mask_double_pruned.to(torch.int8)
            if compress:
                if aspect_ratio > 1 and tiling:
                    d_out, d_in = module.weight.shape
                    if d_out > d_in:
                        num_partitions = int(np.ceil(d_out / (d_in + lora_rank // 4)))
                        del module.weight
                        torch.cuda.empty_cache()
                        module.weight = [None] * 4
                        for i in range(num_partitions):
                            module.weight[i] = torch.nn.Parameter(
                                to_sparse_semi_structured_compiled(w[i * d_in:(i + 1) * d_in, :]),
                                requires_grad=True)
                        w_t = w_double_pruned.t().contiguous()
                        module.weight_t = to_sparse_semi_structured_compiled(w_t)
                    else:
                        module.weight = torch.nn.Parameter(to_sparse_semi_structured_compiled(w), requires_grad=True)
                        num_partitions = int(np.ceil(d_in / (d_out + lora_rank // 4)))
                        module.weight_t = [None] * 4
                        for i in range(num_partitions):
                            module.weight_t[i] = to_sparse_semi_structured_compiled(
                                w_double_pruned.t()[i * d_out:(i + 1) * d_out, :].contiguous())
                else:
                    module.weight = torch.nn.Parameter(to_sparse_semi_structured_compiled(w), requires_grad=True)
                    module.weight_t = to_sparse_semi_structured_compiled(w_double_pruned.t().contiguous())
                key = module.weight if type(module.weight) != list else module.weight[0]
                grad_dict[key] = {
                    'grad': None,
                    'mask': module.mask,
                    'weight': module.weight,
                    'weight_t': module.weight_t
                }
                module.forward = MethodType(slope_linear_forward, module)
            else:
                module.weight = torch.nn.Parameter(w, requires_grad=True)


def replace_linear_layers(model, manual_optimizer=False):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            aspect_ratio = max(module.weight.shape) / min(module.weight.shape)
            if aspect_ratio > 5:
                print("Skipping module with size ", module.weight.shape)
                continue
            if manual_optimizer:
                grad_dict[module.weight] = {
                    'grad': None,
                    'weight': module.weight,
                }
            module.forward = MethodType(dense_linear_forward, module)