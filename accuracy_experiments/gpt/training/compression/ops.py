import torch
from .pruning_kernels.sparse_backend import pruner
from .quantization.model_quantizing import *
from .quantization.quantization_kernels.int_matmul import int_matmul

grad_dict = {}


def density_ratio(mat):
    return (mat != 0.).sum() / mat.numel()


def compute_remainder_sparsity(mat, m, n, init_sparsity=None):
    if init_sparsity is None:
        init_sparsity = 1 - (mat.sum() / mat.numel())
    reshaped_mat = mat.reshape(-1, n).to(torch.int8)
    reshaped_mat, _ = torch.sort(reshaped_mat, dim=1, descending=True)
    reshaped_mat[:, 0:m] = 0.
    final_sparsity = 1 - (reshaped_mat.sum() / reshaped_mat.numel())
    return final_sparsity, init_sparsity - final_sparsity


def prune_row_wise(input, n=0, m=0):
    if (n, m) == (0, 0):
        n = 1 if input.dtype == torch.float32 else 2
        m = 2 if input.dtype == torch.float32 else 4
    input = input.contiguous()
    input_shape = input.shape
    sparse_input, mask = pruner.prune(input.reshape(-1, input.shape[-1]), n, m)  # sparsify(input, m, n)
    sparse_input = sparse_input.reshape(input_shape)
    mask = mask.reshape(input_shape)
    return sparse_input, mask


def prune_column_wise(input, transpose=False, n=0, m=0):
    assert not (transpose and (input.dim() != 2))
    input_shape = input.shape
    input = input.reshape(-1, input.shape[-1])
    sparse_input, mask = prune_row_wise(input.t(), n=n, m=m)
    if not transpose:
        sparse_input = sparse_input.t()
        mask = mask.t()
        sparse_input = sparse_input.reshape(input_shape)
        mask = mask.reshape(input_shape)
    return sparse_input, mask


def prune_unstructured(input, sparsity_ratio):
    abs_tensor = torch.abs(input)
    flat_abs_tensor = abs_tensor.flatten()
    kept_elements = int((1 - sparsity_ratio) * flat_abs_tensor.numel())

    _, indices = torch.topk(flat_abs_tensor, k=kept_elements)
    sparse_mask = torch.ones_like(flat_abs_tensor , dtype=torch.bool)
    sparse_mask[indices] = False
    sparse_mask = sparse_mask.view_as(abs_tensor)

    input[sparse_mask] = 0.
    return input, sparse_mask

class Sparsify(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(mask)
        return sparse_input

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[sparsity_mask] = 0.
        return grad_input


class Matmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight):
        input, weight = input.half(), weight.half()
        ctx.save_for_backward(input, weight)
        return torch.matmul(input, weight.t()).half()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight


class DynamicPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight,  weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(sparse_input, weight, mask)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = sparse_input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = sparse_input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None, None, None, None


class ReductionDimDynamicPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight,  weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        sparse_input, mask = prune_row_wise(input)
        ctx.save_for_backward(sparse_input, weight, mask)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = sparse_input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = sparse_input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)

        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        input = input.to(dtype)
        grad_output = grad_output.to(dtype)
        weight = weight.to(dtype)

        input2, _ = prune_column_wise(input, transpose=False)
        if density_ratio(input2) > 0.37:
            input = input2
        with open("density.csv", "a") as f:
            f.write(f"{density_ratio(input2)}\n")
        grad_weight = torch.matmul(grad_output.t(), input)

        grad_input = torch.matmul(grad_output, weight)
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None, None, None, None


class StaticPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        if mask is None:
            sparse_input, mask = prune_row_wise(input)
        else:
            sparse_input = input
            sparse_input[mask] = 0.
        ctx.save_for_backward(sparse_input, weight, mask)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = sparse_input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = sparse_input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)

        output = output.clone()
        return output, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None, None, None, None, None


class ReductionDimStaticPruneInputsMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        if mask is None:
            sparse_input, mask = prune_row_wise(torch.rand_like(input))
        else:
            try:
                sparse_input = input
                sparse_input[mask] = 0.
            except:
                sparse_input = input
        ctx.save_for_backward(sparse_input, weight, mask)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = sparse_input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = sparse_input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)

        output = output.clone()
        return output, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, sparsity_mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        input = input.to(dtype)
        grad_output = grad_output.to(dtype)
        weight = weight.to(dtype)

        input, _ = prune_column_wise(input, transpose=False)
        grad_weight = torch.matmul(grad_output.t(), input)

        grad_input = torch.matmul(grad_output, weight)
        grad_input = grad_input.reshape(input_shape)
        grad_input[sparsity_mask] = 0.
        return grad_input, grad_weight, None, None, None, None, None


class DynamicPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight,  weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        sparse_weight, _ = prune_column_wise(weight)
        ctx.save_for_backward(input, sparse_weight)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], sparse_weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], sparse_weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, sparse_weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, sparse_weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, sparse_weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)

        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))

        weight, _ = prune_row_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None


class ReductionDimDynamicPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight,  weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        sparse_weight, _ = prune_row_wise(weight)
        ctx.save_for_backward(input, sparse_weight)

        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], sparse_weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], sparse_weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, sparse_weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, sparse_weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, sparse_weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))

        weight, _ = prune_column_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None


class StaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        if mask is None:
            sparse_weight, mask = prune_column_wise(weight)
        else:
            sparse_weight = weight
            sparse_weight[mask] = 0.
        ctx.save_for_backward(input, sparse_weight, mask)
        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], sparse_weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], sparse_weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, sparse_weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, sparse_weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, sparse_weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()
        return output, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_weight[mask] = 0.

        weight, _ = prune_row_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None, None


class DynamicPruneOutputGradMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input
        ctx.save_for_backward(input, weight)

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        sparse_grad_output, _ = prune_row_wise(grad_output)
        grad_weight = torch.matmul(sparse_grad_output.t().to(dtype), input.to(dtype))
        grad_input = torch.matmul(sparse_grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None


class ReductionDimDynamicPruneOutputGradMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input
        ctx.save_for_backward(input, weight)

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        sparse_grad_output_transpose, _ = prune_column_wise(grad_output, transpose=True)
        grad_weight = torch.matmul(sparse_grad_output_transpose.to(dtype), input.to(dtype))
        sparse_grad_output, _ = prune_row_wise(grad_output)
        grad_input = torch.matmul(sparse_grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None


class ReductionDimStaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None, n=0, m=0, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        input_quantizer = Quantizer("input")
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)
        else:
            input_for_mul = input
        if mask is None:
            sparse_weight, mask = prune_row_wise(weight, n, m)
            weight.data = sparse_weight.data
        else:
            sparse_weight = weight
            sparse_weight[mask] = 0.
        ctx.save_for_backward(input, sparse_weight, mask, n, m)
        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], sparse_weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], sparse_weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, sparse_weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, sparse_weight.t())

        del input_for_mul
        if quantization_en:
            dim = output.dim()
            if dim == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, sparse_weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()

        return output, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, mask, n, m = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_weight[mask] = 0.

        weight, _ = prune_column_wise(weight, n=n, m=m)

        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None,None, None, None, None


class AcceleratedStaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""
    @staticmethod
    def forward(ctx, input, weight, mask=None, sparse_index=None):
        if input.dtype != torch.float16:
            input = input.half()
        if input.dim() == 3:
            batch_size = input.shape[0]
            sequence_length = input.shape[1]
            input = input.reshape(batch_size * sequence_length, -1)
        else:
            batch_size = input.shape[0]
            sequence_length = -1
        if sparse_index is None and input.size(0) > 16:
            weight = weight.half() + 1e-3
            transpose_A = False
            transpose_B = True
            sparse_A = False
            transposable_mask = True
            sparse_index = []
            sparse_index.append(pruner.setup_spmatmul(input, weight, transpose_A, transpose_B, sparse_A, transposable_mask))
            transpose_B = False
            random_output = torch.randn(input.size(0), weight.size(0), dtype=torch.float16, device=input.device)
            sparse_index.append(pruner.setup_spmatmul(random_output, weight, transpose_A, transpose_B, sparse_A, transposable_mask))
            mask = weight != 0

        if sparse_index is None:
            output = torch.matmul(input, weight.t())
            ctx.save_for_backward(input, -torch.ones(1), mask, weight)
        else:
            sparseA = False
            output = pruner.spmatmul(input, sparse_index[0], sparseA)
            ctx.save_for_backward(input, sparse_index[1], mask, weight)

        if sequence_length != -1:
            output = output.reshape(batch_size, sequence_length, -1)


        return output.clone(), mask, sparse_index

    @staticmethod
    def backward(ctx, grad_output, grad_mask, grad_sparse_index):
        input, sparse_index, mask, weight = ctx.saved_tensors
        if grad_output.dim() == 3:
            batch_size = grad_output.shape[0]
            sequence_length = grad_output.shape[1]
            grad_output = grad_output.reshape(batch_size * sequence_length, -1)
        else:
            batch_size = input.shape[0]
            sequence_length = -1
        dtype = torch.float16
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        if mask is not None:
            if weight.dim() == 1:
                if weight.data not in grad_dict or grad_dict[weight.data] is None:
                    grad_dict[weight.data.item()] = {}
                    grad_dict[weight.data.item()]['mask'] = mask
                    grad_dict[weight.data.item()]['bwd_sparse_idx'] = sparse_index
                    grad_dict[weight.data.item()]['grad'] = pruner.prune_and_compress(grad_weight, mask)
                else:
                    grad_dict[weight.data.item()]['grad'] += pruner.prune_and_compress(grad_weight, mask)
                grad_weight = None
            sparseA = False
            grad_input = pruner.spmatmul(grad_output, sparse_index, sparseA)
        else:
            grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))

        if sequence_length != -1:
            grad_input = grad_input.reshape(batch_size, sequence_length, -1)

        return grad_input, grad_weight, None, None


class UnstructuredStaticPruneWeightMatmul(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, input, weight, mask=None, unstructured_sparsity_ratio=60, weight_quantizer = None, quantization_en= False,  qbitwidth=8):
        input_quantizer = Quantizer("input")
        if mask is None:
            # print(weight.data)
            sparse_weight, mask = prune_unstructured(weight, unstructured_sparsity_ratio)

            weight.data = sparse_weight.data

        else:

            sparse_weight = weight
            sparse_weight[mask] = 0.
        ctx.save_for_backward(input, sparse_weight, mask)
        if quantization_en:
            input_for_mul = input.clone()
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(-1, input_for_mul.size(-1))
            input_for_mul = input_quantizer.quantize(input_for_mul, qbitwidth)
            if input.dim() == 3:
                input_for_mul = input_for_mul.reshape(input.shape[0], input.shape[1], -1)

        else:
            input_for_mul = input

        if qbitwidth > 4 and quantization_en:
            if input_for_mul.dim() == 3:
                output = torch.zeros(input_for_mul.shape[0], input_for_mul.shape[1], sparse_weight.t().shape[1],
                                     dtype=torch.float32).cuda()
            else:
                output = torch.zeros(input_for_mul.shape[0], sparse_weight.t().shape[1], dtype=torch.float32).cuda()

            output = torch.matmul(input_for_mul, sparse_weight.t(), out=output)
        else:
            output = torch.matmul(input_for_mul, sparse_weight.t())

        if quantization_en:
            dim = output.dim()
            if output.dim() == 3:
                output_shape = output.shape
                output = output.reshape(-1, output.size(-1))
            output = input_quantizer.dequantize_output(output, sparse_weight, weight_quantizer.scaling_factor)
            if dim == 3:
                output = output.reshape(output_shape[0], output_shape[1], -1)
        output = output.clone()
        return output, mask

    @staticmethod
    def backward(ctx, grad_output, grad_mask):
        input, weight, mask = ctx.saved_tensors
        input_shape = input.shape
        if input.dim() == 3:
            new_batch_size = input_shape[0] * input_shape[1]
            input = input.reshape(new_batch_size, -1)
            grad_output = grad_output.reshape(new_batch_size, -1)
        if input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16 or grad_output.dtype == torch.bfloat16:
            dtype = torch.bfloat16
        elif input.dtype == torch.float16 or weight.dtype == torch.float16 or grad_output.dtype == torch.float16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        grad_weight = torch.matmul(grad_output.t().to(dtype), input.to(dtype))
        grad_weight[mask] = 0.

        #        weight, _ = prune_row_wise(weight)
        grad_input = torch.matmul(grad_output.to(dtype), weight.to(dtype))
        grad_input = grad_input.reshape(input_shape)
        return grad_input, grad_weight, None, None, None, None, None, None


# def sparsify(mat, m, n):
#     reshaped_mat = mat.clone().reshape(-1, n)
#     mask = torch.zeros_like(reshaped_mat, dtype=torch.bool)
#     if (m, n) == (1, 2):
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         rows = (indices == 1).sum(dim=1)
#         mask[:, 0] = rows
#         mask[:, 1] = torch.logical_not(rows)
#     elif (m, n) == (2, 4):
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         rows = torch.logical_not((indices == 0).sum(dim=1))
#         mask[:, 0] = rows
#         rows = torch.logical_not((indices == 1).sum(dim=1))
#         mask[:, 1] = rows
#         rows = torch.logical_not((indices == 2).sum(dim=1))
#         mask[:, 2] = rows
#         rows = torch.logical_not((indices == 3).sum(dim=1))
#         mask[:, 3] = rows
#     elif m < n / 2:
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=m, dim=1, sorted=False, largest=True)
#         for i in range(n):
#             rows = torch.logical_not((indices == i).sum(dim=1))
#             mask[:, i] = rows
#     else:
#         _, indices = torch.topk(torch.abs(reshaped_mat), k=(n - m), dim=1, sorted=False, largest=False)
#         for i in range(n):
#             rows = (indices == i).sum(dim=1)
#             mask[:, i] = rows
#     reshaped_mat[mask] = 0.
#     return reshaped_mat.reshape(mat.shape), mask.reshape(mat.shape)



if __name__ == '__main__':
    N, M = 2, 8
    dtype=torch.half
    mat = torch.randn(1024, 1024, dtype=dtype).to('cuda')
    print(mat)
    print(torch.sum(mat != 0.) / mat.numel())
    mat[0, 0] = 0.
    mat, mask = prune_row_wise(mat, n=N, m=M)
    print(mat)
    print(torch.sum(mat != 0.) / mat.numel())
    mat, mask = prune_column_wise(mat, n=N, m=M)
    print(mat)
    print(torch.sum(mat != 0.) / mat.numel())