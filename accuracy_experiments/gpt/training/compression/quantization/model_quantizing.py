import torch
from torch.nn.parameter import Parameter

def quantize_two_matrices_sum_qbitwidth(A, B, qbitwidth, m):
    # Quantize A into m-bits and B into (qbitwidth - m)-bits
    weight_quantizer_a = Quantizer("weight")
    weight_quantizer_b = Quantizer("weight")
    quantized_a = weight_quantizer_a.quantize(a, m)
    quantized_b = weight_quantizer_b.quantize(b, qbitwidth - m)
    
    # Returns weight quantizer, just in case there is a need to dequantize back
    return quantized_a, quantized_b, weight_quantizer_a, weight_quantizer_b

class Quantizer:

    def __init__(self, matrix_type):
        self.matrix_type = matrix_type

    def quantize(self, mat, num_bits=8):
        if self.matrix_type == "weight":
            return self.quantize_weight(mat, num_bits)

        elif self.matrix_type == "input":
            return self.quantize_input(mat, num_bits)

    def quantize_weight(self, mat, num_bits=8, use_std=False, std_factor=3, max_bitwidth=8):
        """absmax quantization"""
        max_q = 2 ** (num_bits - 1) - 1
        if use_std:
            abs_max = std_factor * torch.sqrt((mat ** 2).mean())
        else:
            abs_max = mat.abs().max()
        scaling_factor = max_q / abs_max
        quantized_mat = torch.round(mat * scaling_factor)
        if use_std:
            max_q = 2 ** max_bitwidth - 1
        quantized_mat = torch.clamp(quantized_mat, -max_q, max_q)

        self.scaling_factor = scaling_factor

        return quantized_mat

    # doesn't change the original matrix
    def dequantize_absmax(self, quantized_mat, scaling_factor=None):
        """absmax dequantization"""
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        deq_mat = quantized_mat / scaling_factor

        return deq_mat

    def quantize_input(self, mat, num_bits=8):
        """ Zero-point quantization for inputs """
        max_q = 2 ** (num_bits - 1) - 1
        mat_max = mat.max(dim=1, keepdim=True)[0]
        mat_min = mat.min(dim=1, keepdim=True)[0]
        scale = (2 * max_q) / (mat_max - mat_min)
        zero_point = -1 * torch.round(scale * mat_min) - max_q
        quantized_mat = torch.round(scale * mat + zero_point)
        quantized_mat = torch.clamp(quantized_mat, -max_q, max_q)

        self.zero_point = zero_point
        self.scaling_factor = scale

        return quantized_mat

    def dequantize_output(self, output, quantized_weight, weight_sf):
        """ Dequantization of the output when input is quantized with the row-wise zero point algorithm and the weight matrix is quantized with absmax"""
        output = self.dequantize_absmax(output, weight_sf)
        weight_transpose = quantized_weight.t()
        deq_weight = self.dequantize_absmax(weight_transpose, weight_sf)
        output -= self.zero_point * torch.sum(deq_weight, dim=0)
        deq_output = self.dequantize_absmax(output, self.scaling_factor)
        return deq_output

    def dequantize_zero_point(self, mat, zero_points, scales):
        """ Zero-point dequantization for inputs """
        return (mat - zero_points) / scales

    def quantize_rowwise_absmax(self, mat, num_bits=8):
        """Row-wise quantization using absmax algorithm"""
        max_q = 2 ** (num_bits - 1) - 1
        abs_max_values, _ = mat.abs().max(dim=1, keepdim=True)
        scaling_factors = max_q / abs_max_values

        quantized_mat = torch.round(mat * scaling_factors)
        quantized_mat = torch.clamp(quantized_mat, -max_q, max_q)

        return quantized_mat, scaling_factors

    def dequantize_rowwise_absmax(self, mat, sf_tensor):
        """Row-wise dequantization using absmax algorithm"""
        dequantized_mat = mat / sf_tensor

        return dequantized_mat


def quantize_model(model,
                   skip_layers=[],
                   quantization_en=False,
                   qbitwidth=8,
                   update_lora=False,
                   accelerate_en=False,
                   is_main_process=lambda: True):

    if is_main_process() and quantization_en:
        print(f"Quantizing the model's weights")
    known_modules = {"Linear", "LinearActivation"}

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in known_modules:
            if module in skip_layers:
                if is_main_process() and quantization_en:
                    print("Skipping Module: ", module)
                continue

            module.quantization_en = quantization_en
            module.qbitwidth = qbitwidth
            module.quantizer = None
            if quantization_en:
                weight_quantizer = Quantizer("weight")
                module.quantizer = weight_quantizer
                module.accelerate = accelerate_en
                # print("weightbeforequant: "+str(module.weight))
                
                # Preserve weight for later if needed
                update_lora_with_error = update_lora and torch.all(module.lora_right.data == 0)
                if update_lora_with_error:
                    original_weight_data = module.weight.data.clone()

                module.weight.data = weight_quantizer.quantize(module.weight, qbitwidth)
                # print("weightafterquant: "+str(module.weight))
                # if hasattr(module, 'weight'):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                
                # Update LoRA left and right's weight based on quantization error if it's initialized to zeros
                if update_lora_with_error:
                    quantization_error = original_weight_data - weight_quantizer.dequantize_absmax(module.weight.data, weight_quantizer.scaling_factor)
                    
                    # Use SVD to decompose Error to left and right, and combine diagonal to both of them. TODO: Should n_iter be a parameter?
                    U, S, V = torch.svd_lowrank(quantization_error, q=module.lora_rank, niter=5, M=None)
                    sqrt_S = torch.sqrt(torch.diag(S))
                    module.lora_left = torch.nn.Parameter(torch.mm(U, sqrt_S)).to(module.weight.device)
                    module.lora_right = torch.nn.Parameter(torch.mm(sqrt_S, V.t())).to(module.weight.device)
