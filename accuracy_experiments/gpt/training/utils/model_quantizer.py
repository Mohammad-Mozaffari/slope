import torch
import layers.quantized_conv2d as quantized_conv2d
import layers.quantized_linear as quantized_linear
import layers.reshaper_linear as reshaper_linear


def add_layer(module, layers, quantize=True, num_bits=8, initial_bit_mask_params=8, device='cuda'):
    classname = module.__class__.__name__
    if classname == 'Linear':
        if quantize:
            layers.append(quantized_linear.QuantizedLinear(module, num_bits=num_bits, initial_bit_mask_params=initial_bit_mask_params, device=device))
        else:
            layers.append(reshaper_linear.ReshaperLinear(module))
    elif quantize:
        if classname == 'Conv2d':
            layers.append(quantized_conv2d.QuantizedConv2D(module, num_bits=num_bits, initial_bit_mask_params=initial_bit_mask_params, device=device))
        else:
            layers.append(module)
    else:
        layers.append(module)


def model_to_list(model, layers=None, quantize=False, num_bits=8, initial_bit_mask_params=8, device='cuda'):
    if layers is None:
        layers = []
    i = 0
    for name, module in model.named_children():
        layers = model_to_list(module, layers, quantize, num_bits, initial_bit_mask_params, device)
        i += 1
    if i == 0:
        add_layer(model, layers, quantize, num_bits, initial_bit_mask_params, device)
    return layers


def quantize_model(model, attribute_list=[], num_bits=8, initial_bit_mask_params=8, device='cuda'):
    i = 0
    for name, module in model.named_children():
        i += 1
        attribute_list.append(name)
        quantize_model(module, num_bits, initial_bit_mask_params, device)
    if i == 0:
        model = torch.nn.Linear(1, 2)
        # if isinstance(model, torch.nn.Linear):
        #     model = quantized_linear.QuantizedLinear(model, num_bits=num_bits, initial_bit_mask_params=initial_bit_mask_params, device=device)
        # elif isinstance(model, torch.nn.Conv2d):
        #     model = quantized_conv2d.QuantizedConv2D(model, num_bits=num_bits, initial_bit_mask_params=initial_bit_mask_params, device=device)
        # else:
        #     model = model
    