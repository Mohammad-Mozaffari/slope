import torch


def lora_hook(module, input, output):
    output = output + torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank


def add_lora(model, disable_grads=True, rank=4, skip_layers=[]):
    print("Replacing weights with LoRA")
    known_modules = {"Linear", "LinearActivation"}
    for name, module in model.named_modules():
        if type(module).__name__ in known_modules:
            if module in skip_layers:  
                print("Skipping Module: ", module)
                continue

            module.lora_left = torch.nn.Parameter(torch.randn(module.weight.shape[1], rank)).to(module.weight.device)
            module.lora_right = torch.nn.Parameter(torch.zeros(rank, module.weight.shape[0])).to(module.weight.device)
            module.lora_rank = rank
            if disable_grads:
                module.weight.requires_grad = False
            
            if type(module).__name__ in ["Linear", "LinearActivation"]:
                module.register_forward_hook(lora_hook)
            else:
                raise NotImplementedError("LoRA not implemented for this module type")