import torch
import pruning_layers.conv2d as pruning_conv2d
import pruning_layers.linear as pruning_linear
import os


def prune(model, sigmoid_multiplier=10.0, method="learnable", sparsity_ratio=None):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            pruning_linear.prune_linear(module, sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)
        elif isinstance(module, torch.nn.Conv2d):
            pruning_conv2d.prune_conv2d(module, sigmoid_multiplier, method=method, sparsity_ratio=sparsity_ratio)

        prune(module, sigmoid_multiplier, method, sparsity_ratio)


def is_pruned(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            return hasattr(module, "freeze_masks")


def fix_masks(model, threshold=1e-2, save_masks=False, masks_list=None, model_name="model_name"):
    save_file = save_masks and masks_list is not None
    if save_masks and masks_list is None:
        masks_list = []
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            module.fix_masks(threshold)
            if save_masks:
                masks_list.append(module.frozen_weight_mask.clone().detach().cpu())

        fix_masks(module, threshold, save_masks, masks_list)
    
    if save_file:
        os.makedirs(f"./masks/{model_name}", exist_ok=True)
        torch.save(masks_list, f"./masks/{model_name}/masks_list.pt")
        

def accelerate_matmul(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            pruning_linear.accelerate_matmul(module)

        accelerate_matmul(module)

