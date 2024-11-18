import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.prune import fix_masks


def parameter_regularization(model, sigmoid_multiplier=10):
    sum = 0.0
    for param in model.parameters(): # parameters() returns a generator
        if param.requires_grad:
            sum += torch.sum(torch.sigmoid((param - 1) * sigmoid_multiplier))
    return sum


def weight_norm_regularization(model):
    sum = 0.0
    for param in model.parameters():
        if param.requires_grad:
            sum += torch.sum(param ** 2)
    return sum


def compute_sparsity_ratio(model, sigmoid_multiplier=20, model_name="", threshold=1e-2, plot_histogram=False, plot_layer_ratio=False):
    sum = 0
    total = 0
    if plot_layer_ratio:
        layer_ratios = []
    if plot_histogram:
        hist = np.zeros(100)
        hist_range = np.linspace(0, 1, 100)
    for name, param in model.named_parameters():
        if "mask" in name:
            mask = torch.sigmoid((param - 1) * sigmoid_multiplier)
            sum += torch.sum(mask < threshold)
            total += torch.numel(param)
            if plot_layer_ratio:
                layer_ratios.append((torch.sum(mask < threshold) / torch.numel(param)).item())
            if plot_histogram:
                hist += np.histogram(mask.clone().detach().view(-1).cpu().numpy(), bins=100, range=(0, 1))[0]
                
    if plot_layer_ratio:
        plt.bar(np.arange(1, len(layer_ratios) + 1), layer_ratios)
        plt.title(f"{model_name} Layer Sparsity Ratios")
        plt.xlabel("Layer")
        plt.ylabel("Sparsity Ratio")
        plt.show()
    if plot_histogram:
        plt.semilogy(hist_range, hist)
        plt.xlim(0, 1)
        plt.ylim(0)
        plt.title(f"{model_name} Mask Histogram")
        plt.xlabel("Mask Value")
        plt.ylabel("Number of Weights")
        plt.show()
    return sum / total


def find_optimal_threshold(model, model_name, test_loader, test_func):
    baseline_test_acc, baseline_test_loss = test_func(model, test_loader)
    min_threshold, max_threshold = 0, 1
    torch.save(model.state_dict(), "tmp.t7")
    for i in range(10):
        model.load_state_dict(torch.load("tmp.t7"))
        threshold = (min_threshold + max_threshold) / 2
        print("Threshold: ", threshold)
        fix_masks(model, threshold=threshold)
        test_acc, test_loss = test_func(model, test_loader)
        if abs(baseline_test_acc - test_acc) < 0.5:
            min_threshold = threshold
        else:
            max_threshold = threshold
    model.load_state_dict(torch.load("tmp.t7"))
    os.remove("tmp.t7")
    fix_masks(model, threshold=threshold, save_masks=True, model_name=model_name)
    model_sparsity_ratio = float(compute_sparsity_ratio(model, threshold=threshold)) * 100
    best_sparsity_ratio = model_sparsity_ratio
    print("Best Sparsity Ratio: ", best_sparsity_ratio)
    return model_sparsity_ratio, best_sparsity_ratio