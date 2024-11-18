import torch

def get_skip_layers(model, args):
    skip_layers = set()
    if args.pruner.skip_attention:
        for block in model.transformer.layers:
            skip_layers.update({
                block.mixer.Wqkv,
                block.mixer.out_proj,
            })
    if args.pruner.skip_first_block:
        first_block = model.transformer.layers[0]
        skip_layers.update({
            first_block.mixer.Wqkv,
            first_block.mixer.out_proj,
            block.mlp.fc1,
            block.mlp.fc2,
        })
    if args.pruner.skip_last_block:
        last_block = model.transformer.layers[-1]
        skip_layers.update({
            last_block.mixer.Wqkv,
            last_block.mixer.out_proj,
            block.mlp.fc1,
            block.mlp.fc2,
        })
    skip_layers.update({
                            model.transformer.layers[0].mixer.Wqkv,
                            model.lm_head,
    })
    return skip_layers


def prune_layer(params, sparsity_ratio):
    if sparsity_ratio > 1:
        sparsity_ratio /= 100
    mask = torch.zeros_like(params)
    num_weights = mask.numel()
    num_pruned = int(num_weights * sparsity_ratio)
    _, nonzero_indices = torch.topk(torch.abs(params).view(-1), num_weights - num_pruned)
    mask.view(-1)[nonzero_indices] = 2.0
    return mask


def block_diagonal_mask(shape, num_blocks):
    mask = torch.zeros(shape)
    block_size = shape[0] // num_blocks
    for i in range(num_blocks):
        mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = 1.0
    mask[i * block_size:, i * block_size:] = 1.0
    return mask


