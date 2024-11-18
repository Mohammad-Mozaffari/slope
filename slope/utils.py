import torch
from pruning_kernels.sparse_backend import pruner


def report_memory_usage(message=""):
    print(f"{message} Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def prune_row_wise(input, n=0, m=0):
    if (n, m) == (0, 0):
        n = 1 if input.dtype == torch.float32 else 2
        m = 2 if input.dtype == torch.float32 else 4
    input = input.contiguous()
    input_shape = input.shape
    sparse_input, mask = pruner.prune(input.reshape(-1, input.shape[-1]), n, m)
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


def error(mat, target):
    return torch.norm(mat.float() - target.float()) / torch.norm(target.float())


def density_ratio(mat):
    return (mat != 0).float().mean()