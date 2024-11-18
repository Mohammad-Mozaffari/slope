from torch.utils.cpp_extension import load
import os
import torch

base_path = __file__.replace("sparse_backend.py", "")


if not os.path.exists(f"{base_path}/build"):
    os.makedirs(f"{base_path}/build")

pruner = load(name='pruner',
              sources=[f'{base_path}/sparse_backend.cpp',
                       f'{base_path}/sparse_backend.cu',
                       ],
              build_directory=f'{base_path}/build',
              with_cuda=True,
              verbose=False)

