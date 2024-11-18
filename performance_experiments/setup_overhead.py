import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Measure the setup overhead of to_sparse_semi_structured')
parser.add_argument('--d', type=int, default=1024, help='Dimension of the matrix')
parser.add_argument('--b', type=int, default=2048, help='Batch size')
parser.add_argument('--num_experiments', type=int, default=1000, help='Number of experiments')

args = parser.parse_args()

d = args.d
b = args.b
num_experiments = args.num_experiments

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

matmul_times = []
setup_times = []

for i in range(num_experiments):
    w = torch.randn(d, d, device='cuda', dtype=torch.float16)
    x = torch.randn(b, d, device='cuda', dtype=torch.float16)
    start.record()
    torch.cuda.synchronize()
    w_sparse = to_sparse_semi_structured(w)
    end.record()
    torch.cuda.synchronize()
    setup_times.append(start.elapsed_time(end))
    start.record()
    y = torch.matmul(x, w)
    end.record()
    torch.cuda.synchronize()
    matmul_times.append(start.elapsed_time(end))

print(f"d: {d}, b: {b}, Matmul Time: {np.median(matmul_times)} ms, Setup Time: {np.median(setup_times)} ms")

if not os.path.exists('setup_overhead.csv'):
    with open('setup_overhead.csv', 'w') as f:
        f.write('d,b,Multiplication Time,Setup Time\n')
    
with open('setup_overhead.csv', 'a') as f:
    f.write(f'{d},{b},{np.median(matmul_times)},{np.median(setup_times)}\n')