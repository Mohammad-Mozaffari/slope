import torch
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=1024)
parser.add_argument("--num_experiments", type=int, default=1000)
parser.add_argument("--bs", type=int, default=2048)

args = parser.parse_args()

d = args.d
num_experiments = args.num_experiments
bs = args.bs


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

if not os.path.exists("matmul_utilization.csv"):
    with open("matmul_utilization.csv", "w") as f:
        f.write("d,rank,throughput,time\n")

for rank_ratio in [0.01, 0.02, 0.05, 0.1, 1.0]:
    rank = int(rank_ratio * d)
    times = []
    for i in range(num_experiments):
        x = torch.randn(bs, d, dtype=torch.float16, device="cuda")
        l = torch.randn(d, rank, dtype=torch.float16, device="cuda")
        r = torch.randn(rank, d, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        start.record()
        y1 = torch.matmul(x, l)
        y2 = torch.matmul(y1, r)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    throughput = 2 * bs * d * rank / np.median(times) / 1e6
    print(f"Rank: {rank}, Throughput: {throughput} GFLOPS, Time: {np.median(times)}+-{np.std(times)} ms")
    with open("matmul_utilization.csv", "a") as f:
        f.write(f"{d},{rank_ratio},{throughput},{np.median(times)}\n")

        