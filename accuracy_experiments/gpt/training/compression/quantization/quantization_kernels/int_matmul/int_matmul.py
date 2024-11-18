import torch
from torch.utils.cpp_extension import load
import os

base_path = __file__.replace("int_matmul.py", "")

if not os.path.exists(f"{base_path}/build"):
        os.makedirs(f"{base_path}/build")

intMatmul = load(name='intMatmul', 
                 sources=[  f'{base_path}/int_matmul.cpp',
                            f'{base_path}/int_matmul.cu'],
                 build_directory=f'{base_path}/build', 
                 verbose=True)

#This function is actually computing AB^T
def int_matmul(a, b):
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "The input tensors should be int8"
    assert a.device == b.device, "The input tensors should be on the same device"
    c = torch.empty(a.size(0), b.size(0), dtype=torch.int32, device=a.device)
    intMatmul.int_matmul(b, a, c)
    return c


if __name__ == "__main__":
    import numpy as np

    dim = 1024
    a = torch.randint(-255, 256, [dim, dim]).to(torch.int8).cuda()
    b = torch.randint(-255, 256, [dim, dim]).to(torch.int8).cuda()
    a = torch.ones(dim, dim).to(torch.int8).cuda()
    b = torch.ones(dim, dim).to(torch.int8).cuda()
    # print(a)
    # print(b)
    cuda_int32 = int_matmul(a, b)
    cpu_int8 = torch.matmul(a.cpu(), b.t().cpu()).cuda()
    cuda_fp32 = torch.matmul(a.float(), b.t().float())

    print(torch.norm(cuda_int32.float() - cuda_fp32))
    print(torch.norm(cuda_int32.float() - cpu_int8.float()))

    num_iterations = 1000
    warmup_cnt = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    results_file = "int_matmul_timings.csv"
    with open(results_file, "w") as f:
        device_name = torch.cuda.get_device_name()
        f.write(f"Matrix Dimension, {device_name} int8 Time, {device_name} fp16 Time, "
                f"{device_name} in8 vs. fp16 Speedup\n")
    # Scalability
    for dim in [512, 1024, 2048, 4098, 8192, 16384, 32768, 65536, 131072, 262144]:
        int_times = []
        fp16_times = []
        for i in range(warmup_cnt + num_iterations):
            a = torch.randint(-255, 256, [dim, dim]).to(torch.int8).cuda()
            b = torch.randint(-255, 256, [dim, dim]).to(torch.int8).cuda()
            start.record()
            cuda_int32 = int_matmul(a, b)
            end.record()
            torch.cuda.synchronize()
            if i >= warmup_cnt:
                int_times.append(start.elapsed_time(end))
            a = torch.randn(dim, dim).to(torch.float16).cuda()
            b = torch.randn(dim, dim).to(torch.float16).cuda()
            start.record()
            cuda_fp16 = torch.matmul(a, b.t())
            end.record()
            torch.cuda.synchronize()
            if i >= warmup_cnt:
                fp16_times.append(start.elapsed_time(end))
        print(f"Dim: {dim}x{dim}:")
        print(f"int32: {np.mean(int_times)} +- {np.std(int_times)}ms, min: {np.min(int_times)}ms, max: {np.max(int_times)}ms, median: {np.median(int_times)}ms")
        print(f"fp16: {np.mean(fp16_times)} +- {np.std(fp16_times)}ms, min: {np.min(fp16_times)}ms, max: {np.max(fp16_times)}ms, median: {np.median(fp16_times)}ms")
        with open(results_file, "a") as f:
            f.write(f"{dim},{np.median(int_times)},{np.median(fp16_times)},{np.median(fp16_times) / np.median(int_times)}\n")
