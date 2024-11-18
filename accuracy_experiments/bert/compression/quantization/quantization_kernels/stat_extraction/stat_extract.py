import torch
from torch.utils.cpp_extension import load
import os
import matplotlib.pyplot as plt

base_path = "."

if not os.path.exists(base_path + "/quantization/kernels/stat_extraction/build"):
    os.makedirs(base_path + "/quantization/kernels/stat_extraction/build")

extractor = load(name='extractor', sources=[base_path + '/quantization/kernels/stat_extraction/stat_extract.cpp',
                                            base_path + '/quantization/kernels/stat_extraction/stat_extract.cu'],
                 build_directory=base_path + "/quantization/kernels/stat_extraction/build", verbose=True)

# base_path = "."
#
# if not os.path.exists(base_path + "/build"):
#     os.makedirs(base_path + "/build")
#
# extractor = load(name='extractor', sources=[base_path + '/stat_extract.cpp',
#                                             base_path + '/stat_extract.cu'],
#                  build_directory=base_path + "/build", verbose=True)
def get_min_max(mat):
    mat_dtype = mat.dtype
    if mat_dtype == torch.float16 or mat_dtype == torch.float32:
        mins = torch.empty(mat.size(0), 1, dtype=mat_dtype, device=mat.device)
        maxs = torch.empty(mat.size(0), 1, dtype=mat_dtype, device=mat.device)
        extractor.get_min_max(mat, mins, maxs)
    else:
        raise ValueError("Invalid data type (the torch tensor has to be either of type float16 or float32)")
    return mins, maxs

if __name__ == '__main__':
    pytorch_times = []
    kernel_times = []
    sizes = []
    dtypes = [torch.float16, torch.float32]
    for dtype in dtypes:
        for s in range(8,16):
            sizes.append(2**s)
            mat = torch.rand((2**s,2**s), dtype=dtype, device="cuda")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for i in range(500):
                mat_max = mat.max(dim=1, keepdim=True)[0]
                mat_min = mat.min(dim=1, keepdim=True)[0]

            start_event.record()
            for i in range(500):
                mat_max = mat.max(dim=1, keepdim=True)[0]
                mat_min = mat.min(dim=1, keepdim=True)[0]
            end_event.record()
            torch.cuda.synchronize()
            stat_time = start_event.elapsed_time(end_event)/500
            for i in range(500):
                ker_min, ker_max = get_min_max(mat)

            start_event.record()
            for i in range(500):
                ker_min, ker_max = get_min_max(mat)
            end_event.record()
            torch.cuda.synchronize()
            stat_time2 = start_event.elapsed_time(end_event)/500
            print(2**s)
            print((ker_max == mat_max).all())
            print((ker_min == mat_min).all())

            pytorch_times.append(stat_time)
            kernel_times.append(stat_time2)

    print(kernel_times)
    print(pytorch_times)
    fig, ax = plt.subplots()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Speedup')
    plt.title('Custom Min/Max Kernel vs Pytorch')
    result = [x / y for x, y in zip(pytorch_times[0:8], kernel_times[0:8])]
    result2 = [x / y for x, y in zip(pytorch_times[8:], kernel_times[8:])]
    ax.semilogx(sizes[0:8], result, color='blue', label="float16 kernel")
    ax.semilogx(sizes[8:], result2, color='green', label="float32 kernel")
    plt.legend()
    plt.show()



