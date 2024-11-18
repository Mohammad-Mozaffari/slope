import torch
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from slope.slope import prune_model



parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=100)
parser.add_argument("--num_experiments", type=int, default=100)
parser.add_argument("--d_in", type=int, default=1024 * 4)
parser.add_argument("--d_out", type=int, default=4 * 1024 * 4)
parser.add_argument("--bs", type=int, default=2048)
parser.add_argument("--backend", type=str, default="local")
parser.add_argument("--tiling", action="store_true")

args = parser.parse_args()


def time_linear_layers(sparse=False, backend="local"):
    layers = []
    for i in range(args.num_layers // 2):
        layers += [torch.nn.Linear(args.d_in, args.d_out, bias=False).half().cuda(),
                   torch.nn.Linear(args.d_out, args.d_in, bias=False).half().cuda()]

    for layer in layers:
        layer.shape = (layer.weight.shape[1], layer.weight.shape[0])

    model = torch.nn.Sequential(*layers)


    if sparse:
        prune_model(model, backend=backend, tiling=args.tiling)

    linear_times = {}

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_pre_hook(module, input):
        start.record()

    def time_post_hook(module, input, output):
        end.record()
        torch.cuda.synchronize()
        layer_time = start.elapsed_time(end)
        if module.shape in linear_times:
            linear_times[module.shape].append(layer_time)
        else:
            linear_times[module.shape] = [layer_time]


    for layer in layers:
        layer.register_forward_pre_hook(time_pre_hook)
        layer.register_forward_hook(time_post_hook)


    for iter in range(args.num_experiments):
        x = torch.randn(args.bs, args.d_in, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            model(x)
    return linear_times


dense_times = time_linear_layers(sparse=False)
sparse_times = time_linear_layers(sparse=True, backend=args.backend)

for key in dense_times:
    print(key, np.median(dense_times[key]) / np.median(sparse_times[key]))


tiling= "_tiling" if args.tiling else ""
file_name = f"layer_speedup_bs{args.bs}_{args.backend}_backend{tiling}.csv"
if not os.path.exists(file_name):
    with open(file_name, "w") as f:
        f.write("d_in,d_out,Dense Time,Sparse Time,Speedup\n")


with open(file_name, "a") as f:
    for key in dense_times:
        f.write(f"{key[0]},{key[1]},{np.median(dense_times[key])},{np.median(sparse_times[key])},"
                f"{np.median(dense_times[key]) / np.median(sparse_times[key])}\n")