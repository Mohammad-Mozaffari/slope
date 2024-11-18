import torch
from transformers import AutoConfig, AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--inference_only", action="store_true")
parser.add_argument("--num_hidden_layers", type=int, default=-1)
parser.add_argument("--add_lora", action="store_true")
parser.add_argument("--lora_rank", type=float, default=0.)
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--cutlass", action="store_true")


optimizer_state_type = torch.float32

args = parser.parse_args()


if args.cutlass:
    SparseSemiStructuredTensor._FORCE_CUTLASS = True


def compute_memory(model_name, prune=False, inference_only=False, add_lora=False):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    init_memory = torch.cuda.memory_allocated()

    config = AutoConfig.from_pretrained(model_name, cache_dir="cache")
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    try:
        model = AutoModelForPreTraining.from_config(config=config).half().cuda()
    except:
        model = AutoModelForCausalLM.from_config(config=config).half().cuda()
    if not inference_only:
        weight_transpose = {}

    def get_skip_layers(model, model_name):
        if "bert" in model_name:
            return [model.cls.predictions.decoder,
                    model.cls.seq_relationship]
        elif "gpt" in model_name:
            return [model.lm_head]
        elif "opt" in model_name:
            return [model.lm_head]
        elif "llama":
            return [model.lm_head]
        elif "mistral":
            return [model.lm_head]
        else:
            raise NotImplementedError

    skip_layers = get_skip_layers(model, model_name)
    lora_rank = int(args.lora_rank * config.hidden_size)
    lora_rank = lora_rank - (lora_rank % 16)

    if prune:
        known_modules = ["Linear", "Conv1D"]
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type in known_modules:
                if module in skip_layers:
                    print("Skipping Layer: ", name)
                    continue
                if not inference_only:
                    module.mask = torch.zeros_like(module.weight, dtype=torch.bool)
                if args.dynamic and module.weight.shape[0] != module.weight.shape[1]:
                    # Storing the dense weight for dynamic masks
                    # We only prune the MLP-Mixer layers in dynamic masking
                    module.weight2 = torch.zeros_like(module.weight) 
                if not (args.dynamic and (module.weight.shape[0] == module.weight.shape[1])):
                    # Storing sparse weights
                    # We only prune the MLP-Mixer layers in dynamic masking
                    output_dim = module.weight.shape[0]
                    if add_lora:
                        output_dim += lora_rank
                    module.weight = torch.nn.Parameter(to_sparse_semi_structured(
                        torch.randn(module.weight.shape[0],
                                    int(module.weight.shape[1]),
                                    dtype=module.weight.dtype,
                                    device=module.weight.device)))

                if not inference_only and (not (args.dynamic and (module.weight.shape[0] == module.weight.shape[1]))):
                    weight_transpose[module.weight] = torch.nn.Parameter(to_sparse_semi_structured(
                        torch.randn(module.weight.shape[1], int(module.weight.shape[0]),
                                    dtype=module.weight.dtype,
                                    device=module.weight.device)))
                if add_lora:
                    module.lora_right = torch.nn.Parameter(
                        torch.zeros(lora_rank, module.weight.shape[0],
                                    dtype=module.weight.dtype,
                                    device=module.weight.device))
        

        if not inference_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            for param in optimizer.param_groups[0]["params"]:
                optimizer.state[param]['step'] = 0
                if param.dim() == 2 and not args.dynamic:
                    optimizer.state[param]['exp_avg'] = torch.zeros(
                        param.shape[0], int(param.shape[1] / 2), dtype=optimizer_state_type, device=param.device)
                    optimizer.state[param]['exp_avg_sq'] = torch.zeros(
                        param.shape[0], int(param.shape[1] / 2), dtype=optimizer_state_type, device=param.device)
                    optimizer.state[param]['grad'] = torch.zeros(
                        param.shape[0], int(param.shape[1] / 2), dtype=param.dtype, device=param.device)
                else:
                    optimizer.state[param]['exp_avg'] = torch.zeros(
                        param.shape, dtype=optimizer_state_type, device=param.device)
                    optimizer.state[param]['exp_avg_sq'] = torch.zeros(
                        param.shape, dtype=optimizer_state_type, device=param.device)
                    optimizer.state[param]['grad'] = torch.zeros(
                        param.shape, dtype=param.dtype, device=param.device)
    else:
        if not inference_only:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            for param in optimizer.param_groups[0]["params"]:
                optimizer.state[param]['step'] = 0
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param, dtype=optimizer_state_type)
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param, dtype=optimizer_state_type)
                optimizer.state[param]['grad'] = torch.zeros_like(param)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()
    return final_memory - init_memory


print("Model: ", args.model_name, "Inference", args.inference_only, "LoRA Rank: ", args.lora_rank)
dense_memory = compute_memory(args.model_name, prune=False,  inference_only=args.inference_only,
                              add_lora=args.add_lora)
memory = torch.cuda.memory_allocated()
print("After Dense Call Memory: ", memory)
sparse_memory = compute_memory(args.model_name, prune=True, inference_only=args.inference_only,
                               add_lora=args.add_lora)
print("Memory Reduction: ", sparse_memory / dense_memory)



if not os.path.exists("memory.csv"):
    df = pd.DataFrame(columns=["Model", "Train/Inference", "Dense Memory", "Sparse Memory", "Memory Saving", "Device",
                               "Add LoRA", "LoRA Rank"])
    df.to_csv("memory.csv", index=False)

df = pd.read_csv("memory.csv")

training = "Train" if not args.inference_only else "Inference"
loc = ((df["Model"] == args.model_name) & (df["Train/Inference"] == training) & (df["Add LoRA"] == args.add_lora) &
       (df["LoRA Rank"] == args.lora_rank))
if not loc.any():
    with open("memory.csv", "a") as f:
        f.write(f"{args.model_name},{training},{dense_memory},{sparse_memory},{sparse_memory / dense_memory},"
                f"{torch.cuda.get_device_name()},{args.add_lora},{args.lora_rank}\n")
else:
    
    df.loc[loc, "Dense Memory"] = dense_memory
    df.loc[loc, "Sparse Memory"] = sparse_memory
    df.loc[loc, "Memory Saving"] = sparse_memory / dense_memory
    df.to_csv("memory.csv", index=False)
