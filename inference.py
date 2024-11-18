import os
from itertools import chain
import torch
from datasets import load_dataset, load_from_disk
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.testing_utils import CaptureLogger
from slope.slope import prune_model
import tqdm.auto as tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--hf_token", type=str)
parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
parser.add_argument("--num_layers", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--tiling", action="store_true")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cache_dir", type=str, default="cache")
parser.add_argument("--dataset_name", type=str, default="c4")
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--preprocessing_num_workers", type=int, default=None)
parser.add_argument("--overwrite_cache", action="store_true")
parser.add_argument("--block_size", type=int, default=None)
parser.add_argument("--max_samples", type=int, default=30000)
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--backend", type=str, default="cusparselt")

args = parser.parse_args()


if args.dtype == "bf16":
    dtype = torch.bfloat16
elif args.dtype == "fp16":
    dtype = torch.float16
else:
    raise ValueError("dtype must be one of bf16, fp16")


print("Sparse:", args.sparse)


# Set seed
torch.manual_seed(args.seed)


config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir, token=args.hf_token)
if config.num_hidden_layers != args.num_layers and args.num_layers != -1:
    print("Changing the original number of layers to", args.num_layers)
    config.num_hidden_layers = args.num_layers
model = AutoModelForCausalLM.from_config(
    config=config,
    attn_implementation="flash_attention_2",
).to(dtype).cuda()

print(model)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, token=args.hf_token)


if args.sparse:
    prune_model(model, compress=True, tiling=args.tiling, add_lora=True, lora_rank=64, merge_lora=True,
                backend=args.backend)





################################################################################################################
if os.path.exists(f"{args.cache_dir}/c4-raw.pt"):
    raw_datasets = load_from_disk(f"{args.cache_dir}/c4-raw.pt")
else:
    try:
        raw_datasets = load_dataset('allenai/c4',
                                    'allenai--c4',
                                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                    cache_dir=args.cache_dir)
    except:
        raw_datasets = load_dataset('allenai/c4',
                                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                    cache_dir=args.cache_dir)

    raw_datasets.save_to_disk(f"{args.cache_dir}/c4-raw.pt")

if "validation" not in raw_datasets.keys():
    raw_datasets["validation"] = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[:{args.validation_split_percentage}%]",
        streaming=args.streaming,
    )
    raw_datasets["train"] = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[{args.validation_split_percentage}%:]",
        streaming=args.streaming,
    )
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

# Preprocessing the datasets.
# First we tokenize all the texts.



column_names = list(raw_datasets["validation"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output


if not args.streaming:
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
else:
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

if args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        print(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
            " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--block_size xxx`."
        )
        block_size = 1024
else:
    if args.block_size > tokenizer.model_max_length:
        print(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(args.block_size, tokenizer.model_max_length)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that global_batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

if not args.streaming:
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )
else:
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )



if "validation" not in tokenized_datasets:
    raise ValueError("--do_eval requires a validation dataset")
eval_dataset = lm_datasets["validation"]
if args.max_samples is not None:
    max_eval_samples = min(len(eval_dataset), args.max_samples)
    eval_dataset = eval_dataset.select(range(max_eval_samples))

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)




################################################################################################################
model.config.use_cache = False
with torch.no_grad():
    total_loss = 0.0
    device = "cuda"
    progress_bar = tqdm.tqdm(range(len(eval_dataset) // args.batch_size))
    for i, iter in enumerate(progress_bar):
        start_idx = iter * args.batch_size
        batch = eval_dataset[start_idx:start_idx + args.batch_size]
        if args.batch_size == 1:
            batch["input_ids"] = torch.tensor(batch["input_ids"], device=device).unsqueeze(0)
            batch["labels"] = torch.tensor(batch["labels"], device=device).unsqueeze(0)
            batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.half,
                                                   device=device).unsqueeze(0)
        else:
            batch["input_ids"] = torch.tensor(batch["input_ids"], device=device)
            batch["labels"] = torch.tensor(batch["labels"], device=device)
            batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.half, device=device)


        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        progress_bar.set_postfix_str(f"Loss {(total_loss / (i + 1)):.2f}")