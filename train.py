import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
from datasets import load_dataset, load_from_disk
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version
from optimizers.adamw_sparse import ADAMWSparse, ADAMW
from slope.slope import (prune_model, grad_dict, replace_linear_layers, sync_grads)
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import tqdm.auto as tqdm
import argparse


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."



parser = argparse.ArgumentParser()
parser.add_argument("--hf_token", type=str)
parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
parser.add_argument("--num_layers", type=int, default=-1)
parser.add_argument("--local_batch_size", type=int, default=2)
parser.add_argument("--global_batch_size", type=int, default=64)
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--manual_optimizer", action="store_true")
parser.add_argument("--tiling", action="store_true")
parser.add_argument("--use_huggingface_trainer", action="store_true")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cache_dir", type=str, default="cache")
parser.add_argument("--dataset_name", type=str, default="c4")
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--validation_split_percentage", type=int, default=5)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--preprocessing_num_workers", type=int, default=None)
parser.add_argument("--overwrite_cache", action="store_true")
parser.add_argument("--block_size", type=int, default=None)
parser.add_argument("--max_train_samples", type=int, default=30000)
parser.add_argument("--max_eval_samples", type=int, default=128)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--backend", type=str, default="cusparselt")


args = parser.parse_args()


if args.dtype == "fp16" and args.use_huggingface_trainer:
    raise ValueError("f16 is only supported with the HuggingFace Trainer")

if args.dtype == "bf16":
    dtype = torch.bfloat16
    bf16 = True
elif args.dtype == "fp16":
    dtype = torch.float16
    bf16 = False
else:
    raise ValueError("dtype must be one of bf16, fp16")


print("Sparse:", args.sparse)
print("Manual Optimizer:", args.manual_optimizer)


local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
world_rank = int(os.environ['RANK'])
torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=world_rank)


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
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, token=args.hf_token)


if args.sparse:
    prune_model(model, compress=True, tiling=args.tiling, backend=args.backend)
    sparse_optimizer = ADAMWSparse(model, grad_dict)
else:
    pass
    # Uncomment the following lines for debugging
    # prune_model(model, compress=False)
    # replace_linear_layers(model, manual_optimizer=args.manual_optimizer)
    # sparse_optimizer = ADAMW(model, grad_dict)

optimizer = "adamw_torch"


class SparseOptimizerCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        sparse_optimizer.clip_grad_norm(max_norm=args.max_grad_norm)
        sparse_optimizer.weight_decay = kwargs['optimizer'].param_groups[0]['weight_decay']
        sparse_optimizer.lr = kwargs['lr_scheduler'].get_last_lr()[0]
        sparse_optimizer.step()
        sparse_optimizer.zero_grad()

training_args = TrainingArguments(
    max_grad_norm=torch.inf,
    output_dir="output",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=args.local_batch_size,
    per_device_eval_batch_size=args.local_batch_size,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=1,
    eval_steps=1,
    save_steps=5000,
    save_total_limit=1,
    bf16=bf16,
    fp16=not bf16,
    group_by_length=False,
    gradient_accumulation_steps=args.global_batch_size // args.local_batch_size,
    warmup_steps=5,
    optim=optimizer,
    save_strategy="steps",
    report_to="none",
    gradient_checkpointing=True,
)
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


if training_args.do_train:
    column_names = list(raw_datasets["train"].features)
else:
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


with training_args.main_process_first(desc="Dataset map tokenization"):
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

with training_args.main_process_first(desc="Grouping texts together"):
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

if training_args.do_train:
    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        samples_per_machine = max_train_samples // world_size
        sample_range = range(samples_per_machine * world_rank, samples_per_machine * (world_rank + 1))
        train_dataset = train_dataset.select(range(max_train_samples))

if training_args.do_eval:
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = None

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)


################################################################################################################
model.module.config.use_cache = False
if args.use_huggingface_trainer and not bf16:
    if training_args.do_train:
        model.train()
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval
            else None,
            callbacks=[SparseOptimizerCallback],
        )
        train_result = trainer.train()
        metrics = train_result.metrics

        max_train_samples = (
            max_train_samples if max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
else:
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataset) // args.global_batch_size)
    # Manual training loop
    for epoch in range(training_args.num_train_epochs):
        total_loss = 0.0
        device = f"cuda:{local_rank}"
        progress_bar = tqdm.tqdm(range(len(train_dataset) // args.local_batch_size),
                                 desc=f"Epoch {epoch}")
        for i, iter in enumerate(progress_bar):
            start_idx = iter * args.local_batch_size
            batch = train_dataset[start_idx:start_idx + args.local_batch_size]
            if args.local_batch_size == 1:
                batch["input_ids"] = torch.tensor(batch["input_ids"], device=device).unsqueeze(0)
                batch["labels"] = torch.tensor(batch["labels"], device=device).unsqueeze(0)
                batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.half,
                                                       device=device).unsqueeze(0)
            else:
                batch["input_ids"] = torch.tensor(batch["input_ids"], device=device)
                batch["labels"] = torch.tensor(batch["labels"], device=device)
                batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.half, device=device)

            if i % training_args.gradient_accumulation_steps != training_args.gradient_accumulation_steps - 1:
                with model.no_sync():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    total_loss += loss.item()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                sync_grads()
                total_loss += loss.item()
                optimizer.zero_grad()
                sparse_optimizer.weight_decay = optimizer.param_groups[0]['weight_decay']
                sparse_optimizer.lr = scheduler.get_last_lr()[0]
                sparse_optimizer.step()
                sparse_optimizer.zero_grad()
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix_str(f"Loss {(total_loss / (i + 1)):.2f} - LR {scheduler.get_last_lr()[0]:.2e}")
