# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT pretraining runner."""

import argparse
import json
import loggerplus as logger
# import wandb
import math
import multiprocessing
import numpy as np
import os
import random
import signal
import time
import warnings
import torch

try:
    import kfac
except:
    pass
from apex.optimizers import FusedLAMB
from optimizers.lamb import LAMBOptimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pathlib import Path

import src.modeling as modeling

from src.dataset import ShardedPretrainingDataset, DistributedSampler
from src.schedulers import PolyWarmUpScheduler, LinearWarmUpScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.tokenization import get_wordpiece_tokenizer, get_bpe_tokenizer
from src.utils import is_main_process, get_world_size, get_rank

import optimizers.mkor as mkor
# import optimizers.kaisa as kaisa
import optimizers.eva as eva
import optimizers.backend as eva_backend
import utils.comm as comm
from pruning.model_pruning import prune_model, set_n_m
from pruning.lora import add_lora
from run_utils import get_skip_layers
from torch.optim import SGD, Adam

try:
    from torch.cuda.amp import autocast, GradScaler

    TORCH_FP16 = True
except:
    TORCH_FP16 = False

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

model_info = {}


# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True


# wandb_run = None


signal.signal(signal.SIGTERM, signal_handler)


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, masked_lm_labels,
                seq_relationship_score=None, next_sentence_labels=None):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size),
                                      masked_lm_labels.view(-1))
        if seq_relationship_score is not None and next_sentence_labels is not None:
            next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                              next_sentence_labels.view(-1))
            return masked_lm_loss + next_sentence_loss
        return masked_lm_loss


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Optional json config to override defaults below
    parser.add_argument("--config_file", default=None, type=str,
                        help="JSON config for overriding defaults")

    ## Required parameters. Note they can be provided in the json
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The input data dir containing .hdf5 files for the task.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--model_config_file", default=None, type=str,
                        help="The BERT model config")

    ## Dynamic Masking Parameters
    parser.add_argument("--masked_token_fraction", type=float, default=0.2,
                        help='Fraction of tokens to mask per sequence')
    parser.add_argument("--max_predictions_per_seq", type=int, default=80,
                        help='Maximum masked tokens per sequence')

    ## Training Configuration
    parser.add_argument('--disable_progress_bar', default=False, action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--num_steps_per_checkpoint', type=int, default=100,
                        help="Number of update steps between writing checkpoints.")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--checkpoint_activations', default=False, action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument('--log_prefix', type=str, default='logfile',
                        help='Prefix for log files. This is just the prefix of '
                             'name and should not contain directories')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Use PyTorch AMP training")
    parser.add_argument('--pruned_matrix', default="none", type=str,
                        help="The matrix to prune")
    parser.add_argument('--skip_first_block', default="False", type=str,
                        help="Skip the first block when pruning")
    parser.add_argument('--skip_last_block', default="False", type=str,
                        help="Skip the last block when pruning")
    parser.add_argument('--skip_attention', default="False", type=str,
                        help="Skip the attention block when pruning")
    parser.add_argument('--reduction_dim', default="True", type=str,
                        help="Apply pruning on the reduction dimension")
    parser.add_argument('--add_lora', default="False", type=str,
                        help="Add LoRA to the weights")
    parser.add_argument('--lora_rank', default=4, type=int,
                        help="LoRA rank")
    parser.add_argument('--sparsity_increment', type=str, default="",
                        help="Block numbers to increase sparsity")

    ## Hyperparameters
    parser.add_argument("--optimizer", default="lamb", type=str,
                        help="The name of the optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--lr_decay", default='poly', type=str,
                        choices=['poly', 'linear', 'cosine'],
                        help="Learning rate decay type.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Proportion of training to perform linear learning rate "
                             "warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--global_batch_size", default=2 ** 16, type=int,
                        help="Global batch size for training.")
    parser.add_argument("--local_batch_size", default=8, type=int,
                        help="Per-GPU batch size for training.")
    parser.add_argument("--max_steps", default=1000, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--previous_phase_end_step", default=0, type=int,
                        help="Final step of previous phase")

    ## KFAC Hyperparameters
    parser.add_argument('--kfac', default=False, action='store_true',
                        help='Use KFAC')
    parser.add_argument('--kfac_inv_interval', type=int, default=10,
                        help='iters between kfac inv ops')
    parser.add_argument('--kfac_factor_interval', type=int, default=1,
                        help='iters between kfac cov ops')
    parser.add_argument('--kfac_stat_decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation')
    parser.add_argument('--kfac_damping', type=float, default=0.001,
                        help='KFAC damping factor')
    parser.add_argument('--kfac_kl_clip', type=float, default=0.001,
                        help='KFAC KL gradient clip')
    parser.add_argument('--kfac_skip_layers', nargs='+', type=str,
                        default=['BertLMPredictionHead', 'embedding'],
                        help='Modules to ignore registering with KFAC '
                             '(default: [BertLMPredictionHead, embedding])')
    parser.add_argument('--base_step', type=int, default=8602)
    parser.add_argument('--init_step', type=int, default=0)

    # Set by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    # parser.add_argument('--name', type=str, default="default",
    #                     help='name of the run for wandb')

    args = parser.parse_args()

    for arg in args.__dict__:
        if type(getattr(args, arg)) == str:
            if getattr(args, arg) == "True":
                setattr(args, arg, True)
            elif getattr(args, arg) == "False":
                setattr(args, arg, False)
            else:
                continue

    # Hacky way to figure out to distinguish arguments that were found
    # in sys.argv[1:]
    aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg in vars(args):
        aux_parser.add_argument('--' + arg)
    cli_args, _ = aux_parser.parse_known_args()

    # Argument precedent: cli_args > config_file > argparse defaults
    if args.config_file is not None:
        with open(args.config_file) as jf:
            configs = json.load(jf)
        for key in configs:
            if key in vars(args) and key not in vars(cli_args):
                setattr(args, key, configs[key])

    return args


def setup_training(args):
    assert (torch.cuda.is_available())

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args.model_output_dir = os.path.join(args.output_dir, 'pretrain_ckpts')
    if is_main_process():
        os.makedirs(args.model_output_dir, exist_ok=True)

    logger.init(
        handlers=[
            logger.StreamHandler(verbose=is_main_process()),
            logger.FileHandler(
                os.path.join(args.output_dir, args.log_prefix + '.txt'),
                overwrite=False, verbose=is_main_process()),
            logger.TorchTensorboardHandler(
                os.path.join(args.output_dir, 'tensorboard'),
                verbose=is_main_process()),
            logger.CSVHandler(
                os.path.join(args.output_dir, args.log_prefix + '_metrics.csv'),
                overwrite=False, verbose=is_main_process()),
        ]
    )
    # if is_main_process():
    #     wandb.login()
    #     phase = 1 + (args.previous_phase_end_step > 0)
    #     with open(args.model_config_file) as f:
    #         config = json.load(f)
    #     wandb_run = wandb.init(
    #         project=f"BERT-Large-Uncased-Phase{phase}",
    #         name=args.name,
    #         config=config,
    #         resume=True
    #     )

    # logger.info('Torch distributed initialized (world_size={}, backend={})'.format(
    #     get_world_size(), torch.distributed.get_backend()))

    if not TORCH_FP16 and args.fp16:
        raise ValueError('FP16 training enabled but unable to import torch.cuda.amp.'
                         'Is the torch version >= 1.6?')

    # if args.global_batch_size % get_world_size() != 0 and is_main_process():
    #     warnings.warn('global_batch_size={} is not divisible by world_size={}.'
    #                   ' The last batch will be padded with additional '
    #                   'samples.'.format(
    #         args.global_batch_size, get_world_size()))
    # args.local_accumulated_batch_size = math.ceil(
    #     args.global_batch_size / get_world_size())

    # if args.local_accumulated_batch_size % get_world_size() != 0 and is_main_process():
    #     warnings.warn('local_accumulated_batch_size={} is not divisible '
    #                   'by local_batch_size={}. local_accumulated_batch_size '
    #                   'is global_batch_size // world_size. The last '
    #                   'batch will be padded with additional samples'.format(
    #         args.local_accumulated_batch_size, get_world_size()))
    # args.accumulation_steps = math.ceil(
    #     args.local_accumulated_batch_size / args.local_batch_size)

    return args


def prepare_model(args):
    config = modeling.BertConfig.from_json_file(args.model_config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config).cpu()
    base_model = modeling.BertForPreTraining(config).cpu()

    if args.pruned_matrix != "none":
        skip_layers = get_skip_layers(model, args)
        prune_model(model, skip_layers=skip_layers, pruned_matrix=args.pruned_matrix, reduction_dim=args.reduction_dim,
                    add_lora=args.add_lora, lora_rank=args.lora_rank)
        set_n_m(model, sparsity_increment=args.sparsity_increment)

        skip_layers = get_skip_layers(base_model, args)
        prune_model(base_model, skip_layers=skip_layers, pruned_matrix=args.pruned_matrix, reduction_dim=args.reduction_dim,
                    add_lora=args.add_lora, lora_rank=args.lora_rank)
        set_n_m(base_model, sparsity_increment=args.sparsity_increment)

    checkpoint = None
    global_steps = 0


    base_checkpoint = torch.load(
        os.path.join(args.model_output_dir, "ckpt_{}.pt".format(args.base_step)),
        map_location="cpu"
    )

    base_model.load_state_dict(base_checkpoint['model'], strict=False)
    base_model.cpu()
    base_model.checkpoint_activations(args.checkpoint_activations)

    with open("checkpoint_similarity.csv", "w") as f:
        f.write("Iteration,")
        for i in range(len(model.bert.encoder.layer)):
            if not args.add_lora:
                f.write(f"Query{i},Key{i},Value{i},Projection{i},Upsample{i},Downsample{i},")
            else:
                f.write(f"Query{i},Query{i}_lora,Key{i},Key{i}_lora,Value{i},Value{i}_lora,Projection{i},Projection{i}_lora,Upsample{i},Upsample{i}_lora,Downsample{i},Downsample{i}_lora,")
        f.write("\n")

    for checkpoint_number in range(args.init_step, args.base_step, 100):
        print("Comparing checkpoint {}".format(checkpoint_number))
        checkpoint = torch.load(
            os.path.join(args.model_output_dir, "ckpt_{}.pt".format(checkpoint_number)),
            map_location="cpu"
        )

        model.load_state_dict(checkpoint['model'], strict=False)

        model.cpu()
        model.checkpoint_activations(args.checkpoint_activations)

        def cosine_similarity(mat1, mat2):
            return str((torch.sum(mat1 * mat2) / (torch.norm(mat1) * torch.norm(mat2))).item())

        def get_lora_similarity(module1, module2):
            if hasattr(module1, "lora_left"):
                result = cosine_similarity(torch.cat([module1.lora_left, module1.lora_right.t()]), torch.cat([module2.lora_left, module2.lora_right.t()]))
            else:
                result = "1.0"
            return result

        with open("checkpoint_similarity.csv", "a") as f:
            f.write(str(checkpoint_number) + ",")
            for i in range(len(model.bert.encoder.layer)):
                f.write(cosine_similarity(model.bert.encoder.layer[i].attention.self.query.weight,
                                        base_model.bert.encoder.layer[i].attention.self.query.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].attention.self.query, base_model.bert.encoder.layer[i].attention.self.query) + ",")
                f.write(cosine_similarity(model.bert.encoder.layer[i].attention.self.key.weight,
                                        base_model.bert.encoder.layer[i].attention.self.key.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].attention.self.key, base_model.bert.encoder.layer[i].attention.self.key) + ",")
                f.write(cosine_similarity(model.bert.encoder.layer[i].attention.self.value.weight,
                                        base_model.bert.encoder.layer[i].attention.self.value.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].attention.self.value, base_model.bert.encoder.layer[i].attention.self.value) + ",")
                f.write(cosine_similarity(model.bert.encoder.layer[i].attention.output.dense.weight,
                                        base_model.bert.encoder.layer[i].attention.output.dense.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].attention.output.dense, base_model.bert.encoder.layer[i].attention.output.dense) + ",")
                f.write(cosine_similarity(model.bert.encoder.layer[i].intermediate.dense_act.weight,
                                        base_model.bert.encoder.layer[i].intermediate.dense_act.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].intermediate.dense_act, base_model.bert.encoder.layer[i].intermediate.dense_act) + ",")
                f.write(cosine_similarity(model.bert.encoder.layer[i].output.dense.weight,
                                        base_model.bert.encoder.layer[i].output.dense.weight) + ",")
                if args.add_lora:
                    f.write(get_lora_similarity(model.bert.encoder.layer[i].output.dense, base_model.bert.encoder.layer[i].output.dense) + ",")
            f.write("\n")

    # model = DDP(model, device_ids=[args.local_rank])

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, checkpoint, global_steps, criterion, args




def main(args):
    global timeout_sent

    model, checkpoint, global_steps, criterion, args = prepare_model(args)
    


if __name__ == "__main__":
    args = parse_arguments()

    if args.model_config_file is None:
        raise ValueError('--model_config_file must be provided via arguments '
                         'or the config file')

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    args = setup_training(args)
    logger.info('TRAINING CONFIG: {}'.format(args))
    with open(args.model_config_file) as f:
        logger.info('MODEL CONFIG: {}'.format(json.load(f)))

    start_time = time.time()
    main(args)