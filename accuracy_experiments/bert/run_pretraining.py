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
from compression.model_compression import prune_model, set_n_m
from compression.quantization.model_quantizing import quantize_model
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
    parser.add_argument('--lora_weight_decay', default=0.0001, type=float,
                        help="Add LoRA weight decay")
    parser.add_argument('--unstructured_masking', default="False", type=str,
                        help="Apply unstructured masking")
    parser.add_argument('--unstructured_mask_sparsity', default=60, type=int,
                        help="Unstructured mask sparsity")
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
    ## Quantization parameters
    parser.add_argument("--quantization_en", type=str, default="False",
                        help='Use quantization')
    parser.add_argument("--qbitwidth", type=int, default=8,
                        help='The bitwidth to which the quantization changes the type')
    parser.add_argument("--accelerate", type=str, default="False",
                        help="Whether to use int8 matmul")

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

    args.local_rank = int(os.environ['LOCAL_RANK'])
    
    return args


def setup_training(args):
    assert (torch.cuda.is_available())

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

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

    logger.info('Torch distributed initialized (world_size={}, backend={})'.format(
        get_world_size(), torch.distributed.get_backend()))

    if not TORCH_FP16 and args.fp16:
        raise ValueError('FP16 training enabled but unable to import torch.cuda.amp.'
                         'Is the torch version >= 1.6?')

    if args.global_batch_size % get_world_size() != 0 and is_main_process():
        warnings.warn('global_batch_size={} is not divisible by world_size={}.'
                      ' The last batch will be padded with additional '
                      'samples.'.format(
            args.global_batch_size, get_world_size()))
    args.local_accumulated_batch_size = math.ceil(
        args.global_batch_size / get_world_size())

    if args.local_accumulated_batch_size % get_world_size() != 0 and is_main_process():
        warnings.warn('local_accumulated_batch_size={} is not divisible '
                      'by local_batch_size={}. local_accumulated_batch_size '
                      'is global_batch_size // world_size. The last '
                      'batch will be padded with additional samples'.format(
            args.local_accumulated_batch_size, get_world_size()))
    args.accumulation_steps = math.ceil(
        args.local_accumulated_batch_size / args.local_batch_size)

    return args


def prepare_model(args):
    config = modeling.BertConfig.from_json_file(args.model_config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    if args.pruned_matrix != "none":
        skip_layers = get_skip_layers(model, args)
        prune_model(model, skip_layers=skip_layers, pruned_matrix=args.pruned_matrix, reduction_dim=args.reduction_dim,
                    add_lora=args.add_lora, lora_rank=args.lora_rank, unstructured_masking = args.unstructured_masking, unstructured_mask_sparsity = args.unstructured_mask_sparsity)
        set_n_m(model, sparsity_increment=args.sparsity_increment)

    def save_activations(module, input):
        reshaped_input = input[0].reshape(-1, input[0].shape[-1])
        # model_info[module]["activation_cov"] += reshaped_input.t().mm(reshaped_input).detach() / reshaped_input.shape[0]
        model_info[module]["activation_norm"] += reshaped_input.norm(p=2, dim=0).detach() ** 2
        model_info[module]["count"] += 1

    def save_grads(module, grad_input, grad_output):
        reshaped_grad_output = grad_output[0].reshape(-1, grad_output[0].shape[-1])
        # model_info[module]["grad_cov"] += reshaped_grad_output.t().mm(reshaped_grad_output).detach() / \
        #                                   reshaped_grad_output.shape[0]

    def generate_info_dict(module, name):
        return {"name": name, "activation_norm": torch.zeros(module.weight.shape[1]).to(args.device),
                # "activation_cov": torch.zeros([module.weight.shape[1]] * 2).to(args.device),
                # "grad_cov": torch.zeros([module.weight.shape[0]] * 2).to(args.device),
                "count": 0, "grad_norm": 0.,
                "first_order_effect": 0., "second_order_only_effect": 0., "second_order_effect": 0., "weight_norm": 0.}

    def attach_hooks(module, name):
        model_info[module] = generate_info_dict(module, name)
        module.register_forward_pre_hook(save_activations)
        module.register_backward_hook(save_grads)

    # for i in range(len(model.bert.encoder.layer)):
    #     block = model.bert.encoder.layer[i]
    #     attach_hooks(block.attention.self.query, f"query_{i}")
    #     attach_hooks(block.attention.self.key, f"key_{i}")
    #     attach_hooks(block.attention.self.value, f"value_{i}")
    #     attach_hooks(block.attention.output.dense, f"projection_{i}")
    #     attach_hooks(block.intermediate.dense_act, f"upsample_{i}")
    #     attach_hooks(block.output.dense, f"downsample_{i}")

    checkpoint = None
    global_steps = 0
    args.resume_step = 0
    checkpoint_names = [f for f in os.listdir(args.model_output_dir)
                        if f.endswith(".pt")]
    if len(checkpoint_names) > 0:
        args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip())
                                for x in checkpoint_names])

        checkpoint = torch.load(
            os.path.join(args.model_output_dir, "ckpt_{}.pt".format(args.resume_step)),
            map_location="cpu"
        )

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.previous_phase_end_step > args.resume_step:
            raise ValueError('previous_phase_end_step={} cannot be larger '
                             'than resume_step={}'.format(
                args.previous_phase_end_step, args.resume_step))
        global_steps = args.resume_step - args.previous_phase_end_step

        logger.info('Resume from step {} checkpoint'.format(args.resume_step))

    model.to(args.device)
    model.checkpoint_activations(args.checkpoint_activations)

    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, checkpoint, global_steps, criterion, args, skip_layers


def set_no_decay(model, optimizer):
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    no_decay_param_group = {"params": [], "weight_decay": 0.0}
    for key, value in optimizer.param_groups[0].items():
        if key not in ["params", "weight_decay"]:
            no_decay_param_group[key] = value
    params = optimizer.param_groups[0]["params"]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            no_decay_param_group["params"].append(param)
            for i in range(len(params)):
                if param.shape == params[i].shape and torch.all(param == params[i]):
                    del params[i]
                    break
    optimizer.param_groups.append(no_decay_param_group)
    if hasattr(optimizer, 'optimizer'):
        optimizer.optimizer.param_groups.append(no_decay_param_group)
    return optimizer


def get_optimizer(grouped_parameters, lr, model, optimizer_name, weight_decay=0.01):
    if optimizer_name == 'eva':
        eva_backend.init("Torch")
        sgd_layers = [module for module in model.modules() if
                      isinstance(module, torch.nn.Linear) and module.out_features == 30528]
        # backend = comm.get_comm_backend()
        optimizer = eva.Eva(model, lr=lr, sgd_layers=sgd_layers,
                            optimizer=FusedLAMB(grouped_parameters, lr=lr),
                            )
        return optimizer
    elif optimizer_name == 'lamb':
        return LAMBOptimizer(grouped_parameters, lr=lr)
    elif optimizer_name == 'fused-lamb':
        return FusedLAMB(grouped_parameters, lr=lr)
    elif optimizer_name == 'mkor':
        sgd_layers = [module for module in model.modules() if
                      isinstance(module, torch.nn.Linear) and module.out_features == 30528]
        # backend = comm.get_comm_backend()
        optimizer = mkor.MKOROptimizer(model, lr=lr, weight_decay=weight_decay, inv_freq=10, measure_time=False,
                                       svd=False,
                                       #    backend=backend,
                                       sgd_layers=sgd_layers, grad_accum_steps=args.accumulation_steps,
                                       half_precision=True,
                                       optimizer=FusedLAMB(grouped_parameters, lr=lr),
                                       )
        return optimizer
    # elif optimizer_name == 'kaisa':
    #     optimizer = kaisa.KAISAOptimizer(model)
    #     return optimizer
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(grouped_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(grouped_parameters, lr=lr, weight_decay=weight_decay)


def prepare_optimizers(args, model, checkpoint, global_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    lora_group = ['lora_left', 'lora_right']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in lora_group)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in lora_group)],
         'weight_decay': args.lora_weight_decay},
    ]

    if args.lr_decay == 'poly':
        Scheduler = PolyWarmUpScheduler
    elif args.lr_decay == 'linear':
        Scheduler = LinearWarmUpScheduler
    elif args.lr_decay == 'cosine':
        Scheduler = CosineAnnealingLR
    else:
        raise ValueError('Unknown lr decay "{}"'.format(args.lr_decay))

    optimizer = get_optimizer(optimizer_grouped_parameters, args.learning_rate, model, args.optimizer,
                              args.weight_decay)

    if checkpoint is not None:
        try:
            if args.resume_step >= args.previous_phase_end_step:
                if args.optimizer != "mkor":
                    keys = list(checkpoint['optimizer']['state'].keys())
                    # Override hyperparameters from previous checkpoint
                    for key in keys:
                        checkpoint['optimizer']['state'][key]['step'] = global_steps
                    for i, item in enumerate(checkpoint['optimizer']['param_groups']):
                        checkpoint['optimizer']['param_groups'][i]['step'] = global_steps
                        checkpoint['optimizer']['param_groups'][i]['t_total'] = args.max_steps
                        checkpoint['optimizer']['param_groups'][i]['warmup'] = args.warmup_proportion
                        checkpoint['optimizer']['param_groups'][i]['lr'] = args.learning_rate
                else:
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['step'] = global_steps
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['t_total'] = args.max_steps
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['warmup'] = args.warmup_proportion
                    checkpoint['optimizer']['optimizer']['param_groups'][0]['lr'] = args.learning_rate
        except:
            pass
        try:
            if args.optimizer == "lamb" and "optimizer" in checkpoint["optimizer"]:
                print("Loaded LAMB Optimizer from MKOR Checkpoint")
                optimizer.load_state_dict(checkpoint['optimizer']['optimizer'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.grad_accum_steps = args.accumulation_steps
            if hasattr(optimizer, 'accumulated_steps'):
                for i in range(len(optimizer.accumulated_steps)):
                    optimizer.accumulated_steps[i] = 0
        except:
            optimizer = get_optimizer(optimizer_grouped_parameters, args.learning_rate, model, args.optimizer,
                                      args.weight_decay)
            for param_group in optimizer.param_groups:
                param_group['step'] = global_steps

    try:
        lr_schedulers = [Scheduler(optimizer, warmup=args.warmup_proportion,
                                   total_steps=args.max_steps)]
    except:
        lr_schedulers = [Scheduler(optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 1e-4)]
    scaler = None
    if args.fp16:
        scaler = GradScaler()
        if checkpoint is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if hasattr(optimizer, 'update_grad_scale'):
            optimizer.update_grad_scale(scaler.get_scale())

    preconditioner = None
    if args.kfac:
        preconditioner = kfac.KFAC(
            model,
            lr=args.learning_rate,
            factor_decay=args.kfac_stat_decay,
            damping=args.kfac_damping,
            kl_clip=args.kfac_kl_clip,
            factor_update_freq=args.kfac_factor_interval,
            inv_update_freq=args.kfac_inv_interval,
            # Skip TrainingHeads which contains the decoder, a Linear module
            # with shape (seq_len, vocab_size), such that it is too large to invert
            skip_layers=args.kfac_skip_layers,
            # BERT calls KFAC very infrequently so no need to optimize for
            # communication. Optimize for memory instead.
            comm_method=kfac.CommMethod.MEM_OPT,
            # Compute the factors and update the running averages during the
            # forward backward pass b/c we are using grad accumulation but
            # not accumulating the input/output data
            accumulate_data=False,
            compute_factor_in_hook=True,
            distribute_layer_factors=False,
            # grad_scaler=scaler,
        )

        lrs = Scheduler(preconditioner, warmup=args.warmup_proportion,
                        total_steps=args.max_steps)
        lr_schedulers.append(lrs)

        if checkpoint is not None and 'preconditioner' in checkpoint:
            preconditioner.load_state_dict(checkpoint['preconditioner'])

        if is_main_process():
            logger.info(preconditioner)

    return optimizer, preconditioner, lr_schedulers, scaler


def prepare_dataset(args, checkpoint):
    input_files = []
    if os.path.isfile(args.input_dir):
        input_files.append(args.input_dir)
    elif os.path.isdir(args.input_dir):
        for path in Path(args.input_dir).rglob('*.hdf5'):
            if path.is_file():
                input_files.append(str(path))

    with open(args.model_config_file) as f:
        configs = json.load(f)
        vocab_size = configs['vocab_size']
        vocab_file = configs['vocab_file']
        lowercase = configs['lowercase']
        tokenizer = configs['tokenizer']
    mask_token_id = None
    if tokenizer == 'wordpiece':
        tokenizer = get_wordpiece_tokenizer(vocab_file, uppercase=not lowercase)
    elif tokenizer == 'bpe':
        tokenizer = get_bpe_tokenizer(vocab_file, uppercase=not lowercase)
    else:
        raise ValueError('Unknown tokenizer \'{}\'. Options are '
                         '\'wordpiece\' and \'bpe\''.format(tokenizer))
    mask_token_id = tokenizer.token_to_id('[MASK]')

    dataset = ShardedPretrainingDataset(input_files, mask_token_id,
                                        args.max_predictions_per_seq, args.masked_token_fraction,
                                        vocab_size=vocab_size)
    sampler = DistributedSampler(dataset, get_world_size(), rank=get_rank())

    if checkpoint is not None and 'sampler' in checkpoint:
        sampler.load_state_dict(checkpoint['sampler'])

    loader = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                         batch_size=args.local_batch_size, num_workers=4, pin_memory=True)

    if is_main_process():
        logger.info('Samples in dataset: {}'.format(len(dataset)))
        logger.info('Samples per device: {}'.format(len(sampler)))
        logger.info('Sampler starting index: {}'.format(sampler.index))
        logger.info('Batches in dataloader: {}'.format(len(loader)))
    return loader, sampler


def take_optimizer_step(optimizer, preconditioner, model, scaler):
    if preconditioner is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        preconditioner.step()
    # for module in model_info.keys():
    #     model_info[module]["grad_norm"] += torch.norm(module.weight.grad).item() / module.weight.grad.numel()
    #     model_info[module]["first_order_effect"] += -torch.mean(module.weight.grad * module.weight).item()
    #     # model_info[module]["second_order_only_effect"] += torch.mean(((model_info[module]["grad_cov"] /
    #     #                                                                model_info[module]["count"]) @ module.weight @ (
    #     #                                                                           model_info[module]["activation_cov"] /
    #     #                                                                           model_info[module][
    #     #                                                                               "count"])) * module.weight).item()
    #     # model_info[module]["second_order_effect"] = model_info[module]["first_order_effect"] + model_info[module][
    #     #     "second_order_only_effect"]
    #     model_info[module]["weight_norm"] += torch.norm(module.weight).item() / module.weight.numel()
    #     model_info[module]["wanda_score_sum"] = (module.weight.abs() * torch.sqrt(model_info[module]["activation_norm"])).sum().item()
    #     model_info[module]["wanda_score_mean"] = model_info[module]["wanda_score_sum"] / module.weight.numel()
    #     # del model_info[module]["grad_cov"]
    #     # del model_info[module]["activation_cov"]
    #     del model_info[module]["activation_norm"]
    # if is_main_process():
    #     title = False
    #     with open("model_info.csv", "w") as f:
    #         for module in model_info.keys():
    #             info_labels = model_info[module].keys()
    #             if not title:
    #                 for label in info_labels:
    #                     f.write(label + ",")
    #                 title = True
    #                 f.write("\n")
    #             for info in info_labels:
    #                 f.write(str(model_info[module][info]) + ",")
    #             f.write("\n")
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        if hasattr(optimizer, "update_grad_scale"):
            optimizer.update_grad_scale(scaler.get_scale())
    else:
        optimizer.step()
    # print("Memory Usage: ", torch.cuda.max_memory_allocated(device=model.device) / (2.0 ** 30))

    try:
        optimizer.zero_grad()
        if hasattr(optimizer, "optimizer"):
            optimizer.optimizer.zero_grad()
    except:
        for param in model.parameters():
            param.grad = None


def forward_backward_pass(model, criterion, scaler, batch, divisor,
                          sync_grads=True):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    if scaler is not None:
        with autocast():
            # if is_main_process():
            #     if torch.any(torch.isnan(input_ids)):
            #         print('input_ids is nan')
            #     if torch.any(torch.isnan(segment_ids)):
            #         print('segment_ids is nan')
            #     if torch.any(torch.isnan(input_mask)):
            #         print('input_mask is nan')
            prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask)
            # if is_main_process():
            #     if torch.isnan(prediction_scores).any():
            #         print('prediction_scores is nan')
            #     if torch.any(torch.isnan(seq_relationship_score)):
            #         print('seq_relationship_score is nan')
            #     for module in model.modules():
            #         if isinstance(module, torch.nn.Linear):
            #             if torch.any(torch.isnan(module.weight)):
            #                 print('weight is nan')
            #             if module.bias is not None:
            #                 if torch.any(torch.isnan(module.bias)):
            #                     print('bias is nan')
            #                     break
            loss = criterion(
                prediction_scores,
                masked_lm_labels,
                seq_relationship_score,
                next_sentence_labels)
    else:
        prediction_scores, seq_relationship_score = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask)
        loss = criterion(
            prediction_scores,
            masked_lm_labels,
            seq_relationship_score,
            next_sentence_labels)

    loss = loss / divisor

    if not sync_grads:
        with model.no_sync():
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    else:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    return loss


def main(args):
    global timeout_sent
    model, checkpoint, global_steps, criterion, args, skip_layers = prepare_model(args)
    optimizer, preconditioner, lr_schedulers, scaler = prepare_optimizers(
        args, model, checkpoint, global_steps)
    dataloader, datasampler = prepare_dataset(args, checkpoint)
#    if args.add_lora:
    quantize_model(model, skip_layers, args.quantization_en, args.qbitwidth, args.add_lora, args.accelerate)
    model.train()
    most_recent_ckpts_paths = []
    average_loss = None
    current_loss = 0.
    epoch = 0
    training_steps = 0
    train_time_start = time.time()
    ignore_time = True

    if checkpoint is not None and 'epoch' in checkpoint:
        epoch = checkpoint['epoch']

    # Note: We loop infinitely over epochs, termination is handled
    #       via iteration count
    while True:
        if not args.disable_progress_bar:
            train_iter = tqdm(dataloader, disable=not is_main_process())
        else:
            train_iter = dataloader

        for batch in train_iter:
            sync_grads = False
            if args.accumulation_steps == 1 or (training_steps > 0
                                                and training_steps % args.accumulation_steps == args.accumulation_steps - 1):
                sync_grads = True

            batch = [t.to(args.device) for t in batch]
            loss = forward_backward_pass(model, criterion, scaler, batch,
                                         args.accumulation_steps, sync_grads=sync_grads)
            if ignore_time:
                modeling.timer.zero_time()
                ignore_time = False
            else:
                if is_main_process() and modeling.timer.measure:
                    modeling.timer.save("results", "BERT Forward Pass Time")

            current_loss += loss.item()

            if sync_grads:
                for lrs in lr_schedulers:
                    lrs.step()
                take_optimizer_step(optimizer, preconditioner,
                                    model, scaler)
                global_steps += 1
                if average_loss is None:
                    average_loss = current_loss
                else:
                    average_loss = 0.95 * average_loss + 0.05 * current_loss
                logger.log(tag='train',
                           step=global_steps + args.previous_phase_end_step,
                           epoch=epoch,
                           average_loss=average_loss,
                           step_loss=current_loss,
                           learning_rate=optimizer.param_groups[0]['lr'])

                # exit()
                # wandb.log({"train/step_loss": current_loss, 
                #            "train/average_loss": average_loss, 
                #            "train/learning_rate": optimizer.param_groups[0]['lr'],
                #            "epoch": epoch,
                #            "step": global_steps + args.previous_phase_end_step,
                #            })
                current_loss = 0.

            if (global_steps >= args.max_steps or timeout_sent or
                    (training_steps > 0 and training_steps % (
                            args.num_steps_per_checkpoint * args.accumulation_steps
                    ) == 0)):
                if global_steps >= args.max_steps or timeout_sent:
                    final_time = time.time() - train_time_start
                if is_main_process() and not args.skip_checkpoint:
                    # Save a trained model
                    logger.info('Saving checkpoint: global_steps={}'.format(
                        global_steps + args.previous_phase_end_step))
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_save_file = os.path.join(
                        args.model_output_dir,
                        "ckpt_{}.pt".format(global_steps + args.previous_phase_end_step))
                    model_dict = {
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'sampler': datasampler.state_dict(),
                        'epoch': epoch,
                    }
                    if preconditioner is not None:
                        model_dict['preconditioner'] = preconditioner.state_dict()
                    if scaler is not None:
                        model_dict['scaler'] = scaler.state_dict()
                    torch.save(model_dict, output_save_file)

                    most_recent_ckpts_paths.append(output_save_file)
                    # if len(most_recent_ckpts_paths) > 3:
                    #     ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                    #     os.remove(ckpt_to_be_removed)

            # Exiting the training due to hitting max steps, or being sent a
            # timeout from the cluster scheduler
            if global_steps >= args.max_steps or timeout_sent:
                return global_steps, final_time

            training_steps += 1

        epoch += 1


if __name__ == "__main__":
    args = parse_arguments()

    if args.input_dir is None:
        raise ValueError('--input_dir must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')
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
    global_steps, train_time = main(args)
    runtime = time.time() - start_time

    logger.info("runtime: {}  train_time: {}  training_seq_per_sec: {}".format(
        runtime, train_time,
        args.global_batch_size * global_steps / train_time))

