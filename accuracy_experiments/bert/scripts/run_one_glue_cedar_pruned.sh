#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --nodes 1
#SBATCH --mem=20G
#SBATCH -t 18:00:00
#SBATCH --account=def-mmehride
#SBATCH --output=/home/islahahm/projects/def-mmehride/islahahm/bert-mkor/scripts/slurmoutput/%j.out

# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
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


# Copy Singularity
mkdir ${SLURM_TMPDIR}/torch-bert.sif && tar -xf /home/islahahm/projects/def-mmehride/islahahm/torch-bert.tar -C $SLURM_TMPDIR

set -e

# echo "Container nvidia build = " $NVIDIA_BUILD_ID
# task_name is in [cola, sst-2, mrpc, sts-b, qqp, mnli, mnli-mm, qnli, rte, wnli]
# Hyperparameters used to obtain similar GLUE results as the MKOR paper:
# batch size = 4, gradient accumulation steps = 1, learning rate = 2.4e-6, warmup proportion = 0.1, epochs = 10, max steps = -1, precision = fp16, max seq length = 128

cd ../

PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=True
SKIP_LAST_BLOCK=True
SKIP_ATTENTION=True
ADD_LORA=True
LORA_RANK=40
REDUCTION_DIM=True


glue_task_name=${1:-"CoLA"}
init_checkpoint="results/checkpoints/ckpt_8601.pt"
num_gpu=1 # # of gpus per node
batch_size=4
out_dir="results/checkpoints-glue/${glue_task_name}"
data_dir="glue/${glue_task_name}"
vocab_file="config/vocab.txt"
config_file="config/bert_large_uncased_config.json"

task_name="${glue_task_name,,}"
gradient_accumulation_steps=${2:-"1"}
learning_rate=${3:-"2.4e-6"} # Pls note that some lr can lead to CoLA mcc goes to 0, original lr was 2.4e-5
warmup_proportion=${4:-"0.1"}
epochs=${5:-"3"} # For MRPC, change it to 5.
max_steps=${6:-"-1.0"}
precision=${7:-"fp16"}
seed=${8:-"0"}
mode=${9:-"train eval"} # "train eval prediction" to add prediction

mkdir -p $out_dir

if [ "$mode" = "eval" ] ; then
  num_gpu=1
  num_nodes=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  # mpi_command=" -m torch.distributed.launch "
  # mpi_command+="--nproc_per_node=$num_gpu  "
  # mpi_command+="--nnodes=$num_nodes "
  # mpi_command+="--node_rank=$node_rank "
  # mpi_command+="--master_addr=129.114.44.125 "
  # mpi_command+="--master_port=12345 "
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu --standalone --master_port=12345 "
fi

#CMD="python3 $mpi_command ../run_glue_nv.py "
CMD="python3 $mpi_command run_glue_nv.py "

CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="$use_fp16 "
CMD+="--pruned_matrix ${PRUNED_MATRIX} \
      --skip_first_block ${SKIP_FIRST_BLOCK} \
      --skip_last_block ${SKIP_LAST_BLOCK} \
      --skip_attention ${SKIP_ATTENTION} \
      --reduction_dim ${REDUCTION_DIM} \
      --add_lora ${ADD_LORA} \
      --lora_rank ${LORA_RANK}  "

LOGFILE=$out_dir/$task_name.log

echo "$CMD | tee $LOGFILE"
singularity exec --bind $PWD:/home/islahahm --bind $SLURM_TMPDIR:/tmp --nv ${SLURM_TMPDIR}/torch-bert.sif $CMD | tee $LOGFILE

