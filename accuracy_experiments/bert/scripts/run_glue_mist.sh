#!/bin/bash
#SBATCH -J glue1-lora4
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH -t 23:30:00


# task_name is in [cola, sst-2, mrpc, sts-b, qqp, mnli, mnli-mm, qnli, rte, wnli]
# [CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI]
# Hyperparameters used to obtain similar GLUE results as the MKOR paper:
# batch size = 4, gradient accumulation steps = 1, learning rate = 2.4e-6, warmup proportion = 0.1, epochs = 10, max steps = -1, precision = fp16, max seq length = 128

cd ../


module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
source activate pytorch

PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=False
SKIP_LAST_BLOCK=False
SKIP_ATTENTION=False
ADD_LORA=True
LORA_RANK=64
REDUCTION_DIM=False
SPARSITY_INCREMENT=-1 #Set to -1 to disable sparsity increment
UNSTRUCTURED_MASKING=True
UNSTRUCTURED_MASK_SPARSITY=90

CHECKPOINT_DIR=phase2-fused-lamb-pruned-static-weight-add_lora64-late8438-8nodes
CHECKPOINT_ITER=8601

# CoLA SST-2 MRPC STS-B QQP RTE
# MNLI QNLI 
for glue_task_name in CoLA SST-2 MRPC STS-B QQP RTE
do
  init_checkpoint="results/$CHECKPOINT_DIR/pretrain_ckpts/ckpt_$CHECKPOINT_ITER.pt"
  num_gpu=1 # # of gpus per node
  batch_size=4
  out_dir="results/$CHECKPOINT_DIR-glue/${glue_task_name}"
  data_dir="glue/${glue_task_name}"
  vocab_file="config/vocab.txt"
  config_file="config/bert_large_uncased_config.json"

  task_name="${glue_task_name,,}"
  gradient_accumulation_steps=${2:-"1"}
  learning_rate=${3:-"2.4e-6"} # Pls note that some lr can lead to CoLA mcc goes to 0, original lr was 2.4e-5
  warmup_proportion=${4:-"0.1"}
  if [ "$glue_task_name" = "CoLA" ] ; then
    epochs=5
  else
    epochs=3
  fi
  # epochs=${5:-"3"} # For MRPC, change it to 5.
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
        --lora_rank ${LORA_RANK}  \
        --unstructured_masking ${UNSTRUCTURED_MASKING} \
        --unstructured_mask_sparsity ${UNSTRUCTURED_MASK_SPARSITY} \
        --sparsity_increment ${SPARSITY_INCREMENT} "

  LOGFILE=$out_dir/$task_name.log

  echo "$CMD | tee $LOGFILE"
  echo "running"
  $CMD | tee $LOGFILE
done