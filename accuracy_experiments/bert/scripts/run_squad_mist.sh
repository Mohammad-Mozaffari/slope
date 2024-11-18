#!/bin/bash
#SBATCH -J sq-lora40           # Job name
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --gpus-per-node=1

# PRUNE_INPUTS="--prune_inputs True" #Comment to disable pruning

cd ../


module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
source activate pytorch

# CHECKPOINT_FOLDER=phase2-fused-lamb-pruned-static-weight-mlp_only-8nodes
CHECKPOINT_FOLDER=phase2-fused-lamb-pruned-static-weight-skip_first_block-skip_last_block-add_lora40-sparsity_increment12-8nodes
CHECKPOINT_NUMBER=8601

PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=True
SKIP_LAST_BLOCK=True
SKIP_ATTENTION=False
ADD_LORA=True
LORA_RANK=40
SPARSITY_INCREMENT=12 #Set to -1 to disable sparsity increment
UNSTRUCTURED_MASKING=True
UNSTRUCTURED_MASK_SPARSITY=90
# Copy Singularity

MODEL_CHECKPOINT=${1:-"results/$CHECKPOINT_FOLDER/pretrain_ckpts/ckpt_$CHECKPOINT_NUMBER.pt"}
if [ $PRUNED_MATRIX == "none" ]; then
  OUTPUT_DIR=${2:-"results/$CHECKPOINT_FOLDER-squad-$CHECKPOINT_NUMBER-dense"}
else
  OUTPUT_DIR=${2:-"results/$CHECKPOINT_FOLDER-squad-$CHECKPOINT_NUMBER-pruned"}
fi
CONFIG_FILE=${3:-"config/bert_large_uncased_config.json"}

LR=3e-5
OUTPUT_DIR=$OUTPUT_DIR-lr$LR

REDUCTION_DIM=True

DATA_DIR="./"
SQUAD_DIR="$DATA_DIR/squad/v1.1"

BERT_MODEL="bert-large-uncased"

NGPUS=4
BATCH_SIZE=6

export OMP_NUM_THREADS=8

LOGFILE="$OUTPUT_DIR/squad_log.txt"

echo "Output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
if [ ! -d "$OUTPUT_DIR" ]; then
	echo "ERROR: unable to make $OUTPUT_DIR"
fi

if [[ $NGPUS > 1 ]]
then
  CMD="python -m torch.distributed.launch --nproc_per_node=$NGPUS run_squad.py"
else
  CMD="python run_squad.py"
fi

CMD+=" --init_checkpoint=$MODEL_CHECKPOINT "

CMD+=" --do_train "
CMD+=" --train_file=$SQUAD_DIR/train-v1.1.json "
CMD+=" --train_batch_size=$BATCH_SIZE "

CMD+=" --do_predict "
CMD+=" --predict_file=$SQUAD_DIR/dev-v1.1.json "
CMD+=" --predict_batch_size=$BATCH_SIZE "
CMD+=" --eval_script=$SQUAD_DIR/evaluate-v1.1.py "
CMD+=" --do_eval "

CMD+=" --do_lower_case "
CMD+=" --bert_model=$BERT_MODEL "
CMD+=" --learning_rate=$LR "
CMD+=" --num_train_epochs=2 "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUTPUT_DIR "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --fp16 "
CMD+="--pruned_matrix ${PRUNED_MATRIX} \
      --skip_first_block ${SKIP_FIRST_BLOCK} \
      --skip_last_block ${SKIP_LAST_BLOCK} \
      --skip_attention ${SKIP_ATTENTION} \
      --reduction_dim ${REDUCTION_DIM} \
      --add_lora ${ADD_LORA} \
      --lora_rank ${LORA_RANK} \
      --unstructured_masking ${UNSTRUCTURED_MASKING} \
      --unstructured_mask_sparsity ${UNSTRUCTURED_MASK_SPARSITY} \
      --sparsity_increment ${SPARSITY_INCREMENT}  "

echo "$CMD | tee $LOGFILE"
$CMD | tee $LOGFILE

# sleep 172800