#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --nodes 1
#SBATCH --mem=8G
#SBATCH -t 7:00:00
#SBATCH --account=def-mmehride

PRUNE_INPUTS="--prune_inputs True" #Comment to disable pruning

cd ../

CHECKPOINT_FOLDER=lamb

# Copy Singularity
mkdir ${SLURM_TMPDIR}/torch.sif && tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch.tar -C $SLURM_TMPDIR;



MODEL_CHECKPOINT=${1:-"results/$CHECKPOINT_FOLDER/ckpt_8601.pt"}
if [ -z $PRUNE_INPUTS ]; then
  OUTPUT_DIR=${2:-"results/$CHECKPOINT_FOLDER-squad"}
else
  OUTPUT_DIR=${2:-"results/$CHECKPOINT_FOLDER-squad-pruned-inputs-accurate"}
fi
CONFIG_FILE=${3:-"config/bert_large_uncased_config.json"}

mkdir ${SLURM_TMPDIR}/squad && tar -xf squad.tar -C ${SLURM_TMPDIR}
DATA_DIR="/tmp"
SQUAD_DIR="$DATA_DIR/squad/v1.1"

BERT_MODEL="bert-large-uncased"

NGPUS=1
BATCH_SIZE=4

export OMP_NUM_THREADS=8

LOGFILE="$OUTPUT_DIR/squad_log.txt"

echo "Output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR
if [ ! -d "$OUTPUT_DIR" ]; then
	echo "ERROR: unable to make $OUTPUT_DIR"
fi

CMD="python -m torch.distributed.launch --nproc_per_node=$NGPUS run_squad.py"

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
CMD+=" --learning_rate=3e-5 "
CMD+=" --num_train_epochs=2 "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUTPUT_DIR "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --fp16 "
CMD+="$PRUNE_INPUTS "

echo "$CMD | tee $LOGFILE"
time singularity exec --bind $PWD:/home/mozaffar --bind $SLURM_TMPDIR:/tmp --nv ${SLURM_TMPDIR}/torch.sif $CMD | tee $LOGFILE
