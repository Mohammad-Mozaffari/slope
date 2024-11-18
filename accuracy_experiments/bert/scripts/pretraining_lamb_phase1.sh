#!/bin/bash
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p gpu-a100
#SBATCH -n 1

# Change to 2 for Phase 2 training
PHASE=2

if [[ "$PHASE" -eq 1 ]]; then
        CONFIG=config/bert_pretraining_phase1_config.json
        DATA=/scratch/00946/zzhang/data/bert/bert_masked_wikicorpus_en/phase1
else
        CONFIG=config/bert_kfac_pretraining_phase2_config.json
        DATA=/scratch/00946/zzhang/data/bert/bert_masked_wikicorpus_en/phase2
fi

mkdir -p sbatch_logs

HOSTFILE=hostfile
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=3

# PHASE 1
# mpirun -hostfile $HOSTFILE -np $NODES -ppn 1  bash scripts/launch_pretraining.sh  \
#     --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
#     --kwargs \
#     --input_dir $DATA \
#     --output_dir results/bert_pretraining_6e-3-wd0.01-wu2843-inv50-4kbatchsize \
#     --config_file $CONFIG \
#     --weight_decay 0.01 \
#     --num_steps_per_checkpoint 200

#    --lr_decay cosine \
PHASE 2
mpirun -np $NODES -hostfile $HOSTFILE -ppn 1 bash scripts/launch_pretraining.sh  \
   --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
   --config_file $CONFIG \
   --input_dir $DATA \
   --output_dir results/phase2_6e-3-wd0.01-wu2843-inv50 \
   --weight_decay 0.01 \
   --num_steps_per_checkpoint 200
