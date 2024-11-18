#!/bin/bash
#SBATCH -J gpt2s-4          # Job name
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node

cd ../

#Parameters to change in each model
# 1- Optimizer
# 2- Batch Size
# 3- Model


OPTIMIZER=mkor
PRUNED_MATRIX="none"
SKIP_FIRST_BLOCK=False
SKIP_LAST_BLOCK=False
SKIP_ATTENTION=False
REDUCTION_DIM=True
BATCH_SIZE=8 #GPT2-S:16
ADD_LORA=False
LORA_RANK=4
MODEL=gpt2s
RESUME_ITER=-1
MAX_STEPS=400000

NAME=$MODEL-$OPTIMIZER



if [[ $PRUNED_MATRIX != "none" ]]; then
    if [[ $REDUCTION_DIM == True ]]; then
        NAME=$NAME-reduction_dim
    fi
    NAME=$NAME-pruned-$PRUNED_MATRIX
    # If all flags are True, add mlp_only to the name
    if [[ $SKIP_FIRST_BLOCK == True ]] && [[ $SKIP_LAST_BLOCK == True ]] && [[ $SKIP_ATTENTION == True ]]; then
        NAME=$NAME-mlp_only
    else
        if [[ $SKIP_FIRST_BLOCK == True ]]; then
            NAME=$NAME-skip_first_block
        fi
        if [[ $SKIP_LAST_BLOCK == True ]]; then
            NAME=$NAME-skip_last_block
        fi
        if [[ $SKIP_ATTENTION == True ]]; then
            NAME=$NAME-skip_attention
        fi
    fi
    
    if [[ $ADD_LORA == True ]]; then
        if [[ $RESUME_ITER != "-1" ]]; then
            NAME=$NAME-late-
        fi
        NAME=$NAME-add_lora$LORA_RANK
    fi
fi

NAME=$NAME-${MAX_STEPS}steps


# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    MAIN_RANK=$HOSTNAME
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


# NAME=$NAME-$NNODES-nodes

CURRENT_DIR=$(pwd)

LOAD="module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja; 
        export WANDB_MODE=offline; 
        export OMP_NUM_THREADS=8; 
        source activate pytorch; 
        HYDRA_FULL_ERROR=1; 
        export WANDB_API_KEY=755c093d24b1dcff6500f39cc878c3e6b6da5653;
        export DATA_DIR=$PWD/data; 
        export HF_DATASETS_CACHE=$PWD/data; 
        cd $CURRENT_DIR; "


LAUNCHER="srun python 
            training/run.py 
            experiment=owt/$MODEL 
            trainer.devices=4 
            datamodule.batch_size=$BATCH_SIZE 
            trainer.num_nodes=$NNODES 
            name=$NAME 
            resume=True 
            logger=csv 
            optimizer=$OPTIMIZER 
            pruner.pruned_matrix=$PRUNED_MATRIX 
            pruner.skip_attention=$SKIP_ATTENTION 
            pruner.skip_first_block=$SKIP_FIRST_BLOCK 
            pruner.skip_last_block=$SKIP_LAST_BLOCK 
            pruner.reduction_dim=$REDUCTION_DIM 
            pruner.add_lora=$ADD_LORA 
            pruner.lora_rank=$LORA_RANK 
            trainer.max_steps=$MAX_STEPS"

FULL_CMD="$LOAD $LAUNCHER"
export HYDRA_FULL_ERROR=1
echo $FULL_CMD
eval $FULL_CMD &

wait