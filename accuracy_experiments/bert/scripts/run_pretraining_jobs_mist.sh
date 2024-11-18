#!/bin/bash
#SBATCH -J bert1-sparse           # Job name
#SBATCH -N 25
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node

cd ../

OPTIMIZER=fused-lamb
PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=False
SKIP_LAST_BLOCK=False
SKIP_ATTENTION=False
ADD_LORA=False
LORA_RANK=40
UNSTRUCTURED_MASKING=True
UNSTRUCTURED_MASK_SPARSITY=90
SPARSITY_INCREMENT=-1 #Set to -1 to disable sparsity increment
PHASE=1

NAME=phase${PHASE}-$OPTIMIZER

# SINGULARITY=True
REDUCTION_DIM=True


if [[ $PRUNED_MATRIX != "none" ]]; then
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
        NAME=$NAME-add_lora${LORA_RANK}
    fi
    if [[ $SPARSITY_INCREMENT != -1 ]]; then
        NAME=$NAME-sparsity_increment${SPARSITY_INCREMENT}
    fi
    if [[ $UNSTRUCTURED_MASKING == True ]]; then
        NAME=$NAME-unstructured_masking${UNSTRUCTURED_MASK_SPARSITY}
    fi
fi
if [[ $PHASE == 3 ]]; then
        INPUT_PHASE=2
else
        INPUT_PHASE=$PHASE
fi


module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
source activate pytorch


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

if [[ -z "$PRUNE_INPUTS" ]]; then
    OUTPUT_DIR="results/${NAME}-${NNODES}nodes"
else
    OUTPUT_DIR="results/${NAME}-${NNODES}nodes"
fi

CURRENT_DIR=$(pwd)

LOAD="module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja; export OMP_NUM_THREADS=8; source activate pytorch; cd $CURRENT_DIR; "
# SINGULARITY="singularity exec --nv --writable --bind $PWD:/home/mozaffar /scratch/mozaffar/torch.sif"


RANK=0
for NODE in $RANKS; do
    LAUNCHER="python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$NNODES --node_rank=$RANK --master_addr=$MAIN_RANK --master_port=4321\
        run_pretraining.py \
        --input_dir ./phase${INPUT_PHASE} \
        --output_dir ${OUTPUT_DIR} \
        --config_file config/bert_pretraining_phase${PHASE}_config.json \
        --weight_decay 0.01 \
        --num_steps_per_checkpoint 100 \
        --optimizer ${OPTIMIZER} \
        --pruned_matrix ${PRUNED_MATRIX} \
        --skip_first_block ${SKIP_FIRST_BLOCK} \
        --skip_last_block ${SKIP_LAST_BLOCK} \
        --skip_attention ${SKIP_ATTENTION} \
        --reduction_dim ${REDUCTION_DIM} \
        --add_lora ${ADD_LORA} \
        --lora_rank ${LORA_RANK} \
        --unstructured_masking ${UNSTRUCTURED_MASKING} \
        --unstructured_mask_sparsity ${UNSTRUCTURED_MASK_SPARSITY} \
        --sparsity_increment ${SPARSITY_INCREMENT}
        "
    FULL_CMD="$SINGULARITY $LAUNCHER"
    if [[ $NODE == $MAIN_RANK ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	      ssh $NODE "cd $CURRENT_DIR; $LOAD $FULL_CMD" &
    fi
    RANK=$((RANK + 1))
done


wait