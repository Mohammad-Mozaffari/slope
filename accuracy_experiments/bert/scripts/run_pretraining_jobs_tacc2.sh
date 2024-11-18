#!/bin/bash
#SBATCH -J mkor-right2           # Job name
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes 
#SBATCH -n 2              # Total # of mpi tasks
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)

cd ../

OPTIMIZER=lamb
# PRUNE_INPUTS="--prune_inputs True" #Comment to disable pruning
PHASE=1
SINGULARITY=True

# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=hostfile
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
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_${NNODES}nodes"
else
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_prune_inputs_${NNODES}nodes"
fi


if [ "$SINGULARITY" = True ]; then
    COPY_SINGULARITY="mkdir /tmp/torch.sif; tar -xf /scratch/09070/tg883700/torch.tar -C /tmp;"
fi
COPY_DATASET="mkdir /tmp/data; tar -xf /scratch/09070/tg883700/datasets/bert/pretraining/phase${PHASE}.tar -C /tmp/data;"


if [ "$SINGULARITY" = True ]; then
    LOAD="module load tacc-apptainer; export OMP_NUM_THREADS=8;"
    SINGULARITY="singularity exec --bind $PWD:/home --bind /tmp:/tmp --nv /tmp/torch.sif  "
else
    SINGULARITY=""
    LOAD="export OMP_NUM_THREADS=8; source source /scratch/09070/tg883700/pytorch/bin/activate;"
fi


CURRENT_DIR=$(pwd)

RANK=0
for NODE in $RANKS; do
    LAUNCHER="python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=$RANK --master_addr=$MAIN_RANK --master_port=4321\
        run_pretraining.py \
        --input_dir /tmp/data/phase${PHASE} \
        --output_dir ${OUTPUT_DIR} \
        --config_file config/bert_pretraining_phase${PHASE}_config.json \
        --weight_decay 0.01 \
        --num_steps_per_checkpoint 100 \
        --optimizer ${OPTIMIZER} \
        ${PRUNE_INPUTS} 
        "
    FULL_CMD="$LOAD $COPY_SINGULARITY $COPY_DATASET $SINGULARITY $LAUNCHER"
    if [[ $NODE == $MAIN_RANK ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	    ssh $NODE "cd $CURRENT_DIR; $FULL_CMD" &
    fi
    RANK=$((RANK + 1))
done


wait
sleep 172800
