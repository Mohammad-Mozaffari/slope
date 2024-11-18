#!/bin/bash
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -p gpu-a100

NAME="mkor-gpt2s"
NGPU_PER_NODE=3

cd ../

CURRENT_DIR=$(pwd)
LOADS="module load tacc-apptainer; export OMP_NUM_THREADS=8; HYDRA_FULL_ERROR=1; export WANDB_API_KEY=755c093d24b1dcff6500f39cc878c3e6b6da5653; cd $CURRENT_DIR;"
COPY_SINGULARITY="mkdir /tmp/torch.sif && tar -xf /scratch/09070/tg883700/torch.tar -C /tmp;"
SINGULARITY_PREFIX="singularity exec --bind $PWD:$HOME --bind /tmp:/tmp --nv /tmp/torch.sif "




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


RUN_CMD="python training/run.py experiment=owt/gpt2s trainer.devices=$NGPU_PER_NODE datamodule.batch_size=16 trainer.num_nodes=$NNODES name=$NAME resume=True"


RANK=0
for NODE in $RANKS; do
    NODE_INFO="export NODE_RANK=$RANK; export MASTER_ADDR=$MAIN_RANK; export MASTER_PORT=1234;"
    FULL_CMD="$LOADS $COPY_SINGULARITY $NODE_INFO $SINGULARITY_PREFIX $RUN_CMD"
    if [[ $NODE == $MAIN_RANK ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	    ssh $NODE "bash -lc '$FULL_CMD'" &
    fi
    RANK=$((RANK + 1))
done


wait
