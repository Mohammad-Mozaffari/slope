#!/bin/bash
#SBATCH --gpus-per-node=v100l:4
#SBATCH --nodes 4
#SBATCH --mem=36G
#SBATCH --ntasks-per-node=4
#SBATCH -t 72:00:00
#SBATCH --account=rrg-mmehride


cd ../


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


Distributed system configuration
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


COPY_SINGULARITY="mkdir ${SLURM_TMPDIR}/torch-gpt.sif && 
                    tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch-gpt.tar -C $SLURM_TMPDIR &&
                    mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki &&
                    mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls &&
                    mkdir ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls/certs &&
                    cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-gpt.sif/etc/pki/tls/certs/ca-bundle.crt;"

COPY_DATASET="mkdir $SLURM_TMPDIR/data && cp -r data/* $SLURM_TMPDIR/data/;"


SINGULARITY="singularity exec 
                --bind $PWD:/home/mozaffar 
                --bind $SLURM_TMPDIR:/tmp 
                --nv ${SLURM_TMPDIR}/torch-gpt.sif "
POST_SINGULARITY_LOAD="export WANDB_MODE=offline; 
                        export OMP_NUM_THREADS=8; 
                        HYDRA_FULL_ERROR=1; 
                        export WANDB_API_KEY=755c093d24b1dcff6500f39cc878c3e6b6da5653; 
                        export DATA_DIR=/tmp/data; 
                        export HF_DATASETS_CACHE=/tmp/data; 
                        cd $CURRENT_DIR; 
                        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/training/pruning/spmm/libcusparse_lt/lib; "
LAUNCHER="python 
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

CURRENT_DIR=$(pwd)

RANK=0
for NODE in $RANKS; do
    PRE_SINGULARITY_LOAD="module load apptainer; export LOCA_RANK=$RANK; export NODE_RANK=$RANK; export MASTER_ADDR=$MAIN_RANK; export MASTER_PORT=4321; "
    FULL_CMD="$COPY_SINGULARITY $COPY_DATASET $PRE_SINGULARITY_LOAD $SINGULARITY $LAUNCHER"
    FULL_CMD="${FULL_CMD//$'\n'/}" #Remove \n from the command
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


sleep 1000000000000