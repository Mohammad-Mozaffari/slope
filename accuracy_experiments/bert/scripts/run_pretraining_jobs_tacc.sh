#!/bin/bash
#SBATCH -J bert2-24           # Job name
#SBATCH --nodes=8             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=1
#SBATCH -t 8:00:00
#SBATCH -p gpu-a100

cd ../

OPTIMIZER=fused-lamb
PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=False
SKIP_LAST_BLOCK=False
SKIP_ATTENTION=False
ADD_LORA=True
LORA_RANK=4
SPARSITY_INCREMENT=-1 #Set to -1 to disable sparsity increment
PHASE=2

SINGULARITY=True
REDUCTION_DIM=True

for LORA_RANK in 64
do
    for CHECKPOINT_ITER in 8538
    do
        NAME=phase${PHASE}-$OPTIMIZER
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
            if [[ -z $CHECKPOINT_ITER ]]; then
                echo No Late LoRA
            else
                NAME=$NAME-late$CHECKPOINT_ITER
            fi
        fi

        scontrol show hostname `echo $SLURM_JOB_NODELIST` > hostfile

        mkdir -p sbatch_logs

        HOSTFILE=hostfile
        cat $HOSTFILE

        MASTER_RANK=$(head -n 1 $HOSTFILE)
        NNODES=$(< $HOSTFILE wc -l)
        PROC_PER_NODE=3


        if [[ -z "$PRUNE_INPUTS" ]]; then
            OUTPUT_DIR="results/${NAME}-${NNODES}nodes"
        else
            OUTPUT_DIR="results/${NAME}-${NNODES}nodes"
        fi

        mkdir $OUTPUT_DIR
        mkdir $OUTPUT_DIR/pretrain_ckpts
        cp results/phase2-fused-lamb-pruned-static-weight-sparsity_increment12-8nodes/pretrain_ckpts/ckpt_$CHECKPOINT_ITER.pt $OUTPUT_DIR/pretrain_ckpts


        mpirun -np $NNODES -hostfile $HOSTFILE -ppn 1 bash scripts/launch_pretraining.sh  \
                --ngpus $PROC_PER_NODE --nnodes $NNODES --master $MASTER_RANK --singularity $SINGULARITY --phase $PHASE \
                --kwargs \
                --input_dir /tmp/data/phase${PHASE} \
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
                --sparsity_increment ${SPARSITY_INCREMENT}
    done
done
sleep 172800
