cd ../

OPTIMIZER=mkor
PRUNED_MATRIX="none"
SKIP_FIRST_BLOCK=True
SKIP_LAST_BLOCK=True
SKIP_ATTENTION=True
REDUCTION_DIM=True
ADD_LORA=True
BATCH_SIZE=1
ACCELERATE=True
MODEL=gpt2s


#module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
#source activate pytorch

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/training/compression/pruning_kernels/tensor_cores/libcusparse_lt/lib

PWD=$(pwd)
export WANDB_API_KEY=755c093d24b1dcff6500f39cc878c3e6b6da5653
export NAME=$MODEL-$OPTIMIZER-local
export HYDRA_FULL_ERROR=1
export DATA_DIR=$PWD/data
export HF_DATASETS_CACHE=$PWD/data
export WANDB_MODE=offline

rm -rf checkpoints/$NAME

python training/run.py \
        experiment=owt/$MODEL \
        trainer.devices=1 \
        datamodule.batch_size=$BATCH_SIZE \
        train.global_batch_size=8  \
        logger=csv \
        name=$NAME \
        resume=True \
        optimizer=$OPTIMIZER \
        pruner.pruned_matrix=$PRUNED_MATRIX \
        pruner.skip_attention=$SKIP_ATTENTION \
        pruner.skip_first_block=$SKIP_FIRST_BLOCK \
        pruner.skip_last_block=$SKIP_LAST_BLOCK \
        pruner.reduction_dim=$REDUCTION_DIM \
        pruner.add_lora=$ADD_LORA \
        trainer.max_steps=1000 \
        pruner.accelerate=$ACCELERATE
