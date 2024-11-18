# SLoPe: Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining of LLMs

This repository is copied from the [FlashAttention implementation](https://github.com/Dao-AILab/flash-attention) 
and modified to support the SLoPe pretraining method. The original README.md
is available [here](FLASH_ATTENTION_README.md).
In addition, please refer to [this](training/README.md) document for setting up the code.

## SLoPe Pretraining
For pretraining BERT using SLoPe, you can use the following command in a 
setting (sample scripts are available in the `scripts` directory in bash
files `run_jobs_*.sh`):

```bash
python  training/run.py \
        experiment=owt/$MODEL \
        trainer.devices=$NGPU_PER_NODE \ 
        datamodule.batch_size=$BATCH_SIZE \
        trainer.num_nodes=$NNODES \
        name=$NAME \
        resume=True \
        logger=csv \
        optimizer=$OPTIMIZER \
        pruner.pruned_matrix=$PRUNED_MATRIX \
        pruner.skip_attention=$SKIP_ATTENTION \ 
        pruner.skip_first_block=$SKIP_FIRST_BLOCK \ 
        pruner.skip_last_block=$SKIP_LAST_BLOCK \
        pruner.reduction_dim=$REDUCTION_DIM \
        pruner.add_lora=$ADD_LORA \
        pruner.lora_rank=$LORA_RANK \
        trainer.max_steps=400000
```

We use the following settings for the SLoPe pretraining:

```bash
OPTIMIZER="adamw" # Or "adamw-zero" depending on the default settings of the model
PRUNED_MATRIDX="static-weight" # Corresponds to the double-pruning method
SKIP_FIRST_BLOCK="False" # Skip the first block of the model
SKIP_LAST_BLOCK="False" # Skip the last block of the model
SKIP_ATTENTION="False" # Skip the attention layers
REDUCTION_DIM="True" # Should be set to True to prune along the reduction dimension to use 2:4 backends
UNSTRUCTURED_MASKING="False" # Use unstructured masking
SPARSITY_INCREMENT=-1 # All layers will be pruned using 2:4 sparsity
SPARSITY_INCREMENT=0,12,-1 # First 12 layers will be pruned using 2:8 sparsity, the rest will be pruned using 2:4 sparsity
SPARSITY_INCREMENT=12 # First 12 layers will be pruned using 2:4 sparsity, the rest will be pruned using 2:8 sparsity
```

We use `ADD_LORA="True"` only in the last 1% of training.