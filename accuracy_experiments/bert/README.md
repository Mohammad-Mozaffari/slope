# SLoPe: Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining of LLMs

This repository is copied from the [NVIDIA BERT implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) 
and modified 
to support the SLoPe pretraining method. The original README.md is available 
[here](SETUP.md), please refer to it for setting up the code.

## SLoPe Pretraining

For pretraining BERT using SLoPe, you can use the following command in a 
setting (sample scripts are available in the `scripts` directory in bash
files `run_pretraining_jobs_*.sh`):

```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$NNODES --node_rank=$RANK --master_addr=$MAIN_RANK --master_port=4321\
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
```

We use the following settings for the SLoPe pretraining:

```bash
OPTIMIZER="fused-lamb"
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


## Fine-tuning on Downstream Tasks

We fine-tune the models on GLUE and SQuAD v1.1 downstream tasks. Sample fine-tuning
scripts are available in `scripts/run_glue_mist.sh` and `scripts/run_squad_mist.sh`.
The flags in fine-tuning scripts are similar to the pretraining scripts.

More details about gathering the fine-tuning dataset and running the fine-tuning task
are provided in [this](NVIDIA%20BERT%20fine-tuning%20GLUE.md) document.



