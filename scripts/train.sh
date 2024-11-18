HF_TOKEN="ADD_YOUR_HF_TOKEN"
TILING='--tiling'
SPARSE='--sparse'
DTYPE='bf16'
BACKEND='cutlass'
MODEL="facebook/opt-125m"

torchrun --nproc_per_node=1 --nnodes=1 train.py \
    --hf_token $HF_TOKEN \
    $SPARSE \
    --dtype $DTYPE \
    --backend $BACKEND \
    --model $MODEL \
    $TILING \

