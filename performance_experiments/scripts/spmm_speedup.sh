# This script is tuned to fit the models in A100-40GB GPUs


BACKEND="cutlass"

OPT_SIZES1="768 1024 2048 2560 4096 5120"
for dim1 in $OPT_SIZES1
do
    # Upsample and Downsample Matrices
    dim2=$(($dim1 * 4))
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 50 --num_experiments 100 \
    --backend $BACKEND
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 50 --num_experiments 100 \
    --backend $BACKEND --tiling
    # Attention Matrices
    dim2=$(($dim1 * 1))
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 50 --num_experiments 100 \
        --backend $BACKEND
done

OPT_SIZES2="7168 9216"
for dim1 in $OPT_SIZES2
do
    # Upsample and Downsample Matrices
    dim2=$(($dim1 * 4))
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 6 --num_experiments 400 \
    --backend $BACKEND
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 6 --num_experiments 400 \
    --backend $BACKEND --tiling
    # Attention Matrices
    dim2=$(($dim1 * 1))
    python performance_experiments/layer_speedup.py --d_in $dim1 --d_out $dim2 --num_layers 6 --num_experiments 400 \
        --backend $BACKEND
done