

OPT_SIZES="768 1024 2048 2560 4096 5120 7168 9216"

for dim in $OPT_SIZES
do
    python performance_experiments/matmul_utilization.py --d $dim
done