

OPT_SIZES="768 1024 2048 2560 4096 5120 7168 9216"

for dim in $OPT_SIZES
do
    python setup_overhead.py --d $dim
done