#!/bin/bash

NGPUS=1
NNODES=1
LOCAL_RANK=""
MASTER=""
KWARGS=""

while [[ "$1" == -* ]]; do
    case "$1" in
        -h|--help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  -h,--help           Display this help message"
            echo "  -N,--ngpus  [int]   Number of GPUs per node (default: 1)"
            echo "  -n,--nnodes [int]   Number of nodes this script is launched on (default: 1)"
            echo "  -r,--rank   [int]   Node rank (default: \"\")"
            echo "  -m,--master [str]   Address of master node (default: \"\")"
            echo "  -a,--kwargs [str]   Training arguments. MUST BE LAST ARG! (default: \"\")"
            exit 0
        ;;
        -N|--ngpus)
            shift
            NGPUS="$1"
        ;;
        -n|--nnodes)
            shift
            NNODES="$1"
        ;;
        -m|--master)
            shift
            MASTER="$1"
        ;;
        -r|--rank)
            shift
            LOCAL_RANK="$1"
        ;;
        -s|--singularity)
            shift
            SINGULARITY="$1"
        ;;
        -p|--phase)
            shift
            PHASE="$1"
        ;;
        -a|--kwargs)
            shift
            KWARGS="$@"
        ;;
        *)
          echo "ERROR: unknown parameter \"$1\""
          exit 1
        ;;
    esac
    shift
done

# echo Activating conda environment
# # source /work/00946/zzhang/ls6/python-envs/torch-1.13/bin/activate
# # conda activate bert-pytorch
# source /work/00946/zzhang/ls6/python-envs/torch-1.13/bin/activate
# echo Conda environment activated. Python Path:
# which python

if [[ -z "$LOCAL_RANK" ]]; then
    if [[ -z "${OMPI_COMM_WORLD_RANK}" ]]; then
        LOCAL_RANK=${MV2_COMM_WORLD_RANK}
    else
        LOCAL_RANK=${OMPI_COMM_WORLD_RANK}
    fi
fi

LOCAL_RANK=${PMI_RANK}

NUM_THREADS=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')
export OMP_NUM_THREADS=$((NUM_THREADS / NGPUS))

PWD=$(pwd)

if [ "$SINGULARITY" = True ]; then
    mkdir /tmp/torch.sif 
    tar -xf /scratch/09778/rafii/torch.tar -C /tmp
    SINGULARITY="singularity exec --bind $PWD:/home --bind /tmp:/tmp --nv /tmp/torch.sif "
else
    source /scratch/09778/rafii/pytorch/bin/activate
    SINGULARITY=""
fi

mkdir /tmp/data 
tar -xf /scratch/09778/rafii/datasets/bert/pretraining/phase${PHASE}.tar -C /tmp/data

module load tacc-apptainer
export OMP_NUM_THREADS=8


echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, OMP_NUM_THREADS=$OMP_NUM_THREADS, host=$HOSTNAME
echo $SINGULARITY python -m torch.distributed.launch --nproc_per_node=$NGPUS run_pretraining.py $KWARGS


if [[ "$NNODES" -eq 1 ]]; then
    $SINGULARITY python -m torch.distributed.launch --nproc_per_node=$NGPUS \
        run_pretraining.py $KWARGS
else
    $SINGULARITY python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS --nnodes=$NNODES \
        --node_rank=$LOCAL_RANK --master_addr=$MASTER \
        run_pretraining.py $KWARGS
fi
