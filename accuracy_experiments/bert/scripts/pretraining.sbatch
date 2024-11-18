#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J bertmkor                 # Job name
#SBATCH -o sbatch_logs/bertmkor.o%j # Name of stdout output file
#SBATCH -N 2                      # Total # of nodes 
#SBATCH -n 2                      # Total # of mpi tasks
#SBATCH -t 48:00:00                # Run time (hh:mm:ss)
#SBATCH -p gpu-a100
#SBATCH -A TACC-DIC    # Allocation

mkdir -p sbatch_logs


HOSTFILE=hostfile
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=4

# PHASE 1
mpirun -np $NODES -hostfile $HOSTFILE  bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --config config/bert_pretraining_phase1_config.json \
	--input data/hdf5/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/ \
    --output results/bert_pretraining

# PHASE 2
mpirun -np $NODES -hostfile $HOSTFILE  bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --config config/bert_pretraining_phase2_config.json \
	--input data/hdf5/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/ \
    --output results/bert_pretraining

