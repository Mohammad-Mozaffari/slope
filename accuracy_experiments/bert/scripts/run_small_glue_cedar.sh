#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --nodes 1
#SBATCH --mem=20G
#SBATCH -t 18:00:00
#SBATCH --account=def-mmehride
#SBATCH --output=/home/islahahm/projects/def-mmehride/islahahm/bert-mkor/scripts/slurmoutput/%j.out


for task in 'CoLA' 'STS-B' 'MRPC' 'RTE'; do
    CMD="./run_one_glue_cedar_pruned.sh $task"
    $CMD
done



