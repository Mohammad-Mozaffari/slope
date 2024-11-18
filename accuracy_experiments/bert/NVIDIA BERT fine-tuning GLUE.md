# NVIDIA BERT fine-tuning GLUE

This file contains the detailed instructions for fine-tuning BERT with GLUE dataset. 

Before reading the rest of the instructions, please make sure that your BERT is pretrained with [scripts](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/README.md#pre-training) provided by NVIDIA. Otherwise, it might have problems loading the checkpoint.

Please make sure you have [apex]([GitHub - NVIDIA/apex: A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch](https://github.com/NVIDIA/apex)) installed before reading the rest.

## Prepare GLUE datasets

Use [scripts/download_glue_data.sh](https://github.com/zhengmk321/BERT_Finetuning/blob/master/scripts/download_glue_data.sh) to download all the GLUE datasets.

Please change the cmd varaible after `--data_dir` to your own directory.

## How to run

### Finetune all GLUE datasets

You can run finetune all GLUE datasets with one script: [scripts/run_all_glue_nv.sh](https://github.com/zhengmk321/BERT_Finetuning/blob/master/scripts/run_all_glue_nv.sh).

```shell
#!/bin/bash

ckpt_dir="/work2/00946/zzhang/lossy-ckpts/1e-6" # Assume all the pretrained ckpts are stored under this directory
out="../results/bert_pretraining/GLUE" # Define where the results will be stored
names=$( hostname )

arr=($(echo "$names" | tr '.' '\n'))
server="${arr[1]}"


if [ $server = 'frontera' ]; then
  num_gpu_per_node='4'
elif [ $server = 'ls6' ]; then
  num_gpu_per_node='3'
else
  num_gpu_per_node='1'
fi
total_num_gpus=4

    for sec_dir in $ckpt_dir; do
        files=($sec_dir/*.pt)
        for ((i=${#files[@]}-1; i>=0; i--)); do
            FILE="${files[$i]}"

            arr=($(echo "$FILE" | tr '/' '\n'))
            ckpt_file="${arr[-1]}"

            folder_name="${arr[-2]}"

            arr=($(echo "$ckpt_file" | tr '.' '\n'))
            ckpt_name="${arr[0]}"

            out_dir="$out/$folder_name/$ckpt_name" 
            
            # Iterate through all the GLUE datasets
            for task in 'MNLI' 'QQP' 'QNLI' 'SST-2'  'CoLA' 'STS-B' 'MRPC' 'RTE'; do
                batch_size=16
                CMD="./run_one_glue_nv.sh $task $FILE  $num_gpu_per_node $batch_size $out_dir"
                $CMD
            done
        done
    done

```

The script above runs all the fine-tuning tasks using one node with data parallel. If you have multiple nodes, you can parition the tasks onto different nodes.

### Finetune a specific GLUE dataset

Or, if you want, You can finetune a specific GLUE dataset with [scripts/run_one_glue_nv.sh](https://github.com/zhengmk321/BERT_Finetuning/blob/master/scripts/run_one_glue_nv.sh).

### Finetune GLUE datasets using different seeds

It's always good to run multiple experiment to get a more general performance. You can do this using [scripts/run_all_glue_nv_seeds.sh](https://github.com/zhengmk321/BERT_Finetuning/blob/master/scripts/run_all_glue_nv_seeds.sh).


