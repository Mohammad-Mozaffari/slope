cd ../

PHASE=1

export OMP_NUM_THREADS=8
#module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
#source activate pytorch

PRUNED_MATRIX="static-weight"
SKIP_FIRST_BLOCK=True
SKIP_LAST_BLOCK=True
SKIP_ATTENTION=False
ADD_LORA=False
LORA_RANK=40
REDUCTION_DIM=True
SPARSITY_INCREMENT=12 #Set to -1 to disable sparsity increment

torchrun --nproc_per_node=1 run_pretraining.py \
    --input_dir ./phase${PHASE} \
    --output_dir results/phase${PHASE}_local \
    --config_file config/bert_pretraining_phase${PHASE}_config.json \
    --weight_decay 0.01 \
    --optimizer fused-lamb \
    --num_steps_per_checkpoint 200 \
    --global_batch_size 4096 \
    --local_batch_size 1 \
    --pruned_matrix ${PRUNED_MATRIX} \
    --skip_first_block ${SKIP_FIRST_BLOCK} \
    --skip_last_block ${SKIP_LAST_BLOCK} \
    --skip_attention ${SKIP_ATTENTION} \
    --reduction_dim ${REDUCTION_DIM} \
    --add_lora ${ADD_LORA} \
    --lora_rank ${LORA_RANK} \
    --sparsity_increment ${SPARSITY_INCREMENT}

#for i in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400
#do
#  mv results/phase1_local/tmp/ckpt_$i.pt results/phase1_local/pretrain_ckpts/ckpt_$i.pt
#  torchrun --nproc_per_node=1 run_pretraining.py \
#      --input_dir ./phase${PHASE} \
#      --output_dir results/phase${PHASE}_local \
#      --config_file config/bert_pretraining_phase${PHASE}_config.json \
#      --weight_decay 0.01 \
#      --optimizer fused-lamb \
#      --num_steps_per_checkpoint 200 \
#      --global_batch_size 4096 \
#      --local_batch_size 1
#  mv model_info.csv model_info_$i.csv
#done