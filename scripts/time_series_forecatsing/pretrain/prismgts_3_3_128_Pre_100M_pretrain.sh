#!/bin/bash
# PrismGTS: Split Decoder Training Script 

export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs/PrismGTS" ]; then
    mkdir -p ./logs/PrismGTS
fi

if [ ! -d "./logs/Pretrain" ]; then
    mkdir -p ./logs/Pretrain
fi

data_name=utsd
seq_len=512
label_len=48
patch_len=64
stride=64

num_experts=8
prompt_len=8
lambda_smooth=0.1
lambda_periodic=0.1

for pred_len in 96
do
  # Stage 1: Pretraining (Reconstruction)
  python run.py \
    --task_name long_term_forecast \
    --is_pretraining 1 \
    --is_training 0 \
    --root_path ./dataset/$data_name/ \
    --data_path $data_name.npy \
    --model_id prismgts_v4_3_3_128_Pre_160M_${seq_len}_${pred_len} \
    --model PrismGTS \
    --data UTSD \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --c_in 1 \
    --e_layers 3 \
    --d_layers 3 \
    --d_model 128 \
    --domain_len 128 \
    --num_experts $num_experts \
    --prompt_len $prompt_len \
    --lambda_smooth $lambda_smooth \
    --lambda_periodic $lambda_periodic \
    --percent 100 \
    --pretrain_epochs 10 \
    --batch_size 16384 \
    --learning_rate 1e-3 \
    --warmup_steps 10000 \
    --lradj constant_with_warmup \
    --head_type pretrain \
    --num_workers 10 \
    --patience 6 \
    --des 'V4_3_3_128_Pre_160M_stage1' \
    --itr 1 2>&1 | tee logs/Pretrain/prismgts_v4_3_3_128_Pre_160M_${seq_len}_${pred_len}_stage1.log
done


for pred_len in 96
do
  # Stage 2: Tuning (Prediction)
  python run.py \
    --task_name long_term_forecast \
    --is_pretraining 0 \
    --is_training 1 \
    --root_path ./dataset/$data_name/ \
    --data_path $data_name.npy \
    --model_id prismgts_v4_3_3_128_Pre_160M_${seq_len}_${pred_len} \
    --model PrismGTS \
    --data UTSD \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --c_in 1 \
    --e_layers 3 \
    --d_layers 3 \
    --d_model 128 \
    --domain_len 128 \
    --num_experts $num_experts \
    --prompt_len $prompt_len \
    --lambda_smooth 0.0 \
    --lambda_periodic 0.0 \
    --percent 100 \
    --train_epochs 20 \
    --batch_size 16384 \
    --learning_rate 1e-3 \
    --warmup_steps 10000 \
    --lradj constant_with_warmup \
    --head_type prediction \
    --num_workers 20 \
    --patience 6 \
    --des 'V4_3_3_128_Pre_160M_stage2' \
    --itr 1 2>&1 | tee logs/Pretrain/prismgts_v4_3_3_128_Pre_160M_${seq_len}_${pred_len}_stage2.log
done
