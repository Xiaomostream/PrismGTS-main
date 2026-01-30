#!/bin/bash
# Runs ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
# Prediction Lengths: 96, 192, 336, 720

export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs/few_shot" ]; then
    mkdir -p ./logs/few_shot
fi

# ============== Configuration ==============
seq_len=512
label_len=48
patch_len=64
stride=64

num_experts=8
prompt_len=8
lambda_smooth=0.0
lambda_periodic=0.0

# Model settings
model_name=PrismGTS
d_model=128
e_layers=3
d_layers=3

# Pretrained Model Path (Please update this path after pretraining)
PRETRAINED_PATH="./checkpoints/0-shot/PrismGTS.pth"

# ============== Function to run few-shot evaluation ==============
run_fewshot() {
    local data_name=$1
    local root_path=$2
    local data_path=$3
    
    echo ""
    echo "###############################################################"
    echo "# Dataset: $data_name"
    echo "# checkpoint: $PRETRAINED_PATH"
    echo "###############################################################"
    
    for pred_len in 96 192 192 336 720
    do
        echo ""
        echo "----------------------------------------------------------------"
        echo "[$data_name] Pred Len: $pred_len | Checkpoint: $PRETRAINED_PATH"
        echo "----------------------------------------------------------------"
        
        python run.py \
            --task_name long_term_forecast \
            --is_pretraining 0 \
            --is_training 1 \
            --is_zeroshot 0 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id 3_3_128_10percent_${data_name}_${seq_len}_${pred_len} \
            --model $model_name \
            --data CI \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --patch_len $patch_len \
            --stride $stride \
            --c_in 1 \
            --e_layers $e_layers \
            --d_layers $d_layers \
            --d_model $d_model \
            --domain_len 128 \
            --num_experts $num_experts \
            --prompt_len $prompt_len \
            --lambda_smooth $lambda_smooth \
            --lambda_periodic $lambda_periodic \
            --learning_rate 1e-4 \
            --batch_size 128 \
            --warmup_steps 10000 \
            --lradj constant_with_warmup \
            --head_type prediction \
            --train_epochs 10 \
            --percent 10 \
            --num_workers 10 \
            --pretrained_path "$PRETRAINED_PATH" \
            --des "few_shot_10percent_${data_name}" \
            --itr 1 2>&1 | tee -a logs/few_shot/${model_name}_10percent${data_name}_${pred_len}.log
    done
}

# ============== Run All Datasets ==============

echo "============================================================"
echo "PrismGTS Few-Shot Evaluation - All Datasets"
echo "Prediction Lengths: 96, 192, 336, 720"
echo "Pretrained Model: $PRETRAINED_PATH"
echo "============================================================"

run_fewshot "ETTh1" "./dataset/ETT-small/" "ETTh1.csv"
run_fewshot "ETTh2" "./dataset/ETT-small/" "ETTh2.csv"
run_fewshot "ETTm1" "./dataset/ETT-small/" "ETTm1.csv"
run_fewshot "ETTm2" "./dataset/ETT-small/" "ETTm2.csv"
run_fewshot "weather" "./dataset/weather/" "weather.csv"
run_fewshot "electricity" "./dataset/electricity/" "electricity.csv"
run_fewshot "traffic" "./dataset/traffic/" "traffic.csv"


