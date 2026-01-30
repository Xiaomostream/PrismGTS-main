#!/bin/bash
# Runs ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
# Prediction Lengths: 96, 192, 336, 720

export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs/zero_shot" ]; then
    mkdir -p ./logs/zero_shot
fi

# ============== Configuration ==============
seq_len=512
label_len=48
patch_len=64
stride=64

num_experts=8
prompt_len=8
lambda_smooth=0.1
lambda_periodic=0.1   

# Model settings
model_name=PrismGTS
d_model=128
e_layers=3
d_layers=3

# Pretrained Model Path (Please update this path after pretraining)
PRETRAINED_PATH="./checkpoints/0-shot/PrismGTS.pth"

# ============== Function to run zero-shot evaluation ==============
run_zeroshot() {
    local data_name=$1
    local root_path=$2
    local data_path=$3
    
    echo ""
    echo "###############################################################"
    echo "# Dataset: $data_name"
    echo "# checkpoint: $PRETRAINED_PATH"
    echo "###############################################################"
    
    for pred_len in 96 192 336 720
    do
        echo ""
        echo "----------------------------------------------------------------"
        echo "[$data_name] Pred Len: $pred_len | Checkpoint: $PRETRAINED_PATH"
        echo "----------------------------------------------------------------"
        
        python run.py \
            --task_name long_term_forecast \
            --is_training 0 \
            --is_pretraining 0 \
            --is_zeroshot 1 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id 3_3_128_${data_name}_${seq_len}_${pred_len} \
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
            --batch_size 128 \
            --head_type prediction \
            --num_workers 10 \
            --pretrained_path "$PRETRAINED_PATH" \
            --des "zero_shot_${data_name}" \
            --itr 1 2>&1 | tee -a logs/zero_shot/${model_name}_${data_name}_${pred_len}.log
    done
}

# ============== Run All Datasets ==============

echo "============================================================"
echo "PrismGTS Zero-Shot Evaluation - All Datasets"
echo "Prediction Lengths: 96, 192, 336, 720"
echo "Pretrained Model: $PRETRAINED_PATH"
echo "============================================================"


# run_zeroshot "ETTh1" "./dataset/ETT-small/" "ETTh1.csv"
# run_zeroshot "ETTh2" "./dataset/ETT-small/" "ETTh2.csv"
# run_zeroshot "ETTm1" "./dataset/ETT-small/" "ETTm1.csv"
# run_zeroshot "ETTm2" "./dataset/ETT-small/" "ETTm2.csv"
# run_zeroshot "weather" "./dataset/weather/" "weather.csv"
run_zeroshot "electricity" "./dataset/electricity/" "electricity.csv"
run_zeroshot "traffic" "./dataset/traffic/" "traffic.csv"