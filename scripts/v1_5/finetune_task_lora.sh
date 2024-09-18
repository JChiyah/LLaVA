#!/bin/bash
# source original_entries original_entries liuhaotian/llava-v1.5-7b none

# get args
goal=$1
train_dataset=$2
test_dataset=$3
mask=$4
model_base=$5

echo "scripts/v1_5/finetune_task_lora.sh $@"


# make sure goal is 'source' or 'target'
if [ "$goal" != "source" ] && [ "$goal" != "target" ]; then
    echo "Invalid goal: $goal"
    exit 1
fi

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
#MODEL_BASE=liuhaotian/llava-v1.5-7b
model_short="${model_base#*/}"

output_dir=/users/fjc3/sharedscratch/checkpoints/llava/$timestamp-$model_short-lora-${train_dataset}-${goal}

echo 'starting!'

WANDB_MODE=offline deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --tune_mm_mlp_adapter True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_base \
    --version v1_bw_${goal} \
    --data_path /users/fjc3/sharedscratch/datasets/llava/${mask}/${train_dataset}-${goal}-train.json \
    --image_folder /users/fjc3/sharedscratch/generated_data/576p \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --seed 124124 \
    --turn_masking $mask \
    --prompt_task_instruction system
#    --report_to wandb

echo "Done finetuning, model in $output_dir"
#    --data_path ./playground/data/llava_v1_5_mix665k.json \
#    --image_folder ./playground/data \
#    --bits 4 \
# --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5

#echo "sh scripts/v1_5/eval_task_lora.sh finetune $goal $train_dataset $test_dataset $mask $output_dir $model_base"
#sh scripts/v1_5/eval_task_lora.sh finetune $goal $train_dataset $test_dataset $mask $output_dir $model_base

# Evaluate in all test datasets, saving some time
# (fixed_original_test_instructions fixed_original_test_corrections)
test_datasets=(fixed_original_test_instructions fixed_original_test_corrections)

for test_dataset in "${test_datasets[@]}"; do
    cd ..
    python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/$mask --dataset $test_dataset --turn_masking $mask

    cd LLaVA/
    echo "sh scripts/v1_5/eval_task_lora.sh finetune $goal $train_dataset $test_dataset $mask $output_dir $model_base"
    sh scripts/v1_5/eval_task_lora.sh finetune $goal $train_dataset $test_dataset $mask $output_dir $model_base

done



echo 'Done with finetune_task_lora.sh!'
