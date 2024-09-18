#!/bin/bash
# zeroshot source fixed_original_test fixed_original_test_corrections
# finetune target fixed_original_test fixed_original_test_corrections

# get args
model_task=$1
goal=$2
train_dataset=$3
test_dataset=$4
mask=$5
model_path=$6
model_base=$7


if [ "$model_task" == "zeroshot" ]; then
  # zero-shot
#  MODEL_NAME=liuhaotian/llava-v1.5-7b
  CUDA_VISIBLE_DEVICES=0 python eval_llava.py \
      --model_path $model_base \
      --model_task zeroshot \
      --train_dataset $train_dataset \
      --test_dataset $test_dataset \
      --turn_masking $mask \
      --prompt_task_instruction system \
      --goal $goal \
      --image-folder /users/fjc3/sharedscratch/generated_data/576p \
      --temperature 0 \
      --conv-mode v1_bw_${goal}

#      --output_file /users/fjc3/sharedscratch/checkpoints/llava/$model_base/${test_dataset}-${goal}-train.json \

  exit 0
fi


# fine-tuned model
#MODEL_NAME=$3
#MODEL_BASE=$4
#MODEL_NAME=/users/fjc3/sharedscratch/checkpoints/llava/llava-v1.5-7b-task-lora-${goal}
#MODEL_NAME=/users/fjc3/sharedscratch/checkpoints/llava/llava-v1.5-7b-task-qlora-source-merged-lora-merge

#python scripts/merge_lora_weights.py \
#  --model-path $MODEL_NAME \
#  --model-base liuhaotian/llava-v1.5-7b \
#  --save-model-path $MODEL_NAME-merged

#exit 0

python eval_llava.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_task finetune \
    --train_dataset $train_dataset \
    --test_dataset $test_dataset \
    --turn_masking $mask \
    --prompt_task_instruction system \
    --goal $goal \
    --image-folder /users/fjc3/sharedscratch/generated_data/576p \
    --temperature 0 \
    --conv-mode v1_bw_${goal}

#    --answers-file $model_path/evaluated_${test_dataset}-${goal}-train.json \

echo "Saved to ${model_path}/evaluated_${test_dataset}-${goal}-train.json.log"
