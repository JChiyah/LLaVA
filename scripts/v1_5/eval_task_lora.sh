#!/bin/bash
# zeroshot source fixed_original_test fixed_original_test_corrections
# finetune target fixed_original_test fixed_original_test_corrections
set -e

# get args
model_task=$1
goal=$2
test_dataset=$3
mask=$4
model_path=$5
model_base=$6


readonly LLAVA_ROOT="$(dirname "$0")/../../"


if [ "$model_task" == "zeroshot" ]; then
  # zero-shot
  # MODEL_NAME=liuhaotian/llava-v1.5-7b
  CUDA_VISIBLE_DEVICES=0 python $LLAVA_ROOT/eval_llava.py \
      --model_path $model_base \
      --model_task eval \
      --test_dataset $test_dataset \
      --turn_masking $mask \
      --prompt_task_instruction system \
      --goal $goal \
      --image-folder /users/fjc3/sharedscratch/generated_data/576p \
      --temperature 0 \
      --conv-mode v1_bw_${goal} \
      "${@:7}"

#      --output_file /users/fjc3/sharedscratch/checkpoints/llava/$model_base/${test_dataset}-${goal}-train.json \
    #   --train_dataset $train_dataset \

  exit 0
else
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

    python $LLAVA_ROOT/eval_llava.py \
        --model_path $model_path \
        --model_base $model_base \
        --model_task eval \
        --test_dataset $test_dataset \
        --turn_masking $mask \
        --prompt_task_instruction system \
        --goal $goal \
        --image-folder /users/fjc3/sharedscratch/generated_data/576p \
        --temperature 0 \
        --conv-mode v1_bw_${goal} \
        "${@:7}"

        # --train_dataset $train_dataset \
    #    --answers-file $model_path/evaluated_${test_dataset}-${goal}-train.json \
fi


echo "Saved to ${model_path}/evaluated_${test_dataset}-${goal}-train.json.log"
