#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=llava_eval
#SBATCH --output=/users/fjc3/sharedscratch/scrum/llava_zeroshot_experiments.%J.out
#SBATCH --gres gpu:1
#SBATCH --chdir=/users/fjc3/block-world-training/LLaVA

# sample run:
#     sbatch llava_zeroshot_experiments.sh liuhaotian/llava-v1.5-7b
#     sbatch llava_zeroshot_experiments.sh liuhaotian/llava-v1.6-mistral-7b

flight env activate gridware
module load libs/nvidia-cuda

source /users/fjc3/.bashrc
conda activate /users/fjc3/block-world-training/.envs/llava

echo "Starting training in dmog"


train_dataset=NA
mask=none
model_base=$1


test_dataset=(fixed_original_test_instructions fixed_original_test_corrections)


# calculate total number of experiments
total_experiments=$((${#test_dataset[@]} * 2))
echo "Total number of experiments: $total_experiments"

for test_data in "${test_dataset[@]}"; do
    for goal in "source" "target"; do
        python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/zeroshot --dataset $test_data --turn_masking $mask

        echo "Command: sh scripts/v1_5/eval_task_lora.sh zeroshot $goal $train_dataset $test_data NA $mask $model_base"
        sh scripts/v1_5/eval_task_lora.sh zeroshot $goal $train_dataset $test_data $mask NA $model_base
    done
done


# Define the folder containing the files
#folder="/users/fjc3/sharedscratch/scrum/"
#
## Iterate over each file in the folder
#for file in "${folder}"*; do
#    # Check if the file is a regular file
#    if [[ -f "$file" ]]; then
#        # Process the file
#        sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g' "$file" > "${file}.2out"
#        mv "${file}.2out" "$file"
##        echo "Processed file: $file"
#    fi
#done
