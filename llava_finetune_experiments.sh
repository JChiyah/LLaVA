#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=llava_exps
#SBATCH --output=/users/fjc3/sharedscratch/scrum/llava_finetune_experiments.%J.out
#SBATCH --gres gpu:1
#SBATCH --chdir=/users/fjc3/block-world-training/LLaVA

# sample run:
#     sbatch llava_finetune_experiments.sh source fixed_original_test_instructions liuhaotian/llava-v1.5-7b
#     sbatch llava_finetune_experiments.sh source fixed_original_test_corrections liuhaotian/llava-v1.5-7b
#     sbatch llava_finetune_experiments.sh source fixed_original_test_corrections liuhaotian/llava-v1.6-mistral-7b

flight env activate gridware
module load libs/nvidia-cuda

source /users/fjc3/.bashrc
conda activate /users/fjc3/block-world-training/.envs/llava

echo "Starting training in dmog"


goal=$1
train_dataset=$2
model_base=$3


# 2 datasets to test: instructions and corrections
# (fixed_original_test_instructions fixed_original_test_corrections)
test_datasets=(fixed_original_test_instructions fixed_original_test_corrections)
# 3 possible train datasets: instructions, corrections, both
# (fixed_original_test fixed_original_test_instructions fixed_original_test_corrections)
train_datasets=(fixed_original_test)
# (none all)
turn_masking=(all assistant none)				# assistant masking not needed, see logs


#if [ "$test_dataset" == "fixed_original_test_instructions" ]; then
        # does not make sense to use corrections as train data for instructions
#        train_datasets=(fixed_original_test_instructions fixed_original_test)
        # else: we keep the default with 3 train datasets
#    fi

# calculate total number of experiments
total_experiments=$(( ${#test_datasets[@]} * ${#turn_masking[@]}))
echo "Total number of experiments: $total_experiments"

test_dataset=$train_dataset


for mask in "${turn_masking[@]}"; do
    cd ..
    python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/$mask --dataset $test_dataset --turn_masking $mask
    python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/$mask --dataset $train_dataset --turn_masking $mask

    cd LLaVA/
    echo "sh scripts/v1_5/finetune_task_lora.sh $goal $train_dataset $test_dataset $mask $model_base"
    sh scripts/v1_5/finetune_task_lora.sh $goal $train_dataset $test_dataset $mask $model_base
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
