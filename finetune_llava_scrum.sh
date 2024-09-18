#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=llava_ft
#SBATCH --output=/users/fjc3/sharedscratch/scrum/llava.%J.out
#SBATCH --gres gpu:1
#SBATCH --chdir=/users/fjc3/block-world-training/LLaVA

# sample run:
#   sbatch finetune_llava_scrum.sh source fixed_original_test_instructions fixed_original_test_instructions liuhaotian/llava-v1.5-7b none
#   sbatch finetune_llava_scrum.sh target fixed_original_test_instructions fixed_original_test_instructions liuhaotian/llava-v1.5-7b all

flight env activate gridware
module load libs/nvidia-cuda

source /users/fjc3/.bashrc
conda activate /users/fjc3/block-world-training/.envs/llava

goal=$1
train_dataset=$2
test_dataset=$3
mask=$4
model_base=$5


echo "Loading data"

check_last_command_success() {
    # $1 is the status
    # check if the command was successful
    if [ $1 -ne 0 ]; then
        echo "Command failed! Stopping..."
        exit 1
    fi
}

cd ..
python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/$mask --dataset $train_dataset --turn_masking $mask
check_last_command_success $?
python convert_to_llava.py --output_dir /users/fjc3/sharedscratch/datasets/llava/$mask --dataset $test_dataset --turn_masking $mask
check_last_command_success $?


echo "Starting training in dmog"

cd LLaVA/

echo "sh scripts/v1_5/finetune_task_lora.sh $goal $train_dataset $test_dataset $mask $model_base"
sh scripts/v1_5/finetune_task_lora.sh $goal $train_dataset $test_dataset $mask $model_base


#echo "sh scripts/v1_5/eval_task_lora.sh $@"
#sh scripts/v1_5/eval_task_lora.sh $@


# Define the folder containing the files
#folder="/users/fjc3/sharedscratch/scrum/"

# Iterate over each file in the folder
#for file in "${folder}"*; do
#    # Check if the file is a regular file
#    if [[ -f "$file" ]]; then
#        # Process the file
#        sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g' "$file" > "${file}.2out"
#        mv "${file}.2out" "$file"
##        echo "Processed file: $file"
#    fi
#done
