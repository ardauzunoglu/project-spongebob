#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="train_a"
#SBATCH --output="pruning.txt" # Path to store logs

export HF_DATASETS_CACHE="/scratch4/danielk/auzunog1/hf_cache"
export HF_HOME="/scratch4/danielk/auzunog1/hf_cache"
export CACHE_DIR="/scratch4/danielk/auzunog1/hf_cache"
export TRITON_CACHE_DIR="/scratch4/danielk/auzunog1/hf_cache"

source ~/.bashrc
source activate data_utility

models=(
    "facebook/opt-125m"
    "facebook/opt-350m"
    "facebook/opt-1.3b"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"
    "facebook/opt-13b"
    "facebook/opt-30b"
)

# Example list of tasks. Change as needed
tasks=(
    "openbookqa"
    "arc_challenge"
    "arc_easy"
    "copa"
    "commonsense_qa"
    "logiqa"
    "piqa"
    "pubmed_qa"
    "social_i_qa"
    "truthful_qa"
    "winogrande"
    "boolq"
)

# Nested loops: For each model and for each task, run the script
for task in "${tasks[@]}"
do
    for model in "${models[@]}"
    do
        echo "Running collect_hs.py with --model_name $model and --dataset $task"
        srun python collect_hs.py --model_name "$model" --dataset "$task"
    done
done
