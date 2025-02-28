#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="pruning"
#SBATCH --output="pruning_perturbated.txt" # Path to store logs

export HF_DATASETS_CACHE="/scratch4/danielk/auzunog1/hf_cache"
export HF_HOME="/scratch4/danielk/auzunog1/hf_cache"
export CACHE_DIR="/scratch4/danielk/auzunog1/hf_cache"
export TRITON_CACHE_DIR="/scratch4/danielk/auzunog1/hf_cache"

source ~/.bashrc
source activate data_utility

models=(
    "facebook/opt-30b"
    "facebook/opt-13b"
    "facebook/opt-6.7b"
    "facebook/opt-2.7b"
    "facebook/opt-1.3b"
    "facebook/opt-350m"
    "facebook/opt-125m"
)

llama_models=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-11B-Vision"
)

tasks=(
    "data/aqua_rat.json"
    "data/gpqa.json"
    "data/logiqa.json"
    "data/pubmed_qa.json"
    "data/strategyqa.json"
)

perturbated_tasks=(
    "data/perturbated_aqua_rat.json"
    "data/perturbated_gpqa.json"
    "data/perturbated_logiqa.json"
    "data/perturbated_pubmed_qa.json"
    "data/perturbated_strategyqa.json"
)

tasks2=(
    "data/arc_challenge.json"
    "data/arc_easy.json"
    "data/art.json"
    "data/boolq.json"
    "data/commonsense_qa.json"
    "data/copa.json"
    "data/math_qa.json"
    "data/openbookqa.json"
    "data/quartz.json"
    "data/sciq.json"
    "data/truthful_qa.json"
    "data/wiqa.json"
)

perturbated_tasks2=(
    "data/perturbated_arc_challenge.json"
    "data/perturbated_arc_easy.json"
    "data/perturbated_art.json"
    "data/perturbated_boolq.json"
    "data/perturbated_commonsense_qa.json"
    "data/perturbated_copa.json"
    "data/perturbated_math_qa.json"
    "data/perturbated_openbookqa.json"
    "data/perturbated_quartz.json"
    "data/perturbated_sciq.json"
    "data/perturbated_truthful_qa.json"
    "data/perturbated_wiqa.json"
)

for task in "${perturbated_tasks[@]}"
do
    for model in "${models[@]}"
    do
        echo "Running collect_hs.py with --model_name $model and --dataset $task"
        srun python collect_hs.py --model_name "$model" --dataset "$task"
    done
done
