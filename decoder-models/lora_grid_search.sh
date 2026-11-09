#!/bin/bash
#SBATCH --job-name=lora_grid
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=grid_logs/output/%x_%A_%a.out
#SBATCH --error=grid_logs/error/%x_%A_%a.err
#SBATCH --array=0-44

# LoRA grid search: one array task per (model, learning rate, rank) configuration
# Each task trains for EPOCHS epochs and scores dev/test after every epoch,
# so epochs 1..EPOCHS are all covered by one run (files get an _e{epoch} suffix)
# Usage: sbatch lora_grid_search.sh

# Exit on error
set -e

module purge
module load Python/3.10.4-GCCcore-11.3.0

export HF_HOME=/scratch/$USER/hf_cache

source $HOME/venvs/LMA/bin/activate

set -a
source ../.ENV
set +a

# Defaults
RESULTS_DIR="predictions/lora"
TRAIN_FILE="../data/train.csv"
DEV_FILE="../data/dev.csv"
TEST_FILE="../data/test.csv"
BATCH_SIZE=8
EPOCHS=3

# Grid search params
MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B mistralai/Ministral-8B-Instruct-2410 meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it)
LEARNING_RATES=(1e-4 3e-4 5e-4)
LORA_RANKS=(8 16 32)

# Build the full grid, one "model|lr|rank" entry per configuration
CONFIGS=()
for model in "${MODELS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
            CONFIGS+=("${model}|${lr}|${rank}")
        done
    done
done

# Pick the configuration for this array task
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CONFIGS[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID out of range (grid has ${#CONFIGS[@]} configs)"
    exit 1
fi
IFS='|' read -r MODEL lr rank <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

config_str="lr${lr}_r${rank}"
output_dir="$RESULTS_DIR/${MODEL##*/}"
output_file="${output_dir}/${config_str}.csv"
mkdir -p "$output_dir"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID / $(( ${#CONFIGS[@]} - 1 ))"
echo "LoRA: ${MODEL##*/} | Epochs: 1-$EPOCHS | LR: $lr | Rank: $rank"
echo "Model: $MODEL"
echo "=========================================="

python3 lora.py \
    --model "$MODEL" \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --test "$TEST_FILE" \
    --output_file "$output_file" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$lr" \
    --lora_rank "$rank"

deactivate

echo "Successfully completed: ${MODEL##*/} with config: $config_str"
echo "Predictions saved in: ${output_dir}/${config_str}_e<epoch>.csv"
