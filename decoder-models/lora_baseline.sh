#!/bin/bash
#SBATCH --job-name=lora_baseline
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/output/%x_%j.out
#SBATCH --error=logs/error/%x_%j.err

# Zero-shot baseline (no LoRA training) for every model in the LoRA grid search
# Usage: sbatch lora_baseline.sh

module purge
module load Python/3.10.4-GCCcore-11.3.0

export HF_HOME=/scratch/$USER/hf_cache

source $HOME/venvs/LMA/bin/activate

set -a
source ../.ENV
set +a

# Defaults
RESULTS_DIR="predictions/lora_baseline"
DEV_FILE="../data/dev.csv"
TEST_FILE="../data/test.csv"
BATCH_SIZE=8

# Same models as lora_grid_search.sh
MODELS=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B mistralai/Ministral-8B-Instruct-2410 meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E4B-it)

mkdir -p "$RESULTS_DIR"

# Keep going if one model fails, but remember it for the summary
FAILED=()
for MODEL in "${MODELS[@]}"; do
    output_file="$RESULTS_DIR/${MODEL##*/}.csv"

    echo "=========================================="
    echo "Baseline: ${MODEL##*/}"
    echo "Model: $MODEL"
    echo "=========================================="

    if python3 lora_baseline.py \
        --model "$MODEL" \
        --dev_file "$DEV_FILE" \
        --test "$TEST_FILE" \
        --output_file "$output_file" \
        --batch_size "$BATCH_SIZE"; then
        echo "Predictions saved in: $output_file"
    else
        echo "FAILED: $MODEL"
        FAILED+=("$MODEL")
    fi
done

deactivate

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed models: ${FAILED[*]}"
    exit 1
fi
echo "Successfully completed all baselines"
