#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/output/%x_%j.out
#SBATCH --error=logs/error/%x_%j.err

# LoRA finetuning for a single model (the prompt is defined in lora.py)
# Usage: sbatch lora.sh [MODEL]
#   e.g. sbatch lora.sh Qwen/Qwen3.5-9B

module purge
module load Python/3.10.4-GCCcore-11.3.0

export HF_HOME=/scratch/$USER/hf_cache

source $HOME/venvs/LMA/bin/activate

set -a
source .env
set +a

MODEL=${1:-Qwen/Qwen3.5-0.8B}

echo "=========================================="
echo "LoRA: ${MODEL##*/}"
echo "Model: $MODEL"
echo "=========================================="

python3 lora.py \
    --model $MODEL \
    --output_file predictions/lora_${MODEL##*/}.csv \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --lora_rank 16

deactivate
