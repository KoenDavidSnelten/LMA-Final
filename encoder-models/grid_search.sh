#!/bin/bash
#SBATCH --job-name=bert_grid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --array=0-143


# Exit on error and print commands
set -e
set -x

# Load modules
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Activate virtual environment
source $HOME/venvs/LMA/bin/activate

# Defaults
RESULTS_DIR="predictions"
TRAIN_FILE="../data/train.csv"
DEV_FILE="../data/dev.csv"
TEST_FILE="../data/test.csv"

# gridsearch param
EPOCHS=(2 3 4)
BATCH_SIZES=(16 32)
LEARNING_RATES=(1e-5 2e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.0 0.01)

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p $RESULTS_DIR

# Model configurations: (model_name, hf_model_id)
declare -a MODELS=(
    "bert_base|google-bert/bert-base-uncased"
    "roberta_base|FacebookAI/roberta-base"
    "modernbert_base|answerdotai/ModernBERT-base"
)

# Build the full grid, one "model|ep|bs|lr|wd" entry per configuration
CONFIGS=()
for model_config in "${MODELS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for wd in "${WEIGHT_DECAYS[@]}"; do
                    CONFIGS+=("${model_config}|${ep}|${bs}|${lr}|${wd}")
                done
            done
        done
    done
done

# Pick the configuration for this array task
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CONFIGS[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID out of range (grid has ${#CONFIGS[@]} configs)"
    exit 1
fi
IFS='|' read -r model_name hf_model ep bs lr wd <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID / $(( ${#CONFIGS[@]} - 1 ))"
echo "Running: $model_name | Epochs: $ep | BS: $bs | LR: $lr | WD: $wd"
echo "Model: $hf_model"
echo "=========================================="

# Create output directory
config_str="e${ep}_b${bs}_l${lr}_wd${wd}"
final_output_dir="$RESULTS_DIR/${model_name}"
mkdir -p "$final_output_dir"

output_file="${final_output_dir}/${config_str}.csv"

# Run the training script
python3 train.py \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --test "$TEST_FILE" \
    --output_file "$output_file" \
    --model "$hf_model" \
    --epochs "$ep" \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --weight_decay "$wd"

deactivate

echo "Successfully completed: $model_name with config: $config_str"
echo "Prediction saved in: $output_file"
