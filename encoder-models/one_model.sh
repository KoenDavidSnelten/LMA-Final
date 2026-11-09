#!/bin/bash
#SBATCH --job-name=one_model
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/output/%x_%j.out
#SBATCH --error=logs/error/%x_%j.err

module purge
module load Python/3.10.4-GCCcore-11.3.0

source $HOME/venvs/LMA/bin/activate

MODEL=google-bert/bert-base-uncased
EPOCHS=3
BATCH_SIZE=32
LR=2e-5
WD=0.01

echo "=========================================="
echo "Running: ${MODEL##*/} | Epochs: $EPOCHS | BS: $BATCH_SIZE | LR: $LR | WD: $WD"
echo "Model: $MODEL"
echo "=========================================="

python3 train.py --test \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --weight_decay $WD \
    --output_file predictions/${MODEL##*/}_e${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_wd${WD}.csv

deactivate