#!/bin/bash
#SBATCH --job-name=collect_results
#SBATCH --time=00:05:00
#SBATCH --partition=regular
#SBATCH --mem=1G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load Python/3.10.4-GCCcore-11.3.0

source $HOME/venvs/LMA/bin/activate

python3 collect_results.py \
    --logs "logs/bert_grid_*.out" \
    --output_file results/grid_results.csv

deactivate
