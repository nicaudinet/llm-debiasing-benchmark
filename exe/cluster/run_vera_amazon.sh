#!/bin/bash

#SBATCH --job-name=dsl-use-amazon
#SBATCH --account=C3SE2024-1-17
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-1000

#SBATCH --time=1-00:00:00
#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=results/amazon/logs/output/output_%A_%a.log
#SBATCH --error=results/amazon/logs/error/error_%A_%a.log

module reset
module load R

mkdir -p "results/amazon"

apptainer exec container.sif python experiment_amazon.py "amazon_annotated.pkl" "results/amazon/data_${SLURM_ARRAY_TASK_ID}.npz"
