#!/bin/bash

#SBATCH --job-name=dsl-use-simulation
#SBATCH --account=C3SE2024-1-17
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-1000

#SBATCH --time=1-00:00:00
#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=results/run2/logs/output/output_%A_%a.log
#SBATCH --error=results/run2/logs/error/error_%A_%a.log

module reset
module load R

apptainer exec container.sif python experiment.py "results/run2/data_${SLURM_ARRAY_TASK_ID}.npz"
