#!/bin/bash

#SBATCH --job-name=dsl-use
#SBATCH --account=C3SE2024-1-17
#SBATCH --partition=vera

#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-10

#SBATCH --time=3-00:00:00
#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=results/exp5/logs/output/output_%A_%a.log
#SBATCH --error=results/exp5/logs/error/error_%A_%a.log

module reset
module load R

srun apptainer exec container.sif Rscript dsl-use-data.R "results/exp5/data_${SLURM_ARRAY_TASK_ID}.RData"
