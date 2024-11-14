#!/bin/bash

#SBATCH --job-name=dsl-use
#SBATCH --account=C3SE2024-1-17
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=logs/output/output_%j.log
#SBATCH --error=logs/error/error_%j.log

module reset
module load R

apptainer exec container.sif Rscript dsl-use.R
