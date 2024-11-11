#!/bin/bash

#SBATCH --job-name=dsl_use
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

module load Rstats

Rscript test.R

