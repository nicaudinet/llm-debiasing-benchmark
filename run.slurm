#!/bin/bash

#SBATCH --job-name=dsl_use
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH -p normal # Queue (partition) name
#SBATCH -N 1 # Total number of nodes
#SBATCH -n 1 # Total number of mpi tasks
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

module reset
module load Rstats

apptainer exec r-ver_latest.sif Rscript test.R

