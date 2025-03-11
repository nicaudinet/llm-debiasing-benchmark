#!/bin/bash

#SBATCH --job-name=annotate-amazon
#SBATCH --account=NAISS2025-22-180
#SBATCH --partition=alvis

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=0-01:00:00

#SBATCH --output=results/logs/output_%A_%a.log
#SBATCH --error=results/logs/error_%A_%a.log

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

module purge
module load Python/3.12.3-GCCcore-13.3.0

source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv_alvis/bin/activate

BASE_DIR="/cephyr/users/audinet/Alvis"

python3 "$BASE_DIR/dsl-use/lib/annotate_llama.py"
