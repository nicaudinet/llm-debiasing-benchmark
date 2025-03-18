#!/bin/bash

#SBATCH --job-name=vary-expert
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/vary-expert/output_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/vary-expert/error_%A_%a.log

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

set -eo pipefail

ANNOTATION=$1
DATASET=$2

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

source venv/bin/activate

BASE_DIR="/cephyr/users/audinet/Vera/dsl-use/"
MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/"
DATA_DIR="$MIMER_PATH/experiments/vary-num-expert/$DATASET/data/$ANNOTATION"

mkdir -p $DATA_DIR

python "$BASE_DIR/lib/vary_expert_realworld.py" \
    "$MIMER_PATH/annotations/$DATASET/annotated_$ANNOTATION.json" \
    "$DATA_DIR/data_${SLURM_ARRAY_TASK_ID}.npz" \
    --seed "${SLURM_ARRAY_TASK_ID}"
