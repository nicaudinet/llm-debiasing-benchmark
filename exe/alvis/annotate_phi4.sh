#!/bin/bash

#SBATCH --job-name=annotate-amazon
#SBATCH --account=NAISS2025-22-180
#SBATCH --partition=alvis

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=0-03:00:00

#SBATCH --output=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/phi4_10ex/output_%A_%a.log
#SBATCH --error=/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/logs/phi4_10ex/error_%A_%a.log

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

if [ $# -ne 2 ]; then
    echo "Error: expecting exactly two arguments"
    exit 1
fi

DATASET=$1
NUM_EXPERT=$2

module purge
module load Python/3.12.3-GCCcore-13.3.0

source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv_alvis/bin/activate

BASE_DIR="/cephyr/users/audinet/Alvis/dsl-use"
MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/"
ANN_DIR="$MIMER_PATH/annotations/$DATASET/phi4_10ex"

mkdir -p $ANN_DIR

python3 "$BASE_DIR/lib/annotate_alvis.py" \
	"$DATASET" \
	"$MIMER_PATH/annotations/$DATASET/parsed.json" \
	"$ANN_DIR" \
	--model "microsoft/phi-4" \
	--num 10000 \
	--num_examples "$NUM_EXPERT"
