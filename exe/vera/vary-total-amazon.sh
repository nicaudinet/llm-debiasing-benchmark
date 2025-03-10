#!/bin/bash

#SBATCH --job-name=vary-total-amazon
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:30:00

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

source venv/bin/activate

BASE_DIR="/cephyr/users/audinet/Vera/dsl-use/"
MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/"
DATA_DIR="$MIMER_PATH/experiments/vary-num-total/amazon/data"

mkdir -p $DATA_DIR

echo "Experiment: vary number of total samples"
num_expert=(200 1000 5000)
for n in "${num_expert[@]}"; do
    DATA_DIR_N="$DATA_DIR/n$n"
    mkdir -p $DATA_DIR_N
    python3 "$BASE_DIR/lib/vary_total_realworld.py" \
        "$n" \
	    "$MIMER_PATH/annotations/amazon/annotated_bert.json" \
        "$DATA_DIR_N/data_amazon_${SLURM_ARRAY_TASK_ID}.npz" \
        --seed "${SLURM_ARRAY_TASK_ID}"
done
