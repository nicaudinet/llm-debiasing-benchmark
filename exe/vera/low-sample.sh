#!/bin/bash

#SBATCH --job-name=low-sample
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-1
#SBATCH --time=0-00:15:00

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

source venv/bin/activate

DATA_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/low-sample/simulation/data"

mkdir -p $DATA_DIR

python "/cephyr/users/audinet/Vera/dsl-use/lib/low-sample.py" \
    "$DATA_DIR/data_simulation_${SLURM_ARRAY_TASK_ID}.npz"
