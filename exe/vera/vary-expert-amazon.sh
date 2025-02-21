#!/bin/bash

#SBATCH --job-name=vary-expert-amazon
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --open-mode=append
#SBATCH --output=/dev/null
#SBATCH --error=job_%A_%a_failed.err

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

annotated_reviews="/cephyr/users/audinet/Vera/datasets/amazon/annotated_reviews.pkl"
base_dir="/cephyr/users/audinet/Vera/dsl-use"
result_dir="$base_dir/results/vary-num-expert/amazon"
data_dir="$result_dir/data"
plot_dir="$result_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

{

    python \
        "$base_dir/lib/vary_expert_realworld.py" \
        $annotated_reviews \
        "$data_dir/data_amazon_${SLURM_ARRAY_TASK_ID}.npz"

    error_log="job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_failed.err" 
    exit_code=$?

    if [$exit_code -eq 0 ]; then
        rm -f $error_log 2>/dev/null
    else
        echo "Job failed with exit code: $exit_code" >&2
        echo "Node: $(hostname)" >&2
        echo "Time: $(date)" >&2
    fi

    exit $exit_code
}

> /dev/null 2> $error_log
