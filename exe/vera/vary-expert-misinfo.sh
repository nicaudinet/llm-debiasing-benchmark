#!/bin/bash

#SBATCH --job-name=vary-expert-misinfo
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=results/logs/output_%A_%a.log
#SBATCH --error=results/logs/error_%A_%a.log

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

annotated="/cephyr/users/audinet/Vera/datasets/misinfo/annotated.pkl"
base_dir="/cephyr/users/audinet/Vera/dsl-use"
result_dir="$base_dir/results/vary-num-expert/misinfo"
data_dir="$result_dir/data"
plot_dir="$result_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

python \
    "$base_dir/lib/vary_expert_realworld.py" \
    $annotated \
    "$data_dir/data_misinfo_${SLURM_ARRAY_TASK_ID}.npz"

# python3.12 "$base_dir/lib/vary_expert_plot.py" $data_dir $plot_dir
