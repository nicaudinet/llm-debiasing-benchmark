#!/bin/bash

#SBATCH --job-name=vary-total-misinfo
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

annotated="/cephyr/users/audinet/Vera/datasets/misinfo/annotated.pkl"
base_dir="/cephyr/users/audinet/Vera/dsl-use"
result_dir="$base_dir/results/vary-num-total/misinfo"
data_dir="$result_dir/data"
plot_dir="$result_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

echo "Experiment: vary number of total samples (simulation)"
num_expert=(200 1000 5000)
for n in "${num_expert[@]}"; do
    data_dir_n="$data_dir/n$n"
    mkdir -p $data_dir_n
    python3 \
        "$base_dir/lib/vary_total_realworld.py" \
        "$annotated" \
        "$data_dir_n/data_misinfo_${SLURM_ARRAY_TASK_ID}.npz" \
        "$n" \
        "${SLURM_ARRAY_TASK_ID}"
done
