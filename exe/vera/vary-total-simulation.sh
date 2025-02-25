#!/bin/bash

#SBATCH --job-name=vary-total-simulation
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

base_dir="/cephyr/users/audinet/Vera/dsl-use"
result_dir="$base_dir/results/vary-num-total/simulation"
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
        "$base_dir/lib/vary_total_simulation.py" \
        "$data_dir_n/data_simulation_${SLURM_ARRAY_TASK_ID}.npz" \
        $n
done
