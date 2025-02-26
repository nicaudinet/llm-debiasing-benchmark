#!/bin/bash

#SBATCH --job-name=vary-expert-simulation
#SBATCH --account=C3SE2025-1-14
#SBATCH --partition=vera

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=1-500
#SBATCH --time=0-00:15:00

#SBATCH --mail-user=nicolas.audinet@chalmers.se
#SBATCH --mail-type=all
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

base_dir="/cephyr/users/audinet/Vera/dsl-use"
result_dir="$base_dir/results/vary-num-expert/simulation"
data_dir="$result_dir/data"
plot_dir="$result_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

python \
    "$base_dir/lib/vary_expert_simulation.py" \
    "$data_dir/data_simulation_${SLURM_ARRAY_TASK_ID}.npz"

# python3.12 "$base_dir/lib/vary_expert_plot.py" $data_dir $plot_dir
