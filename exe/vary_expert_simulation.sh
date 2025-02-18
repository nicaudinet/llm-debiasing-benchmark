#!/bin/bash

base_dir="/Users/audinet/Projects/work/dsl-use"
result_dir="$base_dir/results/vary-num-expert/simulation"
data_dir="$result_dir/data"
plot_dir="$result_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

for i in $(seq -w 1 2); do
    echo "Starting run: i=$i"
    python3.12 \
        "$base_dir/lib/vary_expert_simulation.py" \
        "$data_dir/data_simulation_$i.npz"
done

python3.12 "$base_dir/lib/vary_expert_plot.py" $data_dir $plot_dir
