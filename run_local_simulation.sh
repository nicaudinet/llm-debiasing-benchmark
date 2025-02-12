#!/bin/bash

base_dir="/Users/audinet/Projects/work/dsl-use/results/local/vary-num-total/simulation"
data_dir="$base_dir/data"
plot_dir="$base_dir/plots"

mkdir -p $data_dir
mkdir -p $plot_dir

echo "Running the experiment"
num_expert_samples=(200 1000 5000)
for n in "${num_expert_samples[@]}"; do
    data_dir_n="$data_dir/n$n"
    mkdir -p $data_dir_n
    for i in $(seq -w 1 500); do
        echo "Starting n=$n i=$i"
        python3.12 experiment_simulation.py "$data_dir_n/data_$i.npz" $n
    done
    python3.12 plot.py $data_dir_n $plot_dir $n
done
