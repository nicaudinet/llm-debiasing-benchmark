#!/bin/bash

base_dir="/Users/audinet/Projects/work/dsl-use"
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
    for i in $(seq -w 1 200); do
        echo "Starting run: n=$n i=$i"
        python3.12 \
            "$base_dir/lib/vary_total_simulation.py" \
            "$data_dir_n/data_simulation_$i.npz" \
            $n
    done
    python3.12 \
        "$base_dir/lib/vary_total_plot.py" \
        $data_dir_n \
        $plot_dir \
        $n
done
