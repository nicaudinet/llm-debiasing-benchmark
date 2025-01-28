#!/bin/bash

result_dir="/Users/audinet/Projects/work/dsl-use/results/local/simulation"
plot_dir="$result_dir/plots/"

mkdir -p $plot_dir

echo "Running the experiment"
for i in $(seq -w 1 10);
do
    python3.12 experiment_simulation.py $result_dir/data_$i.npz
done

echo "Plotting the results"
python3.12 plot.py $result_dir $plot_dir
