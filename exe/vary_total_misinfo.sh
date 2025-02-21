#!/bin/bash

original_data_path="/Users/audinet/Datasets/misinfo-general/data"
base_dir="/Users/audinet/Projects/work/dsl-use"
annotated_data_path="$base_dir/resources/misinfo_annotated.pkl"
result_dir="$base_dir/results/vary-num-total/misinfo"
data_dir="$result_dir/data"
plot_dir="$result_dir/plot"

mkdir -p $data_dir
mkdir -p $plot_dir

if [ ! -f $annotated_data_path ]; then
    python3.12 \
        "$base_dir/lib/annotate_misinfo.py" \
        $original_data_path \
        $annotated_data_path
fi

echo "Experiment: vary number of total samples (misinfo)"
num_expert=(200 1000 5000)
for n in "${num_expert[@]}"; do
    data_dir_n="$data_dir/n$n"
    mkdir -p $data_dir_n
    for i in $(seq -w 1 200); do
        echo "Starting run: n=$n i=$i"
        python3.12 \
            "$base_dir/lib/vary_total_realworld.py" \
            $annotated_data_path \
            "$data_dir_n/data_misinfo_$i.npz" \
            $n
    done
    python3.12 \
        "$base_dir/lib/vary_total_plot.py" \
        $data_dir_n \
        $plot_dir \
        $n
done
