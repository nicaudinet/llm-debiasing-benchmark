#!/bin/bash

original_data_path="/Users/audinet/Datasets/misinfo-general/data"
base_dir="/Users/audinet/Projects/work/dsl-use"
annotated_data_path="$base_dir/resources/misinfo_annotated.pkl"
result_dir="$base_dir/results/vary-num-expert/misinfo"
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

for i in $(seq -w 1 200); do
    python3.12 \
        "$base_dir/lib/vary_expert_realworld.py" \
        $annotated_data_path \
        "$data_dir/data_misinfo_$i.npz"
done

python3.12 "$base_dir/lib/vary_expert_plot.py" $data_dir $plot_dir
