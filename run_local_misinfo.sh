#!/bin/bash

data_path="/Users/audinet/Datasets/misinfo-general/data/"
annotated_data_path="/Users/audinet/Projects/work/dsl-use/resources/misinfo_annotated.pkl"
result_base_path="/Users/audinet/Projects/work/dsl-use/results/local/vary-num-expert/misinfo"
result_data_path="$result_base_path/data"
result_plot_path="$result_base_path/plot"

mkdir -p $result_data_path
mkdir -p $result_plot_path

echo "Annotating the reviews"
python3.12 annotate_misinfo.py $data_path $annotated_data_path

echo "Running the experiment"
for i in $(seq -w 1 500);
do
    output="$result_data_path/data_misinfo_$i.npz"
    python3.12 experiment_realworld.py $annotated_data_path $output
done

echo "Plotting the results"
python3.12 plot.py $result_data_path $result_plot_path
