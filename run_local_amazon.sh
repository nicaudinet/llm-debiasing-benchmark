#!/bin/bash

data_path="/Users/audinet/Datasets/amazon_reviews/original_reviews.txt"
annotated_data_path="/Users/audinet/Projects/work/dsl-use/resources/amazon_annotated.pkl"
result_base_path="/Users/audinet/Projects/work/dsl-use/results/local/vary-num-expert/amazon"
result_data_path="$result_base_path/data"
result_plot_path="$result_base_path/plot"

mkdir -p $plot_dir

echo "Annotating the reviews"
python3.12 annotate_amazon.py $data_path $annotated_data_path

echo "Running the experiment"
for i in $(seq -w 1 200);
do
    output="$result_data_path/data_amazon_$i.npz"
    python3.12 experiment_realworld.py $annotated_data_path $output
done

echo "Plotting the results"
python3.12 plot.py $result_data_path $result_plot_path
