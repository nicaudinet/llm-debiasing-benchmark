#!/bin/bash

original_reviews_path="/Users/audinet/Datasets/amazon_reviews/original_reviews.txt"
annotated_reviews_path="/Users/audinet/Projects/work/dsl-use/resources/amazon_annotated.pkl"
result_dir="/Users/audinet/Projects/work/dsl-use/results/local/amazon"
plot_dir="$result_dir/plots/"

mkdir -p $plot_dir

echo "Annotating the reviews"
python3.12 annotate_amazon.py $original_reviews_path $annotated_reviews_path

echo "Running the experiment"
for i in $(seq -w 1 200);
do
    python3.12 experiment_amazon.py $annotated_reviews_path $result_dir/data_amazon_$i.npz
done

echo "Plotting the results"
python3.12 plot.py $result_dir $plot_dir
