#! /bin/bash

for i in {1..10}
do
    python lib/vary_expert_realworld.py \
        "results/annotations/misinfo/annotated_deepseek.json" \
        "results/linear/misinfo/deepseek/data/data_$i.npz" \
        "linear" \
        --seed $i \
        --centered
done

python lib/vary_expert_plot.py \
    "results/linear/misinfo/deepseek/data/" \
    "results/linear/misinfo/deepseek/plot/" \
