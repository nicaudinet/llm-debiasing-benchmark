#! /bin/bash

BASE_DIR="/cephyr/users/audinet/Vera/dsl-use/"

module purge
module load matplotlib/3.7.2-gfbf-2023a

###########################
## Plots for vary expert ##
###########################

function plot_expert () {
    local DATASET=$1
    DATA_DIR="$BASE_DIR/results/vary-num-expert/$DATASET/data"
    if [ -d $DATA_DIR ]; then
        echo -e "\nVary expert ($DATASET)"
        PLOT_DIR="$BASE_DIR/results/vary-num-expert/$DATASET/plot"
        mkdir -p $PLOT_DIR
        python "$BASE_DIR/lib/vary_expert_plot.py" $DATA_DIR $PLOT_DIR
    else
        echo -e "\nVary num expert ($DATASET) data dir not found"
    fi
}

plot_expert "simulation"
plot_expert "amazon"
plot_expert "misinfo"
plot_expert "biobias"

##########################
## Plots for vary total ##
##########################

function plot_total () {
    local DATASET=$1
    DATA_DIR="$BASE_DIR/results/vary-num-total/$DATASET/data"
    if [ -d $DATA_DIR ]; then
        for n in 200 1000 5000; do
        echo -e "\nVary total n$n ($DATASET)"
        DATA_DIR_N="$DATA_DIR/n$n"
            PLOT_DIR="$BASE_DIR/results/vary-num-total/$DATASET/plot"
            mkdir -p $PLOT_DIR
            python "$BASE_DIR/lib/vary_total_plot.py" $DATA_DIR_N $PLOT_DIR $n
        done
    else
        echo -e "\nVary num total ($DATASET) data dir not found"
    fi
}

plot_total "simulation"
plot_total "amazon"
plot_total "misinfo"
plot_total "biobias"
