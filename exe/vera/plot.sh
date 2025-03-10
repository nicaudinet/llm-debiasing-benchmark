#! /bin/bash

HOME_DIR="/cephyr/users/audinet/Vera/dsl-use"
BASE_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments"

module purge
module load matplotlib/3.7.2-gfbf-2023a

###########################
## Plots for vary expert ##
###########################

function plot_expert () {
    local DATASET=$1
    DATA_DIR="$BASE_DIR/vary-num-expert/$DATASET/data_deepseek"
    if [ -d $DATA_DIR ]; then
        echo -e "\nVary expert ($DATASET)"
        PLOT_DIR="$BASE_DIR/vary-num-expert/$DATASET/plot_deepseek"
        mkdir -p $PLOT_DIR
        python "$HOME_DIR/lib/vary_expert_plot.py" $DATA_DIR $PLOT_DIR
    else
        echo -e "\nVary num expert ($DATASET) data dir not found"
    fi
}

plot_expert "simulation"
plot_expert "amazon"
plot_expert "misinfo"
plot_expert "biobias"
plot_expert "germeval"

##########################
## Plots for vary total ##
##########################

function plot_total () {
    local DATASET=$1
    DATA_DIR="$BASE_DIR/vary-num-total/$DATASET/data_deepseek"
    if [ -d $DATA_DIR ]; then
        for n in 200 1000 5000; do
        echo -e "\nVary total n$n ($DATASET)"
        DATA_DIR_N="$DATA_DIR/n$n"
            PLOT_DIR="$BASE_DIR/vary-num-total/$DATASET/plot_deepseek"
            mkdir -p $PLOT_DIR
            python "$HOME_DIR/lib/vary_total_plot.py" $n $DATA_DIR_N $PLOT_DIR
        done
    else
        echo -e "\nVary num total ($DATASET) data dir not found"
    fi
}

plot_total "simulation"
plot_total "amazon"
plot_total "misinfo"
plot_total "biobias"
plot_total "germeval"
