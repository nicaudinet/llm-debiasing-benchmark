#! /bin/bash

set -eo pipefail

HOME_DIR="/cephyr/users/audinet/Vera/dsl-use"
BASE_DIR="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments"

module purge
module load matplotlib/3.7.2-gfbf-2023a

###########################
## Plots for vary expert ##
###########################

function plot_expert () {
    local ANNOTATION=$1
    local DATASET=$2
    DATA_DIR="$BASE_DIR/vary-num-expert/$DATASET/data/$ANNOTATION"
    if [ -d $DATA_DIR ]; then
        echo -e "\nVary expert ($DATASET)"
        PLOT_DIR="$BASE_DIR/vary-num-expert/$DATASET/plot/$ANNOTATION"
        mkdir -p $PLOT_DIR
        python "$HOME_DIR/lib/vary_expert_plot.py" $DATA_DIR $PLOT_DIR
    else
        echo -e "\nVary num expert ($DATASET) data dir not found"
    fi
}

plot_expert "bert" "simulation"
plot_expert "bert" "amazon"
plot_expert "bert" "misinfo"
plot_expert "bert" "biobias"
plot_expert "bert" "germeval"

plot_expert "deepseek" "simulation"
plot_expert "deepseek" "amazon"
plot_expert "deepseek" "misinfo"
plot_expert "deepseek" "biobias"
plot_expert "deepseek" "germeval"

##########################
## Plots for vary total ##
##########################

function plot_total () {
    local ANNOTATION=$1
    local DATASET=$2
    DATA_DIR="$BASE_DIR/vary-num-total/$DATASET/data/$ANNOTATION"
    if [ -d $DATA_DIR ]; then
        for n in 200 1000 5000; do
        echo -e "\nVary total n$n ($DATASET)"
        DATA_DIR_N="$DATA_DIR/n$n"
            PLOT_DIR="$BASE_DIR/vary-num-total/$DATASET/plot/$ANNOTATION"
            mkdir -p $PLOT_DIR
            python "$HOME_DIR/lib/vary_total_plot.py" $n $DATA_DIR_N $PLOT_DIR
        done
    else
        echo -e "\nVary num total ($DATASET) data dir not found"
    fi
}

plot_total "bert" "simulation"
plot_total "bert" "amazon"
plot_total "bert" "misinfo"
plot_total "bert" "biobias"
plot_total "bert" "germeval"

plot_total "deepseek" "simulation"
plot_total "deepseek" "amazon"
plot_total "deepseek" "misinfo"
plot_total "deepseek" "biobias"
plot_total "deepseek" "germeval"
