#! /bin/bash

function scp_plot_expert () {
    DATASET=$1
    RESULT_DIR="results/vera/vary-num-expert/$DATASET"
    mkdir -p $RESULT_DIR
    scp -r vera:~/dsl-use/results/vary-num-expert/$DATASET/plot $RESULT_DIR
}

scp_plot_expert "simulation"
scp_plot_expert "amazon"
scp_plot_expert "misinfo"
scp_plot_expert "biobias"
scp_plot_expert "germeval"

function scp_plot_total () {
    DATASET=$1
    RESULT_DIR="results/vera/vary-num-total/$DATASET"
    mkdir -p $RESULT_DIR
    scp -r vera:~/dsl-use/results/vary-num-total/$DATASET/plot $RESULT_DIR
}

scp_plot_total "simulation"
scp_plot_total "amazon"
scp_plot_total "misinfo"
scp_plot_total "biobias"
scp_plot_total "germeval"
