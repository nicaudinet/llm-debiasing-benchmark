#! /bin/bash

MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments"

echo ""
echo "-------------------------"
echo "-- Vary Expert Samples --"
echo "-------------------------"
echo ""

function scp_plot_expert () {
    DATASET=$1
    echo ""
    echo "Copying plots from $DATASET"
    RESULT_DIR="results/vera/vary-num-expert/$DATASET"
    mkdir -p $RESULT_DIR
    scp -r vera:$MIMER_PATH/vary-num-expert/$DATASET/plot_deepseek $RESULT_DIR
}

scp_plot_expert "simulation"
scp_plot_expert "amazon"
scp_plot_expert "misinfo"
scp_plot_expert "biobias"
scp_plot_expert "germeval"

echo ""
echo "------------------------"
echo "-- Vary Total Samples --"
echo "------------------------"
echo ""

function scp_plot_total () {
    DATASET=$1
    echo ""
    echo "Copying plots from $DATASET"
    RESULT_DIR="results/vera/vary-num-total/$DATASET"
    mkdir -p $RESULT_DIR
    scp -r vera:$MIMER_PATH/vary-num-total/$DATASET/plot_deepseek $RESULT_DIR
}

scp_plot_total "simulation"
scp_plot_total "amazon"
scp_plot_total "misinfo"
scp_plot_total "biobias"
scp_plot_total "germeval"
