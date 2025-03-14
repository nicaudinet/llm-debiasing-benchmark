#! /bin/bash

MIMER_PATH="/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments"

echo ""
echo "-------------------------"
echo "-- Vary Expert Samples --"
echo "-------------------------"
echo ""

function scp_plot_expert () {
    ANNOTATION=$1
    DATASET=$2
    echo ""
    echo "Copying plots from $DATASET"
    RESULT_DIR="results/vera/vary-num-expert/$DATASET/$ANNOTATION"
    mkdir -p $RESULT_DIR
    scp -r vera:$MIMER_PATH/vary-num-expert/$DATASET/plot/$ANNOTATION $RESULT_DIR
}

scp_plot_expert "" "simulation"

scp_plot_expert "bert" "amazon"
scp_plot_expert "bert" "misinfo"
scp_plot_expert "bert" "biobias"
scp_plot_expert "bert" "germeval"

scp_plot_expert "deepseek" "amazon"
scp_plot_expert "deepseek" "misinfo"
scp_plot_expert "deepseek" "biobias"
scp_plot_expert "deepseek" "germeval"

scp_plot_expert "phi4" "amazon"
scp_plot_expert "phi4" "misinfo"
scp_plot_expert "phi4" "biobias"
scp_plot_expert "phi4" "germeval"

echo ""
echo "------------------------"
echo "-- Vary Total Samples --"
echo "------------------------"
echo ""

function scp_plot_total () {
    ANNOTATION=$1
    DATASET=$2
    echo ""
    echo "Copying plots from $DATASET"
    RESULT_DIR="results/vera/vary-num-total/$DATASET/$ANNOTATION"
    mkdir -p $RESULT_DIR
    scp -r vera:$MIMER_PATH/vary-num-total/$DATASET/plot/$ANNOTATION $RESULT_DIR
}

scp_plot_total "" "simulation"

scp_plot_total "bert" "amazon"
scp_plot_total "bert" "misinfo"
scp_plot_total "bert" "biobias"
scp_plot_total "bert" "germeval"

scp_plot_total "deepseek" "amazon"
scp_plot_total "deepseek" "misinfo"
scp_plot_total "deepseek" "biobias"
scp_plot_total "deepseek" "germeval"

scp_plot_total "phi4" "amazon"
scp_plot_total "phi4" "misinfo"
scp_plot_total "phi4" "biobias"
scp_plot_total "phi4" "germeval"
