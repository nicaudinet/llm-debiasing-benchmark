#!/bin/bash

set -eo pipefail

if [ $# -eq 0 ]; then
    echo "Error: missing DATSET argument"
    exit 1
elif [ $# -gt 1 ]; then
    echo "Error: too many arguments. Expecting only one argument"
    exit 1
fi

DATASET=$1

ANN_DIR="results/annotations/$DATASET"
mkdir -p "$ANN_DIR/deepseek"

echo ""
echo "############"
echo "# Annotate #"
echo "############"
echo ""

python3 lib/annotate_deepseek.py \
    "$DATASET" \
    "$ANN_DIR/parsed.json" \
    "$ANN_DIR/deepseek" \
    --num 10000

echo ""
echo "##########"
echo "# Gather #"
echo "##########"
echo ""

python3 lib/gather_deepseek.py \
    "$DATASET" \
    "$ANN_DIR/parsed.json" \
    "$ANN_DIR/deepseek" \
    "$ANN_DIR/annotated_deepseek.json"
