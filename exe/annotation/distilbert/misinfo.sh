#!/bin/bash

set -eo pipefail

ANN_DIR="results/annotations/misinfo"

mkdir -p $ANN_DIR

echo ""
echo "#########"
echo "# Parse #"
echo "#########"
echo ""

python3 lib/parse_misinfo.py \
    "$ANN_DIR/original.parquet" \
    "$ANN_DIR/parsed.json"

echo ""
echo "############"
echo "# Annotate #"
echo "############"
echo ""

python3 lib/annotate_bert.py \
    "$ANN_DIR/parsed.json" \
    "$ANN_DIR/annotated.json"
