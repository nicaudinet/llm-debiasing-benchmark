#!/bin/bash

set -eo pipefail

ANN_DIR="results/annotations/amazon"

mkdir -p $ANN_DIR

echo ""
echo "#########"
echo "# Parse #"
echo "#########"
echo ""

python3 lib/parse_amazon.py \
    "$ANN_DIR/original.txt" \
    "$ANN_DIR/parsed.json"

echo ""
echo "############"
echo "# Annotate #"
echo "############"
echo ""

python3 lib/annotate_sentiment.py \
    "$ANN_DIR/parsed.json" \
    "$ANN_DIR/annotated.json"
