#!/bin/bash

ANN_DIR="results/annotations/amazon"

mkdir -p "$ANN_DIR/deepseek"

echo ""
echo "############"
echo "# Annotate #"
echo "############"
echo ""

# python3 lib/annotate_deepseek.py \
#     "amazon" \
#     "$ANN_DIR/parsed.json" \
#     "$ANN_DIR/deepseek" \
#     --num 10000

echo ""
echo "##########"
echo "# Gather #"
echo "##########"
echo ""

python3 lib/gather_deepseek.py \
    "amazon" \
    "$ANN_DIR/parsed.json" \
    "$ANN_DIR/deepseek" \
    "$ANN_DIR/annotated_deepseek.json"
