#!/bin/bash

# --- Phase 1: Training ---
# Training each individual domain expert
datasets=("imagenet" "caltech101" "food101" "eurosat" "dtd" "flowers102")

echo "Starting Training Phase..."
for ds in "${datasets[@]}"
do
    echo "Training expert for: $ds"
    python mda_train.py --dataset "$ds" --geo_layer
done

echo "------------------------------------------------"

# --- Phase 2: Evaluation of Knowledge Stacks ---
# (1) Easy: i -> c -> fo -> e (Coarse-to-fine)
# (2) Medium: i -> fo -> e -> d (Increasing structural complexity)
# (3) Hard: i -> e -> d -> f (High-contrast domain shift)

stacks=(
    "i->c->fo->e"
    "i->fo->e->d"
    "i->e->d->f"
)

echo "Starting Evaluation Phase..."
for stack_str in "${stacks[@]}"
do
    echo "Evaluating Stack: $stack_str"
    # Calling your eval script with the -s flag
    python mda_eval.py -s "$stack_str" --geo_layer
done

echo "MDA Reproduction Complete."