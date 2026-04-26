#!/bin/bash

# --- Phase 1: Training the CIL Sequence ---
# In CIL, we train a sequence of tasks (e.g., 4, 5, or 10 tasks)
# to evaluate long-term stability and catastrophic forgetting.

# Define the number of tasks for the benchmark (e.g., 10 tasks)
NUM_TASKS=4

echo "Starting Class-Incremental Training (Tasks: $NUM_TASKS)..."

# This will train all tasks in sequence and save individual GeoLayer checkpoints
python cil_train.py --num_tasks $NUM_TASKS --geo_layer

echo "------------------------------------------------"

# --- Phase 2: Evaluation ---

# 1. Final Accuracy: Measures performance on all classes after the final task
echo "Evaluating Final Incremental Accuracy..."
python cil_eval.py --num_tasks $NUM_TASKS --geo_layer

# 2. Catastrophic Forgetting: Measures retention of Task-0 after 10 tasks
# This uses the --forgetting flag to specifically check Task-0 accuracy decay
echo "Measuring Catastrophic Forgetting (Retention of Task-0)..."
python cil_eval.py --num_tasks $NUM_TASKS --geo_layer --forgetting

echo "CIL Reproduction Complete."