#!/bin/bash

# Iterate through each line in the file
while read line; do
    # Run eval.py with the line as an argument
    python determine_key_rep_order.py --task_dir batch_experiments/"$line"

done < batch_experiments/evaled_experiments.txt