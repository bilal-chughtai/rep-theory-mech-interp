#!/bin/bash

# Iterate through each line in the file
while read line; do

    python eval.py --task_dir batch_experiments/"$line"
    python make_figures.py --task_dir batch_experiments/"$line"
    python determine_key_rep_order.py --task_dir batch_experiments/"$line"

    # Add the line to evaled_experiments.txt
    echo "$line" >> batch_experiments/evaled_experiments.txt

    # Remove the line from experiments_to_eval.txt
    sed -i "/$line/d" batch_experiments/experiments_to_eval.txt

done < batch_experiments/experiments_to_eval.txt