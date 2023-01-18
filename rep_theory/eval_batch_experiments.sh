#!/bin/bash

# Iterate through each line in the file
while read line; do
    # Run eval.py with the line as an argument
    python eval.py --task_dir batch_experiments/"$line"
    python make_figures.py --task_dir batch_experiments/"$line"
    python determine_key_rep_order.py --task_dir batch_experiments/"$line"

    # Add the line to evaled_experiments.txt
    echo "$line" >> batch_experiments/evaled_experiments.txt

    # Remove the line from ran_experiments.txt
    sed -i "/$line/d" batch_experiments/ran_experiments.txt

done < batch_experiments/ran_experiments.txt