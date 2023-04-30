#!/bin/bash

# Evaluate only the final model for each experiment, and not the checkpoints.
while read line; do
    # Run eval.py with the line as an argument
    python eval.py --task_dir batch_experiments/"$line" --final

    # Add the line to evaled_experiments.txt
    echo "$line" >> batch_experiments/evaled_experiments.txt

    # Remove the line from experiments_to_eval.txt
    sed -i "/$line/d" batch_experiments/experiments_to_eval.txt

done < batch_experiments/experiments_to_eval.txt