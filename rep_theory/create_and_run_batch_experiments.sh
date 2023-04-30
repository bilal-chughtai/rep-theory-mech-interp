#!/bin/bash

# Set up all experiments
python create_batch_experiments.py

# Iterate through each line in the file
while read line; do
    # Run train.py with the line as an argument
    python train_with_checkpoints.py --task_dir batch_experiments/"$line"

    # Remove the line from the file and add it to the experiments_to_eval.txt file
    sed -i "/^$line$/d" batch_experiments/experiments_to_run.txt
    echo "$line" >> batch_experiments/experiments_to_eval.txt

done < batch_experiments/experiments_to_run.txt