#!/bin/bash

# Create experiments
python create_batch_experiments.py

# Iterate through each line in the file
while read line; do
    # Run train.py with the line as an argument
    python train_with_checkpoints.py --task_dir batch_experiments/"$line"

    # Remove the line from the file and add it to the ran_experiments.txt file
    sed -i "/^$line$/d" batch_experiments/unran_experiments.txt
    echo "$line" >> batch_experiments/ran_experiments.txt

done < batch_experiments/unran_experiments.txt