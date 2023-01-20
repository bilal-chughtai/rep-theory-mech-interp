#!/bin/bash

# Iterate through each line in the file
while read line; do
    # Run eval.py with the line as an argument
    python eval.py --task_dir batch_experiments/"$line" --final

    # Add the line to evaled_experiments.txt
    echo "$line" >> batch_experiments/evaled_experiments.txt

    # Remove the line from ran_experiments.txt
    sed -i "/$line/d" batch_experiments/ran_experiments.txt

done < batch_experiments/ran_experiments.txt