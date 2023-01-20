# Create a table of summary statistics 

import pandas as pd
import numpy as np
import os
import sys
import json
import csv

batch_run_dir = '../batch_experiments'
summary_statistics = pd.DataFrame()

#get a list of directories, but not files, in the batch_run_dir
runs = []
for f in os.listdir(batch_run_dir):
    d = os.path.join(batch_run_dir, f)
    if os.path.isdir(d):
        # check if the directory has a "summary_metrics.json" file
        if os.path.isfile(os.path.join(d, 'summary_metrics.json')):
            runs.append(f)
    

def sci_notation(x):
    # convert a number to scientific notation, but only if the magnitude of power is greater than 2
    if x == 0:
        return '0'
    elif abs(x) < 0.01 or abs(x) > 100:
        return '{:.2e}'.format(x)
    else:
        return '{:.2f}'.format(x)


    
    

def percent_notation(x):
    # convert a number to a percentage
    return '{:.2f}'.format(x*100) + '%'


# list of metrics to always get
base_keys = {
    #'epoch',
    #'train_loss',
    #'train_acc',
    'test_loss': 'Test Loss',
    #'test_acc',
    'total_hidden_excluded_loss': 'Excluded Loss',
    'total_hidden_restricted_loss': 'Restricted Loss',
    'percent_logits_explained': 'Logit FVE',
    'percent_x_embed_explained': 'W_a FVE',
    'percent_y_embed_explained': 'W_b FVE',
    'percent_unembed_explained': 'W_U FVE',
    'percent_hidden_explained': 'MLP FVE',
}
    
    # create templates for key_rep dependent summary statistics
templates ={
    #'logit_excluded_loss_{}_rep',
    #'logit_restricted_loss_{}_rep',
    #'logit_{}_rep_trace_similarity': 'logit similarity'
}

# loop over runs
for run in runs:
    try:
        # create a new row in the dataframe
        row = {}
        row['run'] = run
        print(run)
        # get the "summary_metrics.json" file
        summary_metrics = os.path.join(batch_run_dir, run, 'summary_metrics.json')
        # read the json file
        run_metrics = json.load(open(summary_metrics, 'r'))

        # add the base keys to the dataframe
        for key in base_keys:
            if 'loss' in key:
                row[base_keys[key]] = sci_notation(run_metrics[key])
            elif 'percent' in key:
                row[base_keys[key]] = percent_notation(run_metrics[key])
        

        # get the key_reps in order from the text file
        key_reps = os.path.join(batch_run_dir, run, 'key_reps_in_order.txt')
        with open(key_reps, 'r') as f:
            key_reps = f.readlines()
        key_reps = [irrep.strip() for irrep in key_reps]

        key_reps_clean = [irrep.replace('freq_', '') for irrep in key_reps]
        key_reps_clean = [irrep.replace('a5_', '') for irrep in key_reps_clean]
        key_reps_clean = [irrep.replace('s5_', '') for irrep in key_reps_clean]
        key_reps_clean = [irrep.replace('s6_', '') for irrep in key_reps_clean]

        # add the key_reps to the dataframe
        row['Key Irreps'] = ', '.join(key_reps_clean)

        # add the key_rep dependent summary statistics to the dataframe
        for template in templates:
            entry = []
            for key_rep in key_reps:
                entry.append(sci_notation(run_metrics[template.format(key_rep)]))
            row[templates[template]] = ', '.join(entry)

        # add the row to the dataframe, index them in order
        summary_statistics = pd.concat([summary_statistics, pd.DataFrame.from_dict([row])], ignore_index=False)
        # sort by run name
        summary_statistics = summary_statistics.sort_values(by='run')
    except Exception as e:
        print('failed to parse {}'.format(run))
        print(e)
        continue
# save the dataframe to a csv, without the quotes around the strings
summary_statistics.to_csv('universality_results.csv', index=False)
