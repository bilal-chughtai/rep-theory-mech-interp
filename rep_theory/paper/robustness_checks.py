# Create a table of summary statistics 

import pandas as pd
import numpy as np
import os
import sys
import json

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
    # convert a number to scientific notation
    a = '{:.2e}'.format(x)
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


# list of metrics to always get
base_keys = [
    'epoch',
    'train_loss',
    'train_acc',
    'test_loss',
    'test_acc',
    'total_logit_excluded_loss',
    'total_logit_restricted_loss',
    'percent_logits_explained'
]
    
    # create templates for key_rep dependent summary statistics
templates =[
    'logit_excluded_loss_{}_rep',
    'logit_restricted_loss_{}_rep',
    'logit_{}_rep_trace_similarity'
]

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
            row[key] = sci_notation(run_metrics[key])
        

        # get the key_reps in order from the text file
        key_reps = os.path.join(batch_run_dir, run, 'key_reps.txt')
        with open(key_reps, 'r') as f:
            key_reps = f.readlines()
        key_reps = [irrep.strip() for irrep in key_reps]

        # add the key_reps to the dataframe
        row['key_reps'] = key_reps

        # add the key_rep dependent summary statistics to the dataframe
        for template in templates:
            entry = []
            for key_rep in key_reps:
                entry.append(sci_notation(run_metrics[template.format(key_rep)]))
            row[template] = entry

        # add the row to the dataframe, index them in order
        summary_statistics = pd.concat([summary_statistics, pd.DataFrame.from_dict([row])], ignore_index=False)
        # sort by run name
        summary_statistics = summary_statistics.sort_values(by='run')
    except:
        print('failed to parse {}'.format(run))
        continue
# save the dataframe to a csv
summary_statistics.to_csv('robustness_checks.csv')

