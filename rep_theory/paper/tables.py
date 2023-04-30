# Create a table of summary statistics 

import pandas as pd
import numpy as np
import os
import sys
import json
import csv

batch_run_dir = '../batch_experiments'
summary_statistics = pd.DataFrame()

num_seeds = 4
#get a list of directories, but not files, in the batch_run_dir
runs = []
for f in os.listdir(batch_run_dir):
    d = os.path.join(batch_run_dir, f)
    if os.path.isdir(d):
        # check if the directory has a "summary_metrics.json" file
        if os.path.isfile(os.path.join(d, 'summary_metrics.json')):
            # make sure the seed, which is in f and of form seedx has x<=num_seeds
            seed = int(f.split('seed')[-1])
            if seed <= num_seeds:
                # make sure its not a width experiment
                if 'hidden' not in f:
                    runs.append(f)


# list of metrics to always get
base_keys = {
    #'epoch',
    #'train_loss',
    #'train_acc',
    'test_loss': 'Test Loss',
    #'test_acc',
    'total_hidden_excluded_loss': 'Exc. Loss',
    'total_hidden_restricted_loss': 'Res. Loss',
    'percent_logits_explained': 'Logit FVE',
    'percent_x_embed_explained': '$W_a$ FVE',
    'percent_y_embed_explained': '$W_b$ FVE',
    'percent_unembed_explained': '$W_U$ FVE',
    'percent_hidden_explained': 'MLP FVE',
    'percent_hidden_xy_explained': 'MLP $\\rho(ab)$ FVE',
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
        # read the cfg file
        # cfg = os.path.join(batch_run_dir, run, 'cfg.json')
        # cfg = json.load(open(cfg, 'r'))
        # # add the layers['hidden'] to the row
        # row['Hidden'] = cfg['layers']['hidden_dim']

        # add the base keys to the dataframe
        for key in base_keys:
            if 'loss' in key:
                row[base_keys[key]] = run_metrics[key]
            elif 'percent' in key:
                row[base_keys[key]] = run_metrics[key]
        

        # get the key_reps in order from the text file
        key_reps = os.path.join(batch_run_dir, run, 'key_reps_in_order.txt')
        with open(key_reps, 'r') as f:
            key_reps = f.readlines()
        key_reps = [irrep.strip() for irrep in key_reps]

        key_reps_clean = [irrep.replace('freq_', '') for irrep in key_reps]
        key_reps_clean = [irrep.replace('a5_', '') for irrep in key_reps_clean]
        key_reps_clean = [irrep.replace('s5_', '') for irrep in key_reps_clean]
        key_reps_clean = [irrep.replace('s6_', '') for irrep in key_reps_clean]

        # replace any underscores with a dash
        key_reps_clean = [irrep.replace('_', '-') for irrep in key_reps_clean]

        # add the key_reps to the dataframe
        row['Key Irreps'] = ', '.join(key_reps_clean)

        # add the key_rep dependent summary statistics to the dataframe
        for template in templates:
            entry = []
            for key_rep in key_reps:
                entry.append(run_metrics[template.format(key_rep)])
            row[templates[template]] = ', '.join(entry)

        # add the row to the dataframe, index them in order
        summary_statistics = pd.concat([summary_statistics, pd.DataFrame.from_dict([row])], ignore_index=False)
        # sort by run name
        summary_statistics = summary_statistics.sort_values(by='run')
    except Exception as e:
        print('failed to parse {}'.format(run))
        print(e)
        continue



# split the Seed out of the run name, following the last underscore
summary_statistics['Seed'] = summary_statistics['run'].str.split('_').str[-1]
# sort by Seed
summary_statistics = summary_statistics.sort_values(by=['run', 'Seed'])
# remove the Seed from the Seed
summary_statistics['Seed'] = summary_statistics['Seed'].str.replace('seed', '')

# split the dataframe into two dataframes, one for MLP and one for Transformer architectures
mlp = summary_statistics[summary_statistics['run'].str.contains('MLP')].copy()
transformer = summary_statistics[summary_statistics['run'].str.contains('T')].copy()

# extract the text before the first underscore
mlp['Group'] = mlp['run'].str.split('_').str[0]
transformer['Group'] = transformer['run'].str.split('_').str[0]

# turn this into latex by adding an underscore between the letter and number
mlp['Group'] = mlp['Group'].str.replace('C', 'C_')
mlp['Group'] = mlp['Group'].str.replace('D', 'D_')
mlp['Group'] = mlp['Group'].str.replace('S', 'S_')
mlp['Group'] = mlp['Group'].str.replace('A', 'A_')
transformer['Group'] = transformer['Group'].str.replace('C', 'C_')
transformer['Group'] = transformer['Group'].str.replace('D', 'D_')
transformer['Group'] = transformer['Group'].str.replace('S', 'S_')
transformer['Group'] = transformer['Group'].str.replace('A', 'A_')

# surround the number with curly braces
mlp['Group'] = mlp['Group'].str.replace('C_', 'C_{')
mlp['Group'] = mlp['Group'].str.replace('D_', 'D_{')
mlp['Group'] = mlp['Group'].str.replace('S_', 'S_{')
mlp['Group'] = mlp['Group'].str.replace('A_', 'A_{')
transformer['Group'] = transformer['Group'].str.replace('C_', 'C_{')
transformer['Group'] = transformer['Group'].str.replace('D_', 'D_{')
transformer['Group'] = transformer['Group'].str.replace('S_', 'S_{')
transformer['Group'] = transformer['Group'].str.replace('A_', 'A_{')

# add a closing curly brace
mlp['Group'] = mlp['Group'] + '}'
transformer['Group'] = transformer['Group'] + '}'



# add a dollar sign to the start and end of the Group name
mlp['Group'] = '$' + mlp['Group'] + '$'
transformer['Group'] = '$' + transformer['Group'] + '$'

#split each dataframe into four dataframes, one for 'C', 'D', 'S' and 'A' runs
mlp_c = mlp[mlp['run'].str.contains('C')]
mlp_d = mlp[mlp['run'].str.contains('D')]
mlp_s = mlp[mlp['run'].str.contains('S')]
mlp_a = mlp[mlp['run'].str.contains('A')]
transformer_c = transformer[transformer['run'].str.contains('C')]
transformer_d = transformer[transformer['run'].str.contains('D')]
transformer_s = transformer[transformer['run'].str.contains('S')]
transformer_a = transformer[transformer['run'].str.contains('A')]

# put them back together in that order
mlp = pd.concat([mlp_c, mlp_d, mlp_s, mlp_a])
transformer = pd.concat([transformer_c, transformer_d, transformer_s, transformer_a])

# remove the run column
mlp = mlp.drop(columns=['run'])
transformer = transformer.drop(columns=['run'])

#drop the "W_b FVE" column from transformer
transformer = transformer.drop(columns=['$W_b$ FVE'])
#rename the "W_a FVE" column to "W_E FVE"
transformer = transformer.rename(columns={'$W_a$ FVE': '$W_E$ FVE'})

# move the Group, Seed, and Kep Irreps columns to the front
mlp = mlp[['Group', 'Seed', 'Key Irreps'] + [col for col in mlp.columns if col not in ['Group', 'Seed', 'Key Irreps']]]
transformer = transformer[['Group', 'Seed', 'Key Irreps'] + [col for col in transformer.columns if col not in ['Group', 'Seed', 'Key Irreps']]]

# get summary statistics for main body
mlp_avg = mlp.copy()
transformer_avg = transformer.copy()
# drop the logit FVE and key irreps columns
mlp_avg = mlp_avg.drop(columns=['Key Irreps', 'Logit FVE'])
transformer_avg = transformer_avg.drop(columns=['Key Irreps', 'Logit FVE'])
# get the mean of all columns grouped by Group
mlp_avg = mlp_avg.groupby('Group')[["$W_a$ FVE", "$W_b$ FVE", "$W_U$ FVE", 'MLP FVE', 'MLP $\\rho(ab)$ FVE', 'Test Loss', 'Exc. Loss', 'Res. Loss']].mean()
transformer_avg = transformer_avg.groupby('Group')[["$W_E$ FVE", "$W_U$ FVE", 'MLP FVE', 'MLP $\\rho(ab)$ FVE', 'Test Loss', 'Exc. Loss', 'Res. Loss']].mean()

#reset the index so that the Group column is a column again
mlp_avg = mlp_avg.reset_index()
transformer_avg = transformer_avg.reset_index()

pd.set_option('display.max_colwidth', None)
# convert any column containing FVE to a percentage, convert any column containing Loss to scientific notation
dfs = [mlp, transformer, mlp_avg, transformer_avg]
for df in dfs:
    for col in df.columns:
        if 'FVE' in col:
            df[col] = df[col].apply(lambda x: '{:.2%}'.format(x))
            # escape the % sign
            df[col] = df[col].apply(lambda x: x.replace('%', '\%'))
        if 'Res. Loss' in col or 'Test Loss' in col:
            # if loss is between 0.01 and 100 then don't use scientific notation
            df[col] = df[col].apply(lambda x: '{:.2e}'.format(x))
        if 'Exc. Loss' in col:
            # rond to 2 decimal places
            df[col] = df[col].apply(lambda x: '{:.2f}'.format(x))



# stitch together the transformer and mlp dataframes on the Group column
# this is the dataframe that will be used for the main body of the paper
# rename the columns for the transformer dataframe to start with T
transformer_avg.columns = ['T' + col if col != 'Group' and col != 'Seed' else col for col in transformer_avg.columns]
# rename the columns for the mlp dataframe to start with M
mlp_avg.columns = ['M' + col if col != 'Group' and col != 'Seed' else col for col in mlp_avg.columns]
# merge the two dataframes on the Group column
merge = pd.merge(mlp_avg, transformer_avg, on='Group')



# save each dataframe to a latex file 
mlp.to_latex('universality_results/mlp_all.tex', index=False, escape=False, column_format = 'c'*len(mlp.columns))
transformer.to_latex('universality_results/transformer_all.tex', index=False, escape=False, column_format='c'*len(transformer.columns) )
mlp_avg.to_latex('universality_results/mlp_avg.tex', index=False, escape=False, column_format = 'c'*len(mlp_avg.columns))
transformer_avg.to_latex('universality_results/transformer_avg.tex', index=False, escape=False, column_format = 'c'*len(transformer_avg.columns))
merge.to_latex('universality_results/merge.tex', index=False, escape=False, column_format = 'c'*len(merge.columns))

# save each to a csv too
mlp.to_csv('universality_results/mlp_all.csv', index=False)
transformer.to_csv('universality_results/transformer_all.csv', index=False)
mlp_avg.to_csv('universality_results/mlp_avg.csv', index=False)
transformer_avg.to_csv('universality_results/transformer_avg.csv', index=False)
merge.to_csv('universality_results/merge.csv', index=False)






