import sys
import torch
from tqdm import tqdm
from utils.plotting import *
from utils.groups import *
from utils.models import *
from utils.metrics import *
from utils.config import load_cfg
from utils.checkpoints import save_checkpoint, load_checkpoint
import wandb
import argparse
import pandas as pd
from tqdm import tqdm

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_dir', type=str, default='experiments/1L_MLP_sym_S5')

args = parser.parse_args()
task_dir = args.task_dir
checkpoint_dir = os.path.join(task_dir, 'checkpoints')

# read in metrics.csv
df = pd.read_csv(os.path.join(task_dir, 'metrics.csv'))

#read in key_reps from key_reps.txt
with open(os.path.join(task_dir, 'key_reps.txt'), 'r') as f:
    key_reps = f.readlines()
    key_reps = [key_rep.strip() for key_rep in key_reps]

# if blank, get logit similarity from summary metrics
if len(key_reps) == 0:
    key_reps = []
    # read in irreps from irreps.txt
    with open(os.path.join(task_dir, 'irreps.txt'), 'r') as f:
        irreps = f.readlines()
        irreps = [irrep.strip() for irrep in irreps]
        non_trivial_irreps = [irrep for irrep in irreps if irrep != 'trivial']
    # loop over irreps
    for irrep in non_trivial_irreps:
        logit_sim = df[f'logit_{irrep}_rep_trace_similarity'].iloc[-1]
        if logit_sim > 0.05:
            key_reps.append(irrep)
            

            


# Use logit similarity to determine which order the key reps were learnt in
key_reps_in_order = []
# loop over row in dataframe
for i, row in df.iterrows():
    # get the logit similarity of each key rep
    for key_rep in key_reps:
        logit_sim = row[f'logit_{key_rep}_rep_trace_similarity']
        if logit_sim > 0.05 and key_rep not in key_reps_in_order:
            key_reps_in_order.append(key_rep)
            break
        
# add the rest
for key_rep in key_reps:
    if key_rep not in key_reps_in_order:
        key_reps_in_order.append(key_rep)

assert len(key_reps_in_order) == len(key_reps)


# Create a text file listing all the key reps of the group, in order
with open(os.path.join(task_dir, 'key_reps_in_order.txt'), 'w') as f:
    for irrep in key_reps_in_order:
        f.write(f'{irrep}\n')
        
        
