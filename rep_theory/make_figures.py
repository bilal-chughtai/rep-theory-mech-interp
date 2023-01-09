import sys
import torch
from tqdm import tqdm
from utils.plotting import *
from utils.groups import *
from utils.models import *
from utils.metrics import *
from utils.config import load_cfg
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.figures import *
import wandb
import argparse
import pandas as pd

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_dir', type=str, default='experiments/1L_MLP_sym_S5')

args = parser.parse_args()
task_dir = args.task_dir
metrics_path = os.path.join(task_dir, 'metrics.csv')
summary_metrics_path = os.path.join(task_dir, 'summary_metrics.json')

# create the figures directory
figures_dir = os.path.join(task_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

save_dir = figures_dir

# load the metrics
metrics = pd.read_csv(metrics_path)
summary_metrics = json.load(open(summary_metrics_path, 'r'))

# get the name of all the irreps from the text file
non_trivial_irrep_names = []
with open(os.path.join(task_dir, 'irreps.txt'), 'r') as f:
    for line in f:
        non_trivial_irrep_names.append(line.strip())

# get the names of the key reps from the text file
key_rep_names = []
with open(os.path.join(task_dir, 'key_reps.txt'), 'r') as f:
    for line in f:
        key_rep_names.append(line.strip())

non_trivial_irrep_names.remove('trivial')
print(metrics)     

# figure: evolution of cosine similarity
template = "logit_{}_rep_trace_similarity"
lines_from_template(metrics, template, key_rep_names, title="cosine similarity of true logits and hypothesised logits", yaxis="cosine similarity", save=f"{save_dir}/logit_similarity.png")

# figure: percent a, b, c embed by representation over course of training

template = "percent_x_embed_{}_rep"
lines_from_template(metrics, template, key_rep_names, title="Fraction of variance of left embedding explained by representation", yaxis="fraction of variance", save=f"{save_dir}/percent_x_embed.png")

template = "percent_y_embed_{}_rep"
lines_from_template(metrics, template, key_rep_names, title="Fraction of variance of right embedding explained by representation", yaxis="fraction of variance", save=f"{save_dir}/percent_y_embed.png")

template = "percent_unembed_{}_rep"
lines_from_template(metrics, template, key_rep_names, title="Fraction of variance of unembedding explained by representation", yaxis="fraction of variance", save=f"{save_dir}/percent_unembed.png")

# figure: evolution of \rho(a), \rho(b), \rho(ab)
template = "percent_hidden_{}_rep"
lines_from_template(metrics, template, key_rep_names, title="Fraction of variance of MLP neurons explained by representation", yaxis="fraction of variance", save=f"{save_dir}/percent_hidden_ab.png")

# figure: total excluded loss
keys = ['total_excluded_loss', 'test_loss', 'train_loss']
lines_from_keys(metrics, keys, title='Excluded Loss', labels=['Excluded Loss', 'Test Loss', 'Train Loss'], yaxis='Loss', save=f'{save_dir}/total_excluded_loss.png', log_y = True)

# figure: excluded loss by rep
template = 'excluded_loss_{}_rep'
lines_from_template(metrics, template, key_rep_names, title='Excluded Loss by Representation', yaxis='Loss', save=f'{save_dir}/excluded_loss_by_rep.png', log_y = True)

# figure: restricted loss
keys = ['restricted_loss', 'test_loss', 'train_loss']
lines_from_keys(metrics, keys, title='Restricted Loss', labels=['Restricted Loss', 'Test Loss', 'Train Loss'], yaxis='Loss', save=f'{save_dir}/restricted_loss.png', log_y=True)

# figure: sum of square weights
keys = ['sum_of_squared_weights']
lines_from_keys(metrics, keys, title='Sum of Square Weights', labels=['Sum of Square Weights'], yaxis='Sum of Square Weights', save=f'{save_dir}/sum_of_square_weights.png', log_y=True)




        
