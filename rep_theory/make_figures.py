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
if len(non_trivial_irrep_names) > 10:
    reps_to_plot = key_rep_names
else:
    reps_to_plot = non_trivial_irrep_names

# figure: evolution of cosine similarity
template = "logit_{}_rep_trace_similarity"
lines_from_template(metrics, template, reps_to_plot, yaxis="cosine similarity", save=f"{save_dir}/logit_similarity.png", log_x=False, legend_pos='tl')

# figure: losses
keys = ['train_loss', 'test_loss']
lines_from_keys(metrics, keys, labels=keys, yaxis='Loss', save=f'{save_dir}/losses.png', log_x=False, log_y = True, legend_pos='bl')

# figure: evolution of cosine similarity
# template = "other_logit_{}_rep_trace_similarity"
# lines_from_template(metrics, template, reps_to_plot, title="cosine similarity of true logits and hypothesised logits", yaxis="cosine similarity", save=f"{save_dir}/other_logit_similarity.png", log_x=True)

# figure: evolution of logit explained
key = 'percent_logits_explained'
lines_from_keys(metrics, [key], labels=[key], yaxis='fraction of variance', save=f'{save_dir}/percent_logits_explained.png')

# figures: logit excluded and restricted loss by rep
template = "logit_excluded_loss_{}_rep"
lines_from_template(metrics, template, reps_to_plot, other_keys = ['train_loss', 'test_loss'], yaxis="cosine similarity", save=f"{save_dir}/logit_excluded_loss.png", log_x=True, log_y=True, legend_pos='bl')
template = "logit_restricted_loss_{}_rep"
lines_from_template(metrics, template, reps_to_plot, other_keys = ['train_loss', 'test_loss'], yaxis="cosine similarity", save=f"{save_dir}/logit_restricted_loss.png", log_x=True, log_y=True, legend_pos='bl')

# fogires: hidden excluded and restricted loss by rep
template = "hidden_excluded_loss_{}_rep"
lines_from_template(metrics, template, reps_to_plot, other_keys = ['train_loss', 'test_loss'], yaxis="cosine similarity", save=f"{save_dir}/hidden_excluded_loss.png", log_x=True, log_y=True, legend_pos='bl')
template = "hidden_restricted_loss_{}_rep"
lines_from_template(metrics, template, reps_to_plot, other_keys = ['train_loss', 'test_loss'], yaxis="cosine similarity", save=f"{save_dir}/hidden_restricted_loss.png", log_x=True, log_y=True, legend_pos='bl')

# figure: logit losses
keys = ['train_loss', 'test_loss', 'total_logit_excluded_loss', 'total_logit_restricted_loss']
lines_from_keys(metrics, keys, title='Logit Losses', labels=keys, yaxis='Loss', save=f'{save_dir}/logit_losses.png', log_x=True, log_y = True, legend_pos='bl')

# figure: logit losses
keys = ['train_loss', 'test_loss', 'total_hidden_excluded_loss', 'total_hidden_restricted_loss']
lines_from_keys(metrics, keys, title='Hidden Losses', labels=keys, yaxis='Loss', save=f'{save_dir}/hidden_logit_losses.png', log_x=True, log_y = True, legend_pos='bl')

# figure: total embed restricted and excluded loss
#keys = ['total_embed_excluded_loss', 'total_embed_restricted_loss', 'test_loss', 'train_loss']
#lines_from_keys(metrics, keys, title='Excluded Loss', labels=keys, yaxis='Loss', save=f'{save_dir}/total_excluded_loss.png', log_y = True)
# figure: embed excluded loss by rep
#template = 'embed_excluded_loss_{}_rep'
#lines_from_template(metrics, template, reps_to_plot, title='Excluded Loss by Representation', yaxis='Loss', save=f'{save_dir}/excluded_loss_by_rep.png', log_y = True)

# figure: percent a, b, c embed by representation over course of training
template = "percent_x_embed_{}_rep"
lines_from_template(metrics, template, reps_to_plot, yaxis="fraction of variance", save=f"{save_dir}/percent_x_embed.png", legend_pos='tl')

template = "percent_y_embed_{}_rep"
lines_from_template(metrics, template, reps_to_plot, yaxis="fraction of variance", save=f"{save_dir}/percent_y_embed.png", legend_pos='tl')

template = "percent_unembed_{}_rep"
lines_from_template(metrics, template, reps_to_plot, yaxis="fraction of variance", save=f"{save_dir}/percent_unembed.png", legend_pos='tl')

# figure: evolution of \rho(a), \rho(b), \rho(ab)
template = "percent_hidden_xy_{}_rep"
lines_from_template(metrics, template, reps_to_plot, yaxis="fraction of variance", save=f"{save_dir}/percent_hidden_ab.png", legend_pos='tr')



# figure: sum of square weights
keys = ['sum_of_squared_weights']
lines_from_keys(metrics, keys, title='Sum of Square Weights', labels=['Sum of Square Weights'], yaxis='Sum of Square Weights', save=f'{save_dir}/sum_of_square_weights.png', log_y=True)

# figure test loss / restricted loss
keys = ['test_loss_restricted_loss_ratio']
lines_from_keys(metrics, keys, title='Test Loss / Restricted Loss', labels=['Test Loss / Restricted Loss'], yaxis='Test Loss / Restricted Loss', save=f'{save_dir}/test_loss_restricted_loss_ratio.png', log_y=True)



        
