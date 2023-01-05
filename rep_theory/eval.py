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

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_dir', type=str, default='experiments/1L_MLP_sym_S5')

args = parser.parse_args()
task_dir = args.task_dir
checkpoint_dir = os.path.join(task_dir, 'checkpoints')

if torch.cuda.is_available:
    print('CUDA available!')
else:
    print('CUDA not available!')

print(f'Evaluating {task_dir}')

print('Loading cfg...')
seed, frac_train, layers, lr, group_param, weight_decay, num_epochs, group_type, architecture_type, metric_cfg, metric_obj = load_cfg(task_dir)

track_metrics = True

print('Initializing group...')
group = group_type(group_param, init_all=track_metrics)

train_data, test_data, train_labels, test_labels = generate_train_test_data(group, frac_train, seed)

print('Initializing model...')
model = architecture_type(layers, group.order, seed)
model.cuda()

metrics = metric_obj(group, True, track_metrics, train_labels, test_data, test_labels, metric_cfg)



print('Evaluating...')



# Analyse final model first
model = load_checkpoint(model, task_dir, final=True)

# determine the key reps
#metrics.determine_key_reps(model)

with torch.inference_mode():
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        metric = metrics.get_metrics(model, train_logits, train_loss)
        metric['epoch'] = num_epochs

# make all the metrics numbers
for key in metric.keys():
    metric[key] = float(metric[key])

# Save the metrics to a json
with open(os.path.join(task_dir, 'summary_metrics.json'), 'w') as f:
    json.dump(metric, f)

# Create a dataframe to store the metrics
df = pd.DataFrame()


# Check if there are any checkpoints
if not os.path.exists(checkpoint_dir):
    print('No checkpoints found')
    sys.exit()

# Get a list of all the checkpoints
checkpoints = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
epochs = [int(f.split('_')[1].split('.')[0]) for f in checkpoints]


for epoch in epochs:
    # Load the checkpoint
    model = load_checkpoint(task_dir, epoch)
    # Evaluate the model
    with torch.inference_mode():
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        metric = metrics.get_metrics(model, train_logits, train_loss)
        metric['epoch'] = epoch

    # make all the metrics numbers on the cpu
    for key in metric.keys():
        metric[key] = to_numpy(metric[key])
    # Add the metrics to the dataframe
    df = pd.concat([df, pd.DataFrame.from_dict([metric])], ignore_index=True)

# Save the dataframe to a csv
df = df.sort_values(by='epoch')
df.to_csv(os.path.join(task_dir, 'metrics.csv'))

        
