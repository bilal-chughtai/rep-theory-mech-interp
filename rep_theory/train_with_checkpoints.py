import sys
import torch
from tqdm import tqdm
from utils.plotting import *
from utils.groups import *
from utils.models import *
from utils.metrics import *
from utils.config import load_cfg
from utils.checkpoints import save_checkpoint
import wandb
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task_dir', type=str, default='experiments/1L_MLP_sym_S5')
parser.add_argument('--num_checkpoints', type=int, default=1000)


args = parser.parse_args()


task_dir = args.task_dir
num_checkpoints = args.num_checkpoints
# if no checkpoint dir, make it
checkpoint_dir = os.path.join(task_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    



if torch.cuda.is_available:
    print('CUDA available!')
else:
    print('CUDA not available!')

track_metrics = False

print(f'Training {task_dir}')

print('Loading cfg...')
seed, frac_train, layers, lr, group_param, weight_decay, betas, num_epochs, group_type, architecture_type = load_cfg(task_dir)

print('Initializing group...')
group = group_type(group_param, init_all=track_metrics)

wb_config = {
    "seed": seed,
    "frac_train": frac_train,
    "layers": layers,
    "lr": lr,
    "group_param": group_param,
    "weight_decay": weight_decay,
    "betas": betas,
    "num_epochs": num_epochs,
    "group_type": group_type,
    "architecture_type": architecture_type    
}

wb_project_name = f'{group.__class__.__name__}RepTheoryBatch'

wandb.init(project=wb_project_name, entity="bilal-experiments", config=wb_config)

train_data, test_data, train_labels, test_labels, shuffled_indices = generate_train_test_data(group, frac_train, seed)

print('Initializing model...')
model = architecture_type(layers, group.order, seed)
model.cuda()

metrics = Metrics(group, True, track_metrics, train_data, train_labels, test_data, test_labels, shuffled_indices)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

# compute which epochs to checkpoint, exponentially reducing in frequency
checkpoints = list(np.logspace(0, np.log10(num_epochs), num=num_checkpoints, endpoint=True, dtype=np.int32))
checkpoints = [0] + checkpoints + [num_epochs-1]
checkpoints = list(dict.fromkeys(checkpoints))

def cleanup():
    print('Saving final model...')
    save_checkpoint(model, num_epochs, task_dir, final=True)
    print('Done.')

try:
    print('Training...')
    for epoch in tqdm(range(num_epochs)):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch%100 == 0:
            with torch.inference_mode():
                metric = metrics.get_metrics(model, train_logits, train_loss)
                wandb.log(metric)
        
        if epoch in checkpoints:
            save_checkpoint(model, epoch, task_dir)

    cleanup()
except KeyboardInterrupt:
    print('Interrupted...')
    cleanup()
    sys.exit(0)

