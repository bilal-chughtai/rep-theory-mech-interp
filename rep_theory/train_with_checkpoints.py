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
parser.add_argument('--save_every', type=int, default=100)


args = parser.parse_args()


task_dir = args.task_dir
save_every = args.save_every



if torch.cuda.is_available:
    print('CUDA available!')
else:
    print('CUDA not available!')

track_metrics = False

print(f'Training {task_dir}')

print('Loading cfg...')
seed, frac_train, layers, lr, group_param, weight_decay, num_epochs, group_type, architecture_type, metric_cfg, metric_obj = load_cfg(task_dir)

print('Initializing group...')
group = group_type(group_param, init_all=track_metrics)

wb_config = {
    "seed": seed,
    "frac_train": frac_train,
    "layers": layers,
    "lr": lr,
    "group_param": group_param,
    "weight_decay": weight_decay,
    "num_epochs": num_epochs,
    "group_type": group_type,
    "architecture_type": architecture_type    
}

wb_project_name = f'{group.__class__.__name__}RepTheoryBatch'

wandb.init(project=wb_project_name, entity="bilal-experiments", config=wb_config)

train_data, test_data, train_labels, test_labels = generate_train_test_data(group, frac_train, seed)

print('Initializing model...')
model = architecture_type(layers, group.order, seed)
model.cuda()

metrics = metric_obj(group, True, track_metrics, train_labels, test_data, test_labels, metric_cfg)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        
        if epoch%save_every == 0:
            save_checkpoint(model, epoch, task_dir)

    cleanup()
except KeyboardInterrupt:
    print('Interrupted...')
    cleanup()
    sys.exit(0)

