import sys
sys.path.append('../')

import torch
from tqdm.notebook import tqdm

# plotting
import plotly.io as pio
pio.renderers.default = "vscode"

# my own tooling
from utils.plotting import *
from utils.groups import *
from utils.models import *
from utils.metrics import *

import wandb

if torch.cuda.is_available:
  print('Good to go!')
else:
  print('Training might be rather slow')

track_metrics = True

task_dir = "1L_MLP_sym_S5"

print('Loading cfg...')
seed, frac_train, layers, lr, group_param, weight_decay, num_epochs, group_type, architecture_type, metric_cfg = load_cfg(task_dir)
metric_obj = eval(metric_cfg['class'])

print('Initializing group...')
group = group_type(group_param, init_all=track_metrics)

config = {
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

wandb.init(project=task_dir, entity="bilal-experiments", config=config)

train_data, test_data, train_labels, test_labels = generate_train_test_data(group, frac_train, seed)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

print('Initializing model...')
model = architecture_type(layers, group.order, seed)
model.cuda()
metrics = metric_obj(group, train_labels, test_data, test_labels, track_metrics, metric_cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

print('Training...')
for epoch in tqdm(range(num_epochs)):
    train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_losses.append(train_loss.item())
    with torch.inference_mode():
        metric = metrics.get_metrics(model, train_logits, train_loss)
        test_losses.append(metric['test_loss'])
        train_accs.append(metric['train_acc'])
        test_accs.append(metric['test_acc'])
        wandb.log(metric)

    if epoch%1000 == 0:
        print(f"Epoch:{epoch}, Train: L: {train_losses[-1]:.6f} A: {train_accs[-1]*100:.4f}%, Test: L: {test_losses[-1]:.6f} A: {test_accs[-1]*100:.4f}%")

lines([train_losses, test_losses], log_y=True, labels=['train loss', 'test loss'], save=f"{task_dir}/loss.png")
lines([train_accs, test_accs], log_y=False, labels=['train acc', 'test acc'], save=f"{task_dir}/acc.png")
torch.save(model.state_dict(), f"{task_dir}/model.pt")
