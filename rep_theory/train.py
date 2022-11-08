import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import einops

import json
import pickle

from tqdm.notebook import tqdm

# plotting
from functools import partial
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "vscode"

# my own tooling
from utils.hook_points import HookPoint, HookedRootModule
from utils.plotting import *
from utils.groups import *
from utils.models import *
from utils.metrics import *

if torch.cuda.is_available:
  print('Good to go!')
else:
  print('Training might be rather slow')

train = True
save_metrics = False

task_dir = "1L_MLP_sym_S5"
seed, frac_train, layers, lr, group_param, weight_decay, num_epochs, group_type, architecture_type, metrics = load_cfg(task_dir)
group = group_type(group_param)

if train:
    train_data, test_data, train_labels, test_labels = generate_train_test_data(group, frac_train, seed)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    model = architecture_type(layers, group.order, seed)
    model.cuda()
    metrics = Metrics(model, group, metrics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs)):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss.item())
        with torch.inference_mode():
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)
            test_losses.append(test_loss.item())
            train_acc = (train_logits.argmax(1)==train_labels).sum()/len(train_labels)
            test_acc = (test_logits.argmax(1)==test_labels).sum()/len(test_labels)
            train_accs.append(train_acc.item())
            test_accs.append(test_acc.item())

            if save_metrics and epoch % 100 == 0:
                metrics.update_model(model)
                metrics.get_metrics()

        if epoch%1000 == 0:
            print(f"Epoch:{epoch}, Train: L: {train_losses[-1]:.6f} A: {train_accs[-1]*100:.4f}%, Test: L: {test_losses[-1]:.6f} A: {test_accs[-1]*100:.4f}%")
        #if epoch%50000 == 0 and epoch>0:
            #lines([train_losses, test_losses], log_y=True, labels=['train loss', 'test loss'])
            #lines([train_accs, test_accs], log_y=False, labels=['train acc', 'test acc'])


lines([train_losses, test_losses], log_y=True, labels=['train loss', 'test loss'], save=f"{task_dir}/loss.png")
lines([train_accs, test_accs], log_y=False, labels=['train acc', 'test acc'], save=f"{task_dir}/acc.png")
torch.save(model.state_dict(), f"{task_dir}/model.pt")
if save_metrics:
    with open(f'{task_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics.data, f)
