import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hook_points import HookPoint, HookedRootModule
from utils.groups import *
import numpy as np
import json


# Define Models

class BilinearNet(HookedRootModule):
    """
    A completely linear network. W_E_a and W_E_b are embedding layers, whose outputs are elementwise multiplied. The result is unembedded by W_U.
    """
    def __init__(self, layers, n, seed=0):
        # embed_dim : dimension of the embedding
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        embed_dim = layers['embed_dim']
        
        # initialise parameters
        self.W_E_a = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_E_b = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_U = nn.Parameter(torch.randn(embed_dim, n)/np.sqrt(embed_dim))

        self.x_embed = HookPoint()
        self.y_embed = HookPoint()
        self.product = HookPoint()
        self.out = HookPoint()
        
        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch) 
        x_embed = self.x_embed(self.W_E_a[x]) # (batch, embed_dim)
        y = data[:, 1]
        y_embed = self.y_embed(self.W_E_b[y]) # (batch, embed_dim)
        product = self.product(x_embed * y_embed) # (batch, embed_dim)
        out = self.out(product @ self.W_U) # (batch, n)
        return out

class OneLayerMLP(HookedRootModule):
    def __init__(self, layers, n, seed=0):
        # embed_dim: dimension of the embedding
        # hidden : hidden dimension size
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        embed_dim = layers['embed_dim']
        hidden = layers['hidden_dim']

        # xavier initialise parameters
        self.W_E_a = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_E_b = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W = nn.Parameter(torch.randn(2*embed_dim, hidden)/np.sqrt(2*embed_dim))
        self.relu = nn.ReLU()
        self.W_U = nn.Parameter(torch.randn(hidden, n)/np.sqrt(hidden))

        # hookpoints
        self.x_embed = HookPoint()
        self.y_embed = HookPoint()
        self.embed_stack = HookPoint()
        self.hidden = HookPoint()
        self.out = HookPoint()

        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch)
        x_embed = self.x_embed(self.W_E_a[x]) # (batch, embed_dim)
        y = data[:, 1] # (batch)
        y_embed = self.y_embed(self.W_E_b[y]) # (batch, embed_dim)
        embed_stack = self.embed_stack(torch.hstack((x_embed, y_embed))) # (batch, 2*embed_dim)
        hidden = self.hidden(self.relu(embed_stack @ self.W)) # (batch, hidden)
        out = self.out(hidden @ self.W_U) # (batch, n)
        return out

class BetterOneLayerMLP(HookedRootModule):
    def __init__(self, layers, n, seed=0):
        # embed_dim: dimension of the embedding
        # hidden : hidden dimension size
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        embed_dim = layers['embed_dim']

        # xavier initialise parameters
        self.W_x = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_y = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.relu = nn.ReLU()
        self.W_U = nn.Parameter(torch.randn(embed_dim, n)/np.sqrt(embed_dim))

        # hookpoints
        self.x_embed = HookPoint()
        self.y_embed = HookPoint()
        self.embed_stack = HookPoint()
        self.hidden = HookPoint()

        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch)
        x_embed = self.x_embed(self.W_x[x]) # (batch, embed_dim)
        y = data[:, 1] # (batch)
        y_embed = self.y_embed(self.W_y[y]) # (batch, embed_dim)
        embed_stack = x_embed + y_embed # (batch, embed_dim)
        hidden = self.hidden(self.relu(embed_stack)) # (batch, embed_dim)
        logits = hidden @ self.W_U # (batch, n)
        return logits

class BetterTwoLayerMLP(HookedRootModule):
    def __init__(self, layers, n, seed=0):
        # embed_dim: dimension of the embedding
        # hidden : hidden dimension size
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        embed_dim = layers['embed_dim']
        hidden_dim = layers['hidden_dim']

        # xavier initialise parameters
        self.W_x = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_y = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.relu = nn.ReLU()
        self.W = nn.Parameter(torch.randn(embed_dim, hidden_dim)/np.sqrt(embed_dim))
        self.relu2 = nn.ReLU()
        self.W_U = nn.Parameter(torch.randn(hidden_dim, n)/np.sqrt(hidden_dim))

        # hookpoints
        self.x_embed = HookPoint()
        self.y_embed = HookPoint()
        self.embed_stack = HookPoint()
        self.hidden = HookPoint()

        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch)
        x_embed = self.x_embed(self.W_x[x]) # (batch, embed_dim)
        y = data[:, 1] # (batch)
        y_embed = self.y_embed(self.W_y[y]) # (batch, embed_dim)
        embed_stack = x_embed + y_embed # (batch, embed_dim)
        hidden1 = self.hidden(self.relu(embed_stack)) # (batch, embed_dim)
        hidden2 = self.relu2(hidden1 @ self.W) # (batch, hidden_dim)
        logits = hidden2 @ self.W_U # (batch, n)
        return logits


# Generate Data

def generate_train_test_data(group, frac_train, seed=False):
    data = group.get_all_data(seed).cuda()
    train_size = int(frac_train*data.shape[0])
    train = data[:train_size]
    test = data[train_size:]
    train_data = train[:, :2]
    train_labels = train[:, 2]
    test_data = test[:, :2]
    test_labels = test[:, 2]
    return train_data, test_data, train_labels, test_labels

# Loss Function

def loss_fn(logits, labels):
    loss = F.cross_entropy(logits, labels)
    return loss

# Load Model Config

