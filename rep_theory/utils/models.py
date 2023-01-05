import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hook_points import HookPoint, HookedRootModule
from utils.groups import *
import numpy as np
import json


import transformer_lens
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache


# Define Models

class BilinearNet(HookedRootModule):
    """
    A completely linear network. W_x and W_y are embedding layers, whose outputs are elementwise multiplied. The result is unembedded by W_U.
    """
    def __init__(self, layers, n, seed=0):
        # embed_dim : dimension of the embedding
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        embed_dim = layers['embed_dim']
        
        # initialise parameters
        self.W_x = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_y = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.W_U = nn.Parameter(torch.randn(embed_dim, n)/np.sqrt(embed_dim))

        self.hidden = HookPoint()
        
        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch) 
        x_embed = self.W_x[x] # (batch, embed_dim)
        y = data[:, 1]
        y_embed = self.W_y[y] # (batch, embed_dim)
        hidden = self.hidden(x_embed * y_embed) # (batch, embed_dim)
        out = hidden @ self.W_U # (batch, n)

        # for metrics
        self.x_embed = self.W_x
        self.y_embed = self.W_y
        return out

class OneLayerMLP(HookedRootModule):
    def __init__(self, layers, n, seed=0):
        # embed_dim: dimension of the embedding
        # hidden : hidden dimension size
        # n : group order
        super().__init__()
        torch.manual_seed(seed)

        self.embed_dim = layers['embed_dim']
        hidden = layers['hidden_dim']

        # xavier initialise parameters
        self.W_x = nn.Parameter(torch.randn(n, self.embed_dim)/np.sqrt(self.embed_dim))
        self.W_y = nn.Parameter(torch.randn(n, self.embed_dim)/np.sqrt(self.embed_dim))
        self.W = nn.Parameter(torch.randn(2*self.embed_dim, hidden)/np.sqrt(2*self.embed_dim))
        self.relu = nn.ReLU()
        self.W_U = nn.Parameter(torch.randn(hidden, n)/np.sqrt(hidden))

        # hookpoints
        self.embed_stack = HookPoint()
        self.hidden = HookPoint()

        # We need to call the setup function of HookedRootModule to build an 
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, data):
        x = data[:, 0] # (batch)
        half_x_embed = self.W_x[x] # (batch, embed_dim)
        y = data[:, 1] # (batch)
        half_y_embed = self.W_y[y] # (batch, embed_dim)
        embed_stack = self.embed_stack(torch.hstack((half_x_embed, half_y_embed))) # (batch, 2*embed_dim)
        hidden = self.hidden(self.relu(embed_stack @ self.W)) # (batch, hidden)
        out = hidden @ self.W_U # (batch, n)

        # for metrics
        self.x_embed = self.W_x @ self.W[:self.embed_dim, :]
        self.y_embed = self.W_y @ self.W[self.embed_dim:, :]

        return out


class Transformer(HookedTransformer):
    # Hacky subclass of TransformerLens' Transformer to tokenize input data correctly.
    def __init__(self, layers, n, seed=0):
        cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 128,
        d_head = 32,
        d_mlp = 512,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=n+1,
        d_vocab_out=n,
        n_ctx=3,
        init_weights=True,
        device="cuda",
        seed = seed,
    )
        super().__init__(cfg)
        self.n = n

    def forward(self, data):
        x = data[:, 0]
        y = data[:, 1]
        equals_vector = self.n*torch.ones_like(x)
        data = torch.stack((x, y, equals_vector), dim=1).long()
        logits = super().forward(data)
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        return logits


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

