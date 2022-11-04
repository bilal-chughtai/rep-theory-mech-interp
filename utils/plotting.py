
import torch
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "vscode"



# This is mostly a bunch of over-engineered mess to hack Plotly into producing 
# the pretty pictures I want, I recommend not reading too closely unless you 
# want Plotly hacking practice
def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def imshow(tensor, input1=None, input2=None, animation_name='Snapshot', **kwargs):
    # if tensor.shape[0]==p*p:
    #     tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False), 
              labels={'x':input2, 'y':input1, 'animation_name':animation_name}, 
              **kwargs).show()

# Set default colour scheme
imshow_pos = partial(imshow, color_continuous_scale='Blues')
# Creates good defaults for showing divergent colour scales (ie with both 
# positive and negative values, where 0 is white)

imshow = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps 
# of activations with x axis being input 1 and y axis being input 2.

def imshow_fourier(tensor, fourier_basis_names, title='', animation_name='snapshot', facet_labels=[], **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    tensor = torch.squeeze(tensor)
    fig=px.imshow(to_numpy(tensor),
            x=fourier_basis_names, 
            y=fourier_basis_names, 
            labels={'y':'a component', # this is confusing. The y axis of the graph contains input 1
                    'x':'b component', # The x axis of the graph contains input 2 
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0., 
            color_continuous_scale='RdBu', 
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{y}a * %{x}b<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()


inputs_heatmap = partial(imshow, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0)

def line(x, y=None, hover=None, xaxis='', yaxis='', save=False, **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()
    if save:
        fig.write_image(save)

def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, save=False, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()
    if save:
        fig.write_image(save)

def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()


def fft2d(mat, fourier_basis, inverse=False, stack=False):
    # Converts a (p, p, ...) tensor into the 2D Fourier basis, relevant for the cyclic group.
    shape = mat.shape

    # if cyclic group
    if shape[0] == fourier_basis.shape[0]:
        if not inverse:
            fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
        else:
            fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat.unsqueeze(-1), fourier_basis.T, fourier_basis.T)
        return fourier_mat.squeeze()

    # if dihedral group - this is pretty wrong
    if stack:
        fourier_mats=[]
        for m in mat:
            fourier_mats.append(torch.einsum('xyz,fx,Fy->fFz', m, fourier_basis, fourier_basis))
        return torch.stack(fourier_mats).squeeze()
        

        