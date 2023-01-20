from utils.plotting import *


def get_history_from_wb(keys, run):
    history = run.scan_history(keys=keys)
    out = []
    keys.append('epoch')
    for key in keys:
        if key not in run.summary.keys():
            raise ValueError(f"Key {key} not a valid metric")
        out.append([])
    for row in history:
        for i, key in enumerate(keys):
            out[i].append(row[key])
        
    return out

def get_history_local(keys, run):
    # run is a dataframe
    out = []
    for key in keys:
        if key not in run.keys():
            raise ValueError(f"Key {key} not a valid metric")
        out.append(run[key].values)
    out.append(run['epoch'].values)
    return out

def lines_from_keys(metrics, keys, yaxis, labels, save, title=None, **kwargs):
    data = get_history_local(keys, metrics)
    lines(data[:-1], xaxis="epoch", yaxis=yaxis, labels=labels, show=True, save=save, x=data[-1], title=title, **kwargs)

def lines_from_template(metrics, template, fill_with, yaxis, save, other_keys=[], title=None, **kwargs):
    keys = other_keys.copy()
    labels = other_keys.copy()
    for irrep in fill_with:
        keys.append(template.format(irrep))
        labels.append(irrep)
    lines_from_keys(metrics, keys, yaxis, labels, save, title=title, **kwargs)  
