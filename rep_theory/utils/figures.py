from utils.plotting import *


def get_history_from_wb(keys, run):
    """
    keys: list of keys to extract from the run
    run: wandb run object
    """
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
    """
    keys: list of keys to extract from the run
    run: pandas dataframe
    """
    out = []
    for key in keys:
        if key not in run.keys():
            raise ValueError(f"Key {key} not a valid metric")
        out.append(run[key].values)
    out.append(run['epoch'].values)
    return out

def lines_from_keys(metrics, keys, yaxis, labels, save, title=None, trim=None, vlines=None, **kwargs):
    """
    metrics: pandas dataframe
    keys: list of keys to extract from the run
    yaxis: name of yaxis
    labels: list of labels for each key
    save: whether to save the figure
    title: title of the figure
    trim: trim the data to only include epochs up to this value
    vlines: list of x values to draw vertical lines at
    """
    data = get_history_local(keys, metrics)
    if trim is not None:
        # find where data[-1] > trim
        for i, epoch in enumerate(data[-1]):
            if epoch > trim:
                break
        data = [d[:i] for d in data]
    lines(data[:-1], xaxis="epoch", yaxis=yaxis, labels=labels, show=True, save=save, x=data[-1], title=title, vlines=vlines, **kwargs)

def lines_from_template(metrics, template, fill_with, labels=[], yaxis='metric', save=False, other_keys=[], title=None, vlines=None, **kwargs):
    """
    metrics: pandas dataframe
    template: template string to fill with irrep
    fill_with: list of irreps to fill the template with
    labels: list of labels for each key
    yaxis: name of yaxis
    save: whether to save the figure
    other_keys: list of keys to extract from the run
    title: title of the figure
    vlines: list of x values to draw vertical lines at
    """
    keys = other_keys.copy()
    for irrep in fill_with:
        keys.append(template.format(irrep))
        labels.append(irrep)
    lines_from_keys(metrics, keys, yaxis, labels, save, title=title, vlines=vlines, **kwargs)  
