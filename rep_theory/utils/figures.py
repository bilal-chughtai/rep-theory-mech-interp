


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
        if key not in run.columns():
            raise ValueError(f"Key {key} not a valid metric")
        out.append(run[key].values)
    out.append(run['epoch'].values)
    return out

def lines_from_keys(keys, title, yaxis, labels, save, **kwargs):
    data = get_history_from_wb(keys)
    lines(data[:-1], title=title, xaxis="epoch", yaxis=yaxis, labels=labels, show=False, save=save, x=data[-1] **kwargs)

def lines_from_template(template, title, yaxis, save, **kwargs):
    non_trivial_irreps_names = list(group.non_trivial_irreps.keys())
    keys = []
    for irrep in group.non_trivial_irreps:
        keys.append(template.format(irrep))
    lines_from_keys(keys, title, yaxis, non_trivial_irreps_names, save, **kwargs)  
