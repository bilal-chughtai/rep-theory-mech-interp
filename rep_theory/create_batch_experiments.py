import os
import json

parent_directory = 'batch_experiments'

print(f'Creating experiments in {parent_directory}')

acronyms = {
    "OneLayerMLP": "MLP",
    "Transformer": "T",
    "BilinearNet": "B",
    "SymmetricGroup": "S",
    "DihedralGroup": "D",
    "CyclicGroup": "C",
    "AlternatingGroup": "A"
}

experiments = []

def create_experiment(cfg):

    # Define what the directory for this experiment would be
    experiment_name = f'{acronyms[cfg["model"]]}_{acronyms[cfg["group"]]}{cfg["group_parameter"]}_seed{cfg["seed"]}'
    experiment_directory = os.path.join(parent_directory, experiment_name)

    # Check if the experiment directory exists
    if os.path.exists(experiment_directory):
        print(f'Experiment {experiment_name} already exists!')
        return

    # Create a directory for the experiment
    os.mkdir(experiment_directory)
    os.mkdir(os.path.join(experiment_directory, 'checkpoints'))

    # Create a config file for the experiment 
    with open(os.path.join(experiment_directory, 'cfg.json'), 'w') as f:
        json.dump(cfg, f)
    
    experiments.append(experiment_name)


seeds = [1, 2, 3, 4]

def create_on_seeds(base_cfg, cfg):
    for i in seeds:
        experiment_cfg = {**base_cfg, **cfg}
        experiment_cfg["seed"] = i
        create_experiment(experiment_cfg)

base_cfg = {
    "lr" : 1e-3,
    "num_epochs" : 100000,
    "weight_decay" : 1,
    "layers": {
        "embed_dim": 256,
        "hidden_dim": 128
    },
}

cfgs = []


#1: S5, OneLayerMLP, various seeds 
cfg = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.4, # min needed to generalise on wd = 1
}

cfgs.append(cfg)

#2: S5, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.6, # yet to determine this number, still loss spiking
}

cfgs.append(cfg)

#3: S5, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,  # could be reduced
}

cfgs.append(cfg)

#4: S6, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25, # min needed to generalise on wd = 1
}

cfgs.append(cfg)

#5: S6, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.4, 
}

cfgs.append(cfg)

#6: S6, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.3, # probably could be decreased
}

cfgs.append(cfg)

#7: C113, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "CyclicGroup",
    "group_parameter": 113,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#8: C113, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "CyclicGroup",
    "group_parameter": 113, 
    "frac_train" : 0.3, 
}

cfgs.append(cfg)

#9: C113, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "CyclicGroup",
    "group_parameter": 113,
    "frac_train" : 0.3, #TODO: this is overfitting
}

cfgs.append(cfg)

#10: C118, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "CyclicGroup",
    "group_parameter": 118,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#11: C118, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "CyclicGroup",
    "group_parameter": 118,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#12: C118, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "CyclicGroup",
    "group_parameter": 118,
    "frac_train" : 0.3, #TODO: this is overfitting
}


#13: D59, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "DihedralGroup",
    "group_parameter": 59,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#14: D59, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "DihedralGroup",
    "group_parameter": 59,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#15: D59, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "DihedralGroup",
    "group_parameter": 59,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#16: D61, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "DihedralGroup",
    "group_parameter": 61,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#17: D61, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "DihedralGroup",
    "group_parameter": 61,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#18: D61, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "DihedralGroup",
    "group_parameter": 61,
    "frac_train" : 0.3,
}

cfgs.append(cfg)

#19: a5, OneLayerMLP, various seeds
cfg = {
    "model": "OneLayerMLP",
    "group": "AlternatingGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,
}

cfgs.append(cfg)

#20: A5, Transformer, various seeds
cfg = {
    "model": "Transformer",
    "group": "AlternatingGroup",
    "group_parameter": 5,
    "frac_train" : 0.6,
}

cfgs.append(cfg)

#21: A5, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "AlternatingGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,
}

cfgs.append(cfg)








for cfg in cfgs:
    create_on_seeds(base_cfg, cfg)

# add a file in the parent directory that contains the names of all the experiments
with open(os.path.join(parent_directory, 'unran_experiments.txt'), 'a') as f:
    for experiment in experiments:
        f.write(experiment + '\n')
        print(f'Created {experiment}')

# add a file in the parent directory that contains the names of all ran experiments
with open(os.path.join(parent_directory, 'ran_experiments.txt'), 'a') as f:
    pass

# add a file in the parent directory that contains the names of all evaled experiments
with open(os.path.join(parent_directory, 'evaled_experiments.txt'), 'a') as f:
    pass