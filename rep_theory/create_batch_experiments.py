import os
import json

parent_directory = 'batch_experiments'

print(f'Creating experiments in {parent_directory}')

acronyms = {
    "OneLayerMLP": "1L_MLP",
    "SymmetricGroup": "S",
    "DihedralGroup": "D",
    "CyclicGroup": "C",

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

def create_on_seeds(cfg):
    for i in seeds:
        experiment_cfg = {**base_cfg, **cfg}
        experiment_cfg["seed"] = i
        create_experiment(experiment_cfg)

base_cfg = {
    "lr" : 1e-3,
    "num_epochs" : 1,
    "weight_decay" : 1,
    "layers": {
        "embed_dim": 256,
        "hidden_dim": 128
    },
}

#1: S5, OneLayerMLP, various seeds - works
cfg1 = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.4, # min needed to generalise on wd = 1
}

#2: S5, Transformer, various seeds
cfg2 = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.6, # yet to determine this number, still loss spiking
}

#3: S5, BilinearNet, various seeds - works
cfg3 = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,  # could be reduced
}

#4: S6, OneLayerMLP, various seeds - works
cfg4 = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25, # min needed to generalise on wd = 1
}

#5: S6, Transformer, various seeds
cfg5 = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25, # yet to determine this number
}

#6: S6, BilinearNet, various seeds - probably works
cfg6 = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.3, # probably could be decreased
}

#7: C113, OneLayerMLP, various seeds - works
cfg7 = {
    "model": "OneLayerMLP",
    "group": "CyclicGroup",
    "group_parameter": 113,
    "frac_train" : 0.3,
}

#8: C113, Transformer, various seeds - works, but weird kink in loss curve
cfg8 = {
    "model": "OneLayerMLP",
    "group": "CyclicGroup",
    "group_parameter": 113, 
    "frac_train" : 0.3, 
}

#9: C113, BilinearNet, various seeds - started overfitting at the end
cfg9 = {
    "model": "BiLinearNet",
    "group": "CyclicGroup",
    "group_parameter": 113,
    "frac_train" : 0.3, # this is overfitting
}


#10: C118, OneLayerMLP, various seeds - works
cfg10 = {
    "model": "OneLayerMLP",
    "group": "CyclicGroup",
    "group_parameter": 118,
    "frac_train" : 0.3,
}


#11: C118, BilinearNet, various seeds
cfg11 = {
    "model": "BiLinearNet",
    "group": "CyclicGroup",
    "group_parameter": 118,
    "frac_train" : 0.3,
}













cfgs = [cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7, cfg8]
for cfg in cfgs:
    create_on_seeds(cfg)

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