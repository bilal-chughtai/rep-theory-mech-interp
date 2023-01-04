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

#1: S5, OneLayerMLP, various seeds

cfg = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.4,
    "metrics": {
        "class": "SymmetricMetrics"
        }
}

create_on_seeds(cfg)

#2: S6, OneLayerMLP, various seeds

cfg = {
    "model": "OneLayerMLP",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25,
    "metrics": {
        "class": "SymmetricMetrics"
    }
}

#3: S5, Transformer, various seeds

cfg = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.4,
    "metrics": {
        "class": "SymmetricMetrics"
    }
}

create_on_seeds(cfg)

#3: S6, Transformer, various seeds

cfg = {
    "model": "Transformer",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25,
    "metrics": {
        "class": "SymmetricMetrics"
    }
}

create_on_seeds(cfg)

#5: S5, BilinearNet, various seeds

cfg = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,
    "metrics": {
        "class": "SymmetricMetrics"
    }
}

#6: S6, BilinearNet, various seeds

cfg = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.25,
    "metrics": {
        "class": "SymmetricMetrics"
    }
}























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