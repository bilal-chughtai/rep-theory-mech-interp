import os
import json

parent_directory = 'batch_experiments'

print(f'Creating experiments in {parent_directory}')

if not os.path.exists(parent_directory):
    os.mkdir(parent_directory)

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
already_created = []

def create_experiment(cfg, extra_name=""):

    # Define what the directory for this experiment would be
    experiment_name = f'{acronyms[cfg["group"]]}{cfg["group_parameter"]}_{acronyms[cfg["model"]]}{cfg["extra_name"]}_seed{cfg["seed"]}'
    experiment_directory = os.path.join(parent_directory, experiment_name)

    # Check if the experiment directory exists
    if os.path.exists(experiment_directory):
        print(f'Experiment {experiment_name} already exists!')
        already_created.append(experiment_name)
        return

    # Create a directory for the experiment
    os.mkdir(experiment_directory)
    os.mkdir(os.path.join(experiment_directory, 'checkpoints'))

    # Create a config file for the experiment 
    with open(os.path.join(experiment_directory, 'cfg.json'), 'w') as f:
        json.dump(cfg, f)
    
    experiments.append(experiment_name)



def create_on_seeds(base_cfg, cfg, seeds):
    for i in seeds:
        experiment_cfg = {**base_cfg, **cfg}
        experiment_cfg["seed"] = i

        if "num_epochs" not in experiment_cfg:
            if experiment_cfg["model"] == "OneLayerMLP":
                experiment_cfg["num_epochs"] = 250000
            elif experiment_cfg["model"] == "Transformer":
                experiment_cfg["num_epochs"] = 50000
        
        create_experiment(experiment_cfg)

base_cfg = {
    "lr" : 1e-3,
    "weight_decay" : 1,
    "betas": (0.9, 0.98),
    "layers": {
        "embed_dim": 256,
        "hidden_dim": 128
    },
    "extra_name": ""
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

#cfgs.append(cfg)

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
    "num_epochs": 20000,
}

cfgs.append(cfg)

#6: S6, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "SymmetricGroup",
    "group_parameter": 6,
    "frac_train" : 0.3, # probably could be decreased
}

#cfgs.append(cfg)

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

#cfgs.append(cfg)

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

#cfgs.append(cfg)

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
    "num_epochs": 100000,
}

cfgs.append(cfg)

#15: D59, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "DihedralGroup",
    "group_parameter": 59,
    "frac_train" : 0.3,
}

#cfgs.append(cfg)

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
    "num_epochs": 100000,
}

cfgs.append(cfg)

#18: D61, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "DihedralGroup",
    "group_parameter": 61,
    "frac_train" : 0.3,
}

#cfgs.append(cfg)

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
    "num_epochs": 250000 #seems to need longer

}

cfgs.append(cfg)

#21: A5, BilinearNet, various seeds
cfg = {
    "model": "BilinearNet",
    "group": "AlternatingGroup",
    "group_parameter": 5,
    "frac_train" : 0.5,
}

#cfgs.append(cfg)

cfgs_rebuttal = []

# S5, OneLayerMLP, various seeds, various widths

hidden_dims = [32, 48, 64, 80, 96, 112]
for hidden_dim in hidden_dims:
    cfg = {
        "model": "OneLayerMLP",
        "group": "SymmetricGroup",
        "group_parameter": 5,
        "frac_train" : 0.4, # min needed to generalise on wd = 1
        "layers": {
            "embed_dim": hidden_dim * 2,
            "hidden_dim": hidden_dim
        },
        "extra_name": f"_hidden_dim_{hidden_dim}",
        # scale such that embed_dim = 256 is baseline, and more if embed_dim small
        "num_epochs": 250000 * (128 // hidden_dim)
    }
    cfgs_rebuttal.append(cfg)



for cfg in cfgs:
    create_on_seeds(base_cfg, cfg, [1, 2, 3, 4])

for cfg in cfgs_rebuttal:
    create_on_seeds(base_cfg, cfg, [1,2,3,4])

for cfg in cfgs_rebuttal[:1]:
    create_on_seeds(base_cfg, cfg, range(5, 11))

# create the first cfg on the seeds 5 through 50
create_on_seeds(base_cfg, cfgs[0], range(5, 51))

# add a file in the parent directory that contains the names of all the new experiments, that have not yet been run
with open(os.path.join(parent_directory, 'experiments_to_run.txt'), 'a') as f:
    for experiment in experiments:
        f.write(experiment + '\n')
        print(f'Created {experiment}')

# add a file in the parent directory that contains the names of all ran experiments
with open(os.path.join(parent_directory, 'experiments_to_eval.txt'), 'a') as f:
    pass

# add a file in the parent directory that contains the names of all evaled experiments
with open(os.path.join(parent_directory, 'evaled_experiments.txt'), 'a') as f:
    pass

# add a file in the parent directory that contains the names of all experiments
with open(os.path.join(parent_directory, 'all_experiments.txt'), 'w') as f:
    for experiment in already_created:
        f.write(experiment + '\n')
    for experiment in experiments:
        f.write(experiment + '\n')