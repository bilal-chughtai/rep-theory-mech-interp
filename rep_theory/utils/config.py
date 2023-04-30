from utils.models import *
from utils.metrics import *
from utils.groups import *

def load_cfg(task_dir):
    cfg_file = open(f"{task_dir}/cfg.json")
    cfg = json.load(cfg_file)

    seed = cfg['seed']
    frac_train = cfg['frac_train']
    layers = cfg['layers']
    lr = cfg['lr']
    group_param = cfg['group_parameter']
    weight_decay = cfg['weight_decay']
    betas = cfg['betas']
    num_epochs = cfg['num_epochs']
    group_type = eval(cfg['group'])
    architecture_type = eval(cfg['model'])
    return seed, frac_train, layers, lr, group_param, weight_decay, betas, num_epochs, group_type, architecture_type