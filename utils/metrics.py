from utils.plotting import fft2d
from utils.models import *
import torch


class Metrics():
    def __init__(self, model, group, which_metrics, cfg):
        self.model = model
        self.group = group
        self.all_data, self.all_labels, _, _ = generate_train_test_data(self.group, frac_train = 1) 
        self.which_metrics = which_metrics
        self.cfg = cfg

    def update_model(self, model):
        self.model = model

    def get_metrics(self, model):
        self.model.eval()
        self.logits = model(self.all_data)
        for metric in self.which_metrics:
            eval(metric)(self.cfg[metric])
        self.model.train()



    def save_metrics():
        pass

    def restricted_loss_fourier(logits, group, key_freqs):
        fourier_logits = fft2d(logits.reshape(group.order, group.order, -1), group.fourier_basis)
        for freq in key_freqs:
            mask = torch.zeros_like(fourier_logits).squeeze(-1)
            mask[0, 0] = 1
            mask[2*freq, 0] = 1
            mask[0, 2*freq] = 1
            mask[2*freq-1, 0] = 1
            mask[0, 2*freq-1] = 1
            mask[2*freq, 2*freq] = 1
            mask[2*freq-1, 2*freq] = 1
            mask[2*freq, 2*freq-1] = 1
            mask[2*freq-1, 2*freq-1] = 1
            mask = 1-mask
            fourier_logits_restricted = (fourier_logits.permute(2,0,1)*mask).permute(1,2,0)
            logits_restricted=fft2d(fourier_logits, group.fourier_basis, inverse=True)
            fourier_logits.reshape(group.order*group.order, -1)

