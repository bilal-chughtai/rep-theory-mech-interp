from utils.plotting import fft2d
from utils.models import *
import torch


class Metrics():
    def __init__(self, model, group, which_metrics):
        self.model = model
        self.group = group
        self.all_data, _, self.all_labels, _ = generate_train_test_data(self.group, frac_train = 1) 
        self.which_metrics = which_metrics
        self.data = {}


    def update_model(self, model):
        self.model = model

    def get_metrics(self):
        self.model.eval()
        self.logits = self.model(self.all_data)
        for metric in self.which_metrics:
            if not metric in self.data:
                self.data[metric] = {}
            eval('self.'+metric)(self.which_metrics[metric])
        self.model.train()

    def excluded_loss_mod_add(self, cfg):

        key_freqs = cfg['key_freqs']
        fourier_logits = self.logits @ self.group.fourier_basis.T # (batch, position) @ (position, frequency) -> (batch, frequency)
        fourier_logits = fourier_logits.reshape(self.group.order, self.group.order, self.group.index)


        for freq in key_freqs:
            if not freq in self.data['excluded_loss_mod_add']:
                self.data['excluded_loss_mod_add'][freq] = []
            fourier_logits_new = fourier_logits.clone()
            cos_logits_freq = fft2d(fourier_logits_new[:, :, 2*freq-1].unsqueeze(dim=-1), self.group.fourier_basis)
            sin_logits_freq = fft2d(fourier_logits_new[:, :, 2*freq].unsqueeze(dim=-1), self.group.fourier_basis)


            fourier_logits_new[:, :, 2*freq-1] -= fft2d(cos_logits_freq, self.group.fourier_basis, inverse=True)
            fourier_logits_new[:, :, 2*freq] -= fft2d(sin_logits_freq, self.group.fourier_basis, inverse=True)

            fourier_logits_new = fourier_logits_new.reshape(self.group.order*self.group.order, self.group.index)
            logits_new = fourier_logits_new @ self.group.fourier_basis
            loss_new = loss_fn(logits_new, self.all_labels)
            self.data['excluded_loss_mod_add'][freq].append(loss_new.item())
