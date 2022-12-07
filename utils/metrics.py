from utils.plotting import fft2d
from utils.models import loss_fn
import torch
import torch.nn.functional as F

class Metrics():
    def __init__(self, group, training, track_metrics, train_labels=None, test_data=None, test_labels=None):
        self.training = training
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.group = group
        self.track_metrics = track_metrics
        all_data = group.get_all_data().cuda()
        self.all_data = all_data[:, :2]
        self.all_labels = all_data[:, 2]
    
    def get_accuracy(self, logits, labels):
        return ((logits.argmax(1)==labels).sum()/len(labels)).item()

    def get_test_all_logits(self, model):
        test_logits = model(self.test_data)
        all_logits = model(self.all_data)
        return test_logits, all_logits

    def get_standard_metrics(self, test_logits, train_logits, train_loss):
        metrics = {}
        metrics['train_loss'] = train_loss.item()
        metrics['test_loss'] = loss_fn(test_logits, self.test_labels).item()
        metrics['train_acc'] = self.get_accuracy(train_logits, self.train_labels)
        metrics['test_acc'] = self.get_accuracy(test_logits, self.test_labels)
        return metrics


class SymmetricMetrics(Metrics):
    def __init__(self, group, training, track_metrics, train_labels=None, test_data=None, test_labels=None, cfg=None):
        super().__init__(group, training, track_metrics, train_labels, test_data, test_labels)

        self.reps = {
            'trivial': self.group.trivial_reps,
            'sign': self.group.sign_reps,
            'standard': self.group.standard_reps,
            'standard_sign': self.group.standard_sign_reps
        }

        self.rep_trace_tensor_cubes = {
            'trivial': self.group.trivial_trace_tensor_cubes,
            'sign': self.group.sign_trace_tensor_cubes,
            'standard': self.group.standard_trace_tensor_cubes,
            'standard_sign': self.group.standard_sign_trace_tensor_cubes
        }

        self.orth_reps = {
            'trivial': self.group.trivial_reps_orth,
            'sign': self.group.sign_reps_orth,
            'standard': self.group.standard_reps_orth,
            'standard_sign': self.group.standard_sign_reps_orth,
        }
        
        # need to seperately normalise these as not orthonormal by default
        self.hidden_reps_xy = {
            'trivial': self.group.trivial_reps[self.all_labels].reshape(self.group.order*self.group.order, -1),
            'sign': self.group.sign_reps[self.all_labels].reshape(self.group.order*self.group.order, -1),
            'standard': self.group.standard_reps[self.all_labels].reshape(self.group.order*self.group.order, -1),
            'standard_sign': self.group.standard_sign_reps[self.all_labels].reshape(self.group.order*self.group.order, -1)
        }

        for key, value in self.hidden_reps_xy.items():
            self.hidden_reps_xy[key] = value / value.norm(dim=0, keepdim=True)



        if self.group.index == 4:
            self.reps['s4_2d'] = self.group.S4_2d_reps
            self.orth_reps['s4_2d'] = self.group.S4_2d_reps_orth
            self.hidden_reps_xy['s4_2d'] = self.group.S4_2d_reps[self.all_labels].reshape(self.group.order*self.group.order, -1)
        
        if self.group.index == 6:
            self.reps['s6_5d_a'] = self.group.S6_5d_a_reps
            self.orth_reps['s6_5d_a'] = self.group.S6_5d_a_reps_orth
            self.hidden_reps_xy['s6_5d_a'] = self.group.S6_5d_a_reps[self.all_labels].reshape(self.group.order*self.group.order, -1)

            self.reps['s6_5d_b'] = self.group.S6_5d_b_reps
            self.orth_reps['s6_5d_b'] = self.group.S6_5d_b_reps_orth
            self.hidden_reps_xy['s6_5d_b'] = self.group.S6_5d_b_reps[self.all_labels].reshape(self.group.order*self.group.order, -1)
            


    def get_metrics(self, model, train_logits=None, train_loss=None):
        metrics = {}

        if self.training:
            test_logits, all_logits = self.get_test_all_logits(model)
            metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        else:
            all_logits = model(self.all_data)

        if self.track_metrics:

            # losses
            # metrics['alternating_loss'] = self.loss_on_alternating_group(model)
            metrics['all_loss'] = self.loss_all(all_logits)

            # reps
            for rep_name in self.reps.keys():
                metrics[f'logit_{rep_name}_rep_trace_similarity'] = self.logit_trace_similarity(all_logits, self.rep_trace_tensor_cubes[rep_name])
                metrics[f'percent_x_embed_{rep_name}_rep'], metrics[f'percent_std_x_embed_{rep_name}_rep'], metrics[f'percent_y_embed_{rep_name}_rep'], metrics[f'percent_std_y_embed_{rep_name}_rep']= self.percent_total_embed(model, self.orth_reps[rep_name])
                metrics[f'percent_unembed_{rep_name}_rep'], metrics[f'percent_std_unembed_{rep_name}_rep']  = self.percent_unembed(model, self.orth_reps[rep_name])
                metrics[f'percent_hidden_{rep_name}_rep'], metrics[f'percent_std_hidden_{rep_name}_rep'] = self.percent_hidden(model, self.hidden_reps_xy[rep_name])
            
        return metrics

    def logit_trace_similarity(self, logits, trace_cube):
        logits = logits.reshape(self.group.order, self.group.order, -1)
        centred_logits = logits - logits.mean(-1, keepdim=True)
        centred_logits = centred_logits.reshape(self.group.order*self.group.order,-1)
        centred_trace = trace_cube.reshape(self.group.order*self.group.order,-1)
        sims = F.cosine_similarity(centred_logits, centred_trace, dim=-1)
        return sims.mean().item()

    def percent_unembed(self, model, orth_rep):
        norm_U = model.W_U.pow(2).sum()
        coefs_U = orth_rep.T @ model.W_U.T
        conts_U = coefs_U.pow(2).sum(-1) / norm_U
        return conts_U.sum(), conts_U.std()

    def percent_total_embed(self, model, orth_rep):
        # same as above, but multiply out linear layers
        dims = orth_rep.shape[1]
        embed_dim = model.W_x.shape[1]
        x_embed = model.W_x @ model.W[:embed_dim, :]
        y_embed = model.W_y @ model.W[embed_dim:, :]

        norm_x = x_embed.pow(2).sum()
        norm_y = y_embed.pow(2).sum()

        coefs_x = orth_rep.T @ x_embed
        coefs_y = orth_rep.T @ y_embed

        conts_x = coefs_x.pow(2).sum(-1) / norm_x
        conts_y = coefs_y.pow(2).sum(-1) / norm_y

        return conts_x.sum(), conts_x.std(), conts_y.sum(), conts_y.std()
        
    def percent_hidden(self, model, hidden_reps_xy):
        logits, activations = model.run_with_cache(self.all_data, return_cache_object=False)
        hidden = activations['hidden'] # DONT CENTER - activations['hidden'].mean(0, keepdim=True) # center
        hidden_norm = hidden.pow(2).sum()

        coefs_xy = hidden_reps_xy.T @ hidden
        xy_conts = coefs_xy.pow(2).sum(-1) / hidden_norm


        return xy_conts.sum(), xy_conts.std()



    def loss_on_alternating_group(self, model):
        alternating_indices = [i for i in range(self.group.order) if self.group.signature(i) == 1]
        alternating_data = self.group.get_subset_of_data(alternating_indices).cuda()
        alternating_data, alternating_labels = alternating_data[:, :2], alternating_data[:, 2]
        alternating_logits = model(alternating_data)
        alternating_loss = loss_fn(alternating_logits, alternating_labels).item()
        return alternating_loss

    def loss_all(self, all_logits):
        loss = loss_fn(all_logits, self.all_labels).item()
        return loss






class CyclicMetrics(Metrics):
    def __init__(self, group, training, track_metrics, train_labels=None, test_data=None, test_labels=None, cfg=None):
        super().__init__(group, training, track_metrics, train_labels, test_data, test_labels)
        self.key_freqs = cfg['key_freqs']

    def get_metrics(self, model, train_logits=None, train_loss=None):
        test_logits, all_logits = self.get_test_all_logits(model)
        metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        if self.track_metrics:
           embedding_percent_list, metrics['percent_embeddings_explained_by_key_freqs'] = self.percentage_embeddings_explained_by_key_freqs(model, self.key_freqs)
           for i, freq in enumerate(self.key_freqs):
                metrics[f'percent_embeddings_explained_by_freq_{freq}'] = embedding_percent_list[i]
        return metrics

    def percentage_embeddings_explained_by_key_freqs(self, model, key_freqs):
        W_x_norm_embeddings = (model.W_x.T @ self.group.fourier_basis.T).pow(2).sum(0)
        W_y_norm_embeddings = (model.W_y.T @ self.group.fourier_basis.T).pow(2).sum(0)
        W_U_norm_embeddings = (model.W_U @ self.group.fourier_basis.T).pow(2).sum(0)
        W_x_norm_embedding_total = W_x_norm_embeddings.sum()
        W_y_norm_embedding_total = W_y_norm_embeddings.sum()
        W_U_norm_embedding_total = W_U_norm_embeddings.sum()
        total = W_x_norm_embedding_total + W_y_norm_embedding_total + W_U_norm_embedding_total
        percent_embed_by_key_freq = []
        percent_embed_all_freqs = 0
        for freq in key_freqs:
            x = W_x_norm_embeddings[2*freq-1] + W_x_norm_embeddings[2*freq]
            x += W_y_norm_embeddings[2*freq-1] + W_y_norm_embeddings[2*freq]
            x += W_U_norm_embeddings[2*freq-1] + W_U_norm_embeddings[2*freq]
            percent_embed_by_key_freq.append((x/total).item())
            percent_embed_all_freqs += (x/total).item()
        return percent_embed_by_key_freq, percent_embed_all_freqs





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




