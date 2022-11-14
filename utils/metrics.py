from utils.plotting import fft2d
from utils.models import *
import torch

class Metrics():
    def __init__(self, group, train_labels, test_data, test_labels, track_metrics):
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.group = group
        self.track_metrics = track_metrics
        self.all_data = group.get_all_data()[:, :2]
    
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
    def __init__(self, group, train_labels, test_data, test_labels, track_metrics, cfg):
        super().__init__(group, train_labels, test_data, test_labels, track_metrics)
        
    def get_metrics(self, model, train_logits, train_loss):
        test_logits, all_logits = self.get_test_all_logits(model)
        metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        if self.track_metrics:
            metrics['logit_standard_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.standard_trace_tensor_cubes)
            metrics['logit_standard_sign_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.standard_sign_trace_tensor_cubes)
            metrics['percentage_embeddings_explained_by_standard_rep'] = self.percentage_embeddings_explained_by_rep(model, self.group.standard_reps_orth)
            metrics['percentage_embeddings_explained_by_standard_sign_rep'] = self.percentage_embeddings_explained_by_rep(model, self.group.product_standard_sign_reps_orth)

        return metrics

    def logit_trace_similarity(self, logits, trace_cube):
        logits = logits.reshape(self.group.order, self.group.order, -1)
        centred_logits = logits - logits.mean(-1)
        centred_logits = centred_logits.reshape(self.group.order*self.group.order,-1)
        centred_trace = trace_cube.reshape(self.group.order*self.group.order,-1)
        sims = F.cosine_similarity(centred_logits, centred_trace, dim=-1)
        return sims.mean().item()

    def projection_matrix_general(self, B):
            """Compute the projection matrix onto the space spanned by the columns of `B`
            Args:
                B: ndarray of dimension (D, M), the basis for the subspace
            
            Returns:
                P: the projection matrix
            """
            P = B @ (B.T @ B).inverse() @ B.T
            return P

    def percentage_embeddings_explained_by_rep(self, model, orth_reps):

        P= self.projection_matrix_general(orth_reps)
        assert(torch.allclose(P, P@P, atol=1e-6))
        assert(torch.allclose(P, P.T, atol=1e-6))

        proj_a = P @ model.W_E_a
        proj_b = P @ model.W_E_b
        proj_U = P @ model.W_U.T

        num = proj_a.pow(2).sum() + proj_b.pow(2).sum() + proj_U.pow(2).sum()
        denom = model.W_E_a.pow(2).sum() + model.W_E_b.pow(2).sum() + model.W_U.pow(2).sum()

        return (num/denom).item()





class CyclicMetrics(Metrics):
    def __init__(self, group, train_labels, test_data, test_labels, track_metrics, cfg):
        super().__init__(group, train_labels, test_data, test_labels, track_metrics)
        self.key_freqs = cfg['key_freqs']

    def get_metrics(self, model, train_logits, train_loss):
        test_logits, all_logits = self.get_test_all_logits(model)
        metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        if self.track_metrics:
           embedding_percent_list, metrics['percent_embeddings_explained_by_key_freqs'] = self.percentage_embeddings_explained_by_key_freqs(model, self.key_freqs)
           for i, freq in enumerate(self.key_freqs):
                metrics[f'percent_embeddings_explained_by_freq_{freq}'] = embedding_percent_list[i]
        return metrics

    def percentage_embeddings_explained_by_key_freqs(self, model, key_freqs):
        W_E_a_norm_embeddings = (model.W_E_a.T @ self.group.fourier_basis.T).pow(2).sum(0)
        W_E_b_norm_embeddings = (model.W_E_b.T @ self.group.fourier_basis.T).pow(2).sum(0)
        W_U_norm_embeddings = (model.W_U @ self.group.fourier_basis.T).pow(2).sum(0)
        W_E_a_norm_embedding_total = W_E_a_norm_embeddings.sum()
        W_E_b_norm_embedding_total = W_E_b_norm_embeddings.sum()
        W_U_norm_embedding_total = W_U_norm_embeddings.sum()
        total = W_E_a_norm_embedding_total + W_E_b_norm_embedding_total + W_U_norm_embedding_total
        percent_embed_by_key_freq = []
        percent_embed_all_freqs = 0
        for freq in key_freqs:
            x = W_E_a_norm_embeddings[2*freq-1] + W_E_a_norm_embeddings[2*freq]
            x += W_E_b_norm_embeddings[2*freq-1] + W_E_b_norm_embeddings[2*freq]
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




