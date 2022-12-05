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
            'trivial': self.group.trivial_rep_trace_tensor_cubes,
            'sign': self.group.sign_rep_trace_tensor_cubes,
            'standard': self.group.standard_rep_trace_tensor_cubes,
            'standard_sign': self.group.standard_sign_rep_trace_tensor_cubes
        }

        self.orth_reps = {
            'trivial': self.group.trivial_reps_orth,
            'sign': self.group.sign_reps_orth,
            'standard': self.group.standard_reps_orth,
            'standard_sign': self.group.standard_sign_reps_orth,
        }



        if self.group.index == 4:
            self.reps['s4_2d'] = self.group.S4_2d_reps
            self.orth_reps['s4_2d'] = self.group.S4_2d_reps_orth


    def get_metrics(self, model, train_logits=None, train_loss=None):
        metrics = {}



        if self.training:
            test_logits, all_logits = self.get_test_all_logits(model)
            metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        else:
            all_logits = model(self.all_data)



        if self.track_metrics:

            # losses
            metrics['alternating_loss'] = self.loss_on_alternating_group(model)
            metrics['all_loss'] = self.loss_all(all_logits)

            # reps
            for rep_name in self.reps.keys():
                metrics[f'logit_{rep_name}_trace_similarity'] = self.logit_trace_similarity(all_logits, self.rep_trace_tensor_cubes[rep_name])
                metrics[f'percent_total_embed_{rep_name}_rep'] = self.percent_total_embed(all_logits, self.orth_reps[rep_name])
                metrics[f'percent_unembed_{rep_name}_rep'] = self.percent_unembed(all_logits, self.orth_reps[rep_name])
                metrics[f'percent_hidden_{rep_name}_rep'] = self.percent_hidden(all_logits, self.orth_reps[rep_name])
            
        return metrics

    def logit_trace_similarity(self, logits, trace_cube):
        logits = logits.reshape(self.group.order, self.group.order, -1)
        centred_logits = logits - logits.mean(-1, keepdim=True)
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

    def unembeddings_explained_by_rep(self, model, orth_reps):
        total_norm_U = model.W_U.pow(2).sum()
        dims = orth_reps.shape[1]
        conts = torch.zeros(dims).cuda()
        for i in range(dims):
            x = orth_reps[:, i].unsqueeze(-1)
            P = self.projection_matrix_general(x)
            proj_U = P @ model.W_U.T 
            proj_U_square = proj_U.pow(2)
            conts[i] = proj_U_square.sum() / total_norm_U
        #assert(torch.allclose(P, P@P, atol=1e-6))
        #assert(torch.allclose(P, P.T, atol=1e-6))
        std = torch.std(conts)
        return conts.sum(), conts.std()

    # def embeddings_explained_by_rep(self, model, orth_reps):
    #     # same as above, but multiply out linear layers
    #     dims = orth_reps.shape[1]
    #     P = self.projection_matrix_general(orth_reps)
    #     #assert(torch.allclose(P, P@P, atol=1e-6))
    #     #assert(torch.allclose(P, P.T, atol=1e-6))
    #     embed_dim = model.W_x.shape[1]
    #     W_x = model.W_x 
    #     W_y = model.W_y 

    #     norm_x = W_x.pow(2).sum()
    #     norm_y = W_y.pow(2).sum()

    #     conts = torch.zeros(dims).cuda()
    #     for i in range(dims):
    #         x = orth_reps[:, i].unsqueeze(-1)
    #         P = self.projection_matrix_general(x)
    #         proj_x = P @ W_x
    #         proj_y = P @ W_y
    #         conts[i] = (proj_x.pow(2).sum() + proj_y.pow(2).sum()) / (norm_x + norm_y)

    #     return conts.sum(), conts.std()

    def total_embeddings_explained_by_rep(self, model, orth_rep):
        # same as above, but multiply out linear layers
        dims = orth_rep.shape[1]
        embed_dim = model.W_x.shape[1]
        total_W_x = model.W_x @ model.W[:embed_dim, :]
        total_W_y = model.W_y @ model.W[embed_dim:, :]

        norm_x = total_W_x.pow(2).sum()
        norm_y = total_W_y.pow(2).sum()

        coefs_x = orth_rep.T @ x_embed
        coefs_y = orth_rep.T @ y_embed

        conts_x = coefs_x.pow(2).sum(-1) / x_norm
        conts_y = coefs_y.pow(2).sum(-1) / y_norm

        return conts_x.sum(), conts_x.std(), conts_y.sum(), conts_y.std()
        
    def hidden_explained_by_rep(self, model, reps):
        hidden_reps_xy = reps[self.all_labels].reshape(self.group.order * self.group.order, -1)
        hidden_reps_xy = torch.linalg.qr(hidden_reps_xy)[0]
        logits, activations = model.run_with_cache(self.all_data, return_cache_object=False)
        hidden = activations['hidden'] - activations['hidden'].mean(0, keepdim=True) # center
        total_norm = hidden.pow(2).sum()

        dims = hidden_reps_xy.shape[1]
        conts = torch.zeros(dims).cuda()
        for i in range(dims):
            x = hidden_reps_xy[:, i].unsqueeze(-1)
            P = self.projection_matrix_general(x)
            proj_hidden = P @ hidden 
            conts[i] = proj_hidden.pow(2).sum() / total_norm

        return conts.sum(), conts.std()



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




