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
        
    def get_metrics(self, model, train_logits=None, train_loss=None):
        metrics = {}
        if self.training:
            test_logits, all_logits = self.get_test_all_logits(model)
            metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)
        else:
            all_logits = model(self.all_data)
        if self.track_metrics:
            metrics['alternating_loss'] = self.loss_on_alternating_group(model)
            metrics['all_loss'] = self.loss_all(model)

            metrics['logit_standard_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.standard_trace_tensor_cubes)
            #metrics['percent_embed_standard_rep'], metrics['norm_embed_standard_rep'], metrics['embed_variance_standard_rep'] = self.embeddings_explained_by_rep(model, self.group.standard_reps_orth)
            metrics['percent_total_embed_standard_rep'], metrics['total_embed_percent_std_standard_rep'] = self.total_embeddings_explained_by_rep(model, self.group.standard_reps_orth)
            metrics['percent_unembed_standard_rep'], metrics['unembed_percent_std_standard_rep'] = self.unembeddings_explained_by_rep(model, self.group.standard_reps_orth)
            metrics['percent_hidden_standard_rep'], metrics['hidden_percent_std_standard_rep'] = self.hidden_explained_by_rep(model, self.group.standard_reps)   

            metrics['logit_standard_sign_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.standard_sign_trace_tensor_cubes)
            #metrics['percent_embed_standard_sign_rep'], metrics['norm_embed_standard_sign_rep'], metrics['embed_variance_standard_sign_rep'] = self.embeddings_explained_by_rep(model, self.group.standard_sign_reps_orth)
            metrics['percent_total_embed_standard_sign_rep'], metrics['total_embed_percent_std_standard_sign_rep'] = self.total_embeddings_explained_by_rep(model, self.group.standard_sign_reps_orth)
            metrics['percent_unembed_standard_sign_rep'], metrics['unembed_percent_std_standard_sign_rep'] = self.unembeddings_explained_by_rep(model, self.group.standard_sign_reps_orth)
            metrics['percent_hidden_standard_sign_rep'], metrics['hidden_percent_std_standard_sign_rep'] = self.hidden_explained_by_rep(model, self.group.standard_sign_reps)

            metrics['logit_sign_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.sign_trace_tensor_cubes)
            #metrics['percent_embed_sign_rep'], metrics['norm_embed_sign_rep'], metrics['embed_variance_sign_rep'] = self.embeddings_explained_by_rep(model, self.group.sign_reps_orth)
            metrics['percent_total_embed_sign_rep'], metrics['total_embed_percent_std_sign_rep'] = self.total_embeddings_explained_by_rep(model, self.group.sign_reps_orth)
            metrics['percent_unembed_sign_rep'], metrics['unembed_percent_std_sign_rep'] = self.unembeddings_explained_by_rep(model, self.group.sign_reps_orth)
            metrics['percent_hidden_sign_rep'], metrics['hidden_percent_std_sign_rep'] = self.hidden_explained_by_rep(model, self.group.sign_reps)

            metrics['logit_trivial_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.trivial_trace_tensor_cubes)
            #metrics['percent_embed_trivial_rep'],  metrics['norm_embed_trivial_rep'], metrics['embed_variance_trivial_rep']  = self.embeddings_explained_by_rep(model, self.group.trivial_reps_orth)
            metrics['percent_total_embed_trivial_rep'], metrics['total_embed_percent_std_trivial_rep'] = self.total_embeddings_explained_by_rep(model, self.group.trivial_reps_orth)
            metrics['percent_unembed_trivial_rep'], metrics['unembed_percent_std_trivial_rep'] = self.unembeddings_explained_by_rep(model, self.group.trivial_reps_orth)
            metrics['percent_hidden_trivial_rep'], metrics['hidden_percent_std_trivial_rep'] = self.hidden_explained_by_rep(model, self.group.trivial_reps)
            
            
            if self.group.index == 4:
                metrics['logit_S4_2d_rep_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.S4_2d_trace_tensor_cubes)
                #metrics['percent_embed_S4_2d_rep'],  metrics['norm_embed_S4_2d_rep'], metrics['embed_variance_S4_2d_rep'] = self.embeddings_explained_by_rep(model, self.group.S4_2d_reps_orth)
                metrics['percent_total_embed_S4_2d_rep'], metrics['total_embed_percent_std_S4_2d_rep'] = self.total_embeddings_explained_by_rep(model, self.group.S4_2d_reps_orth)
                metrics['percent_unembed_S4_2d_rep'], metrics['unembed_percent_std_S4_2d_rep'] = self.unembeddings_explained_by_rep(model, self.group.S4_2d_reps_orth)
                metrics['percent_hidden_S4_2d_rep'],  metrics['hidden_percent_std_S4_2d_rep'] = self.hidden_explained_by_rep(model, self.group.S4_2d_reps)             
            #metrics['logit_natural_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.natural_trace_tensor_cubes)
            #metrics['percentage_embeddings_explained_by_natural_rep'] = self.percentage_embeddings_explained_by_rep(model, self.group.natural_reps_orth)
            
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
        conts = torch.tensor(conts)
        #assert(torch.allclose(P, P@P, atol=1e-6))
        #assert(torch.allclose(P, P.T, atol=1e-6))
        std = torch.std(conts)
        return conts.sum(), conts.std()

    #def embeddings_explained_by_rep(self, model, orth_reps):
        # DONT USE THIS

     #   P = self.projection_matrix_general(orth_reps)
        #assert(torch.allclose(P, P@P, atol=1e-6))
        #assert(torch.allclose(P, P.T, atol=1e-6))

      #  proj_a = P @ model.W_x
       # proj_b = P @ model.W_y
        #embed_num = proj_a.pow(2).sum() + proj_b.pow(2).sum()
        #embed_den = model.W_x.pow(2).sum() + model.W_y.pow(2).sum()
        #percent_embed = embed_num/embed_den
        #std = torch.std(torch.stack((proj_a.pow(2), proj_b.pow(2))).sum(1)) # sum over rows should be similar if each rep direction is used
        #std_mean = std/embed_num    
        #return percent_embed, embed_num, std

    def total_embeddings_explained_by_rep(self, model, orth_reps):
        # same as above, but multiply out linear layers
        dims = orth_reps.shape[1]
        P = self.projection_matrix_general(orth_reps)
        #assert(torch.allclose(P, P@P, atol=1e-6))
        #assert(torch.allclose(P, P.T, atol=1e-6))
        embed_dim = model.W_x.shape[1]
        total_W_x = model.W_x @ model.W[:embed_dim, :]
        total_W_y = model.W_y @ model.W[embed_dim:, :]

        norm_x = total_W_x.pow(2).sum()
        norm_y = total_W_y.pow(2).sum()

        conts = torch.zeros(dims).cuda()
        for i in range(dims):
            x = orth_reps[:, i].unsqueeze(-1)
            P = self.projection_matrix_general(x)
            proj_x = P @ total_W_x
            proj_y = P @ total_W_y
            conts[i] = (proj_x.pow(2).sum() + proj_y.pow(2).sum()) / (norm_x + norm_y)

        return conts.sum(), conts.std()
        
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

    def loss_all(self, model):
        logits = model(self.all_data)
        loss = loss_fn(logits, self.all_labels).item()
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




