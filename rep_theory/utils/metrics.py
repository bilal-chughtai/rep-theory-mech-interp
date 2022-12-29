from utils.plotting import fft2d
from utils.models import loss_fn
import torch
import torch.nn.functional as F

class Metrics():
    """
    A class to track metrics during training and testing.
    """
    def __init__(self, cfg, group, training, track_metrics, train_labels=None, test_data=None, test_labels=None):
        """
        Initialise the metrics class.

        Args:
            cfg (dict): configuration dictionary
            group (Group): group the task is defined on
            training (bool): whether the model is being trained or tested
            track_metrics (bool): whether to track of only basic metrics (loss, accuracy), or also more complex metrics
            train_labels (torch.tensor, optional): labels for training data. Defaults to None.
            test_data (torch.tensor, optional): data for testing. Defaults to None.
            test_labels (torch.tensor, optional): labels for testing data. Defaults to None.
        """
        self.training = training
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.group = group
        self.track_metrics = track_metrics
        all_data = group.get_all_data().cuda()
        self.all_data = all_data[:, :2]
        self.all_labels = all_data[:, 2]
        self.cfg = cfg
    
    def get_accuracy(self, logits, labels):
        """
        Compute accuracy of model.

        Args:
            logits (torch.tensor): (batch, group.order) tensor of logits
            labels (torch.tensor): (batch) tensor of labels

        Returns:
            float: accuracy
        """
        return ((logits.argmax(1)==labels).sum()/len(labels)).item()

    def get_test_all_logits(self, model):
        """
        Compute logits for test and all data.

        Args:
            model(nn.Module): neural network

        Returns:
            tuple: (test_logits, all_logits)
        """
        test_logits = model(self.test_data)
        all_logits = model(self.all_data)
        return test_logits, all_logits

    def get_standard_metrics(self, test_logits, train_logits, train_loss):
        """
        Generate losses and accuracies for test and train data.

        Args:
            test_logits (torch.tensor): (batch, group.order) tensor of logits for test data
            train_logits (torch.tensor): (batch, group.order) tensor of logits for train data
            train_loss (torch.tensor): scalar tensor of loss for train data

        Returns:
            dict: dictionary of metrics
        """
        metrics = {}
        metrics['train_loss'] = train_loss.item()
        metrics['test_loss'] = loss_fn(test_logits, self.test_labels).item()
        metrics['train_acc'] = self.get_accuracy(train_logits, self.train_labels)
        metrics['test_acc'] = self.get_accuracy(test_logits, self.test_labels)
        return metrics


class SymmetricMetrics(Metrics):
    """
    A class to track metrics during training and testing for symmetric group tasks.

    """
    def __init__(self, group, training, track_metrics, train_labels=None, test_data=None, test_labels=None, cfg=None):

        super().__init__(cfg, group, training, track_metrics, train_labels, test_data, test_labels)

        if track_metrics:
            for rep_name in self.group.irreps.keys():
                self.group.irreps[rep_name].hidden_reps_xy = self.group.irreps[rep_name].rep[self.all_labels].reshape(self.group.order*self.group.order, -1)
                self.group.irreps[rep_name].hidden_reps_xy_orth = torch.qr(self.group.irreps[rep_name].hidden_reps_xy)[0]


    def get_metrics(self, model, train_logits=None, train_loss=None):
        """
        Compute metrics for model.

        Args:
            model (nn.Module): neural network
            train_logits (torch.tensor, optional): Logits for training data. Defaults to None.
            train_loss (torch.tensor, optional): Loss for training data. Defaults to None.

        Returns:
            dict: dictionary of metrics
        """
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
            for rep_name in self.group.irreps.keys():
                metrics[f'logit_{rep_name}_rep_trace_similarity'] = self.logit_trace_similarity(all_logits, self.group.irreps[rep_name].logit_trace_tensor_cube)
                metrics[f'percent_x_embed_{rep_name}_rep'], metrics[f'percent_std_x_embed_{rep_name}_rep'], metrics[f'percent_y_embed_{rep_name}_rep'], metrics[f'percent_std_y_embed_{rep_name}_rep']= self.percent_total_embed(model, self.group.irreps[rep_name].orth_rep)
                metrics[f'percent_unembed_{rep_name}_rep'], metrics[f'percent_std_unembed_{rep_name}_rep']  = self.percent_unembed(model, self.group.irreps[rep_name].orth_rep)
                metrics[f'percent_hidden_{rep_name}_rep'], metrics[f'percent_std_hidden_{rep_name}_rep'] = self.percent_hidden(model, self.group.irreps[rep_name].hidden_reps_xy_orth)
                metrics[f'excluded_loss_{rep_name}_rep'] = self.excluded_loss(model, self.group.irreps[rep_name].orth_rep)

            metrics['restricted_loss'] = self.restricted_loss(model, self.cfg['key_reps'])
            metrics['total_excluded_loss'] = self.total_excluded_loss(model, self.cfg['key_reps'])
            metrics['test_loss_restricted_loss_ratio'] = metrics['test_loss']/metrics['restricted_loss']
            metrics['sum_of_squared_weights'] = self.sum_of_squared_weights(model)

        return metrics

    def logit_trace_similarity(self, logits, trace_cube):
        """
        Compute cosine similarity between true logits and logits computed via tr(\rho(x)\rho(y)\rho(z^-1))

        Args:
            logits (torch.tensor): (batch, group.order) tensor of logits
            trace_cube (torch.tensor): (group.order, group.order, group.order) tensor of tr(\rho(x)\rho(y)\rho(z^-1))

        Returns:
            float: mean cosine similarity over batch
        """
        logits = logits.reshape(self.group.order, self.group.order, -1)
        centred_logits = logits - logits.mean(-1, keepdim=True)
        centred_logits = centred_logits.reshape(self.group.order*self.group.order,-1)
        centred_trace = trace_cube.reshape(self.group.order*self.group.order,-1)
        sims = F.cosine_similarity(centred_logits, centred_trace, dim=-1)
        return sims.mean().item()


    def percent_unembed(self, model, orth_rep):
        """
        Compute the percent of the unembed represented by the representation.

        Args:
            model (nn.Module): neural network
            orth_rep (torch.tensor): orthonormal representation

        Returns:
            (float, float): (total percent, standard deviation over matrix elements of orthonomal representation)

        # TODO: the std is not that meaningful here
        """
        norm_U = model.W_U.pow(2).sum()
        coefs_U = orth_rep.T @ model.W_U.T
        conts_U = coefs_U.pow(2).sum(-1) / norm_U
        return conts_U.sum(), conts_U.std()

    def percent_total_embed(self, model, orth_rep):
        """
        Compute the percent of the total embedding represented by the representation. Total embedding is the matmul of the embedding and the linear layer.

        Args:
            model (nn.Module): neural network
            orth_rep (torch.tensor): orthonormal representation

        Returns:
            (float, float, float, float): (total percent x, std x, total percent y, std y)
        """
        embed_dim = model.W_x.shape[1]
        x_embed = model.x_embed
        y_embed = model.y_embed

        norm_x = x_embed.pow(2).sum()
        norm_y = y_embed.pow(2).sum()

        coefs_x = orth_rep.T @ x_embed
        coefs_y = orth_rep.T @ y_embed

        conts_x = coefs_x.pow(2).sum(-1) / norm_x
        conts_y = coefs_y.pow(2).sum(-1) / norm_y

        return conts_x.sum(), conts_x.std(), conts_y.sum(), conts_y.std()
        
    def percent_hidden(self, model, hidden_reps_xy_orth):
        """
        Compute the percent of the total hidden representation represented by the representation matrices \rho(xy).

        Args:
            model (nn.Module): neural network
            hidden_reps_xy (torch.tensor): orthonormal hidden representations \rho(xy)

        Returns:
            (float, float): (total percent, standard deviation over matrix elements of orthonomal representation)
        """
        logits, activations = model.run_with_cache(self.all_data, return_cache_object=False)
        hidden = activations['hidden'] # DONT CENTER - activations['hidden'].mean(0, keepdim=True) # center
        hidden_norm = hidden.pow(2).sum()

        coefs_xy = hidden_reps_xy_orth.T @ hidden
        xy_conts = coefs_xy.pow(2).sum(-1) / hidden_norm

        return xy_conts.sum(), xy_conts.std()


    def loss_on_alternating_group(self, model):
        """
        Compute the loss on the alternating group.

        Args:
            model (nn.Module): neural network

        Returns:
            float: loss on alternating group
        """
        alternating_indices = [i for i in range(self.group.order) if self.group.signature(i) == 1]
        alternating_data = self.group.get_subset_of_data(alternating_indices).cuda()
        alternating_data, alternating_labels = alternating_data[:, :2], alternating_data[:, 2]
        alternating_logits = model(alternating_data)
        alternating_loss = loss_fn(alternating_logits, alternating_labels).item()
        return alternating_loss

    def loss_all(self, all_logits):
        """
        Compute the loss on the entire group.

        Args:
            all_logits (torch.tensor): (batch, group.order) tensor of logits

        Returns:
            float: loss on entire group
        """
        loss = loss_fn(all_logits, self.all_labels).item()
        return loss
    
    def compute_hidden_from_embeds(self, x_embed, y_embed, model):
        """
        Compute hidden layer on entire distribution from total embeddings.
        """
        x_embed = x_embed.unsqueeze(1)
        y_embed = y_embed.unsqueeze(0)

        if model.__class__.__name__ == "OneLayerMLP":
            hidden = torch.nn.ReLU()((x_embed + y_embed).reshape(self.group.order**2, -1))
        elif model.__class__.__name__  == "BilinearNet":
            hidden = (x_embed * y_embed).reshape(self.group.order**2, -1)
        
        return hidden



    def excluded_loss(self, model, orth_rep):
        """
        Compute the loss having ablated one of the representations.

        Args:
            orth_rep (torch.tensor): orthonormal representation

        Returns:
            float: loss on entire group with one representation ablated
        """
        # TODO: make this not architecture specific

        x_embed = model.x_embed
        y_embed = model.y_embed

        coefs_x = orth_rep.T @ x_embed
        coefs_y = orth_rep.T @ y_embed

        x_embed_cont = orth_rep @ coefs_x
        y_embed_cont = orth_rep @ coefs_y

        x_embed_excluded = x_embed - x_embed_cont
        y_embed_excluded = y_embed - y_embed_cont

        hidden = self.compute_hidden_from_embeds(x_embed_excluded, y_embed_excluded, model)
    
        logits = hidden @ model.W_U 

        loss = loss_fn(logits, self.all_labels).item()

        return loss
    
    def restricted_loss(self, model, key_reps):
        """
        Compute the loss having restricted to only the representations the network learns.

        Args:
            key_reps (list): list of names of representations to restrict to

        Returns:
            float: loss on entire group restricting only to the key representations
        """
        # TODO: make this not architecture specific


        x_embed = model.x_embed
        y_embed = model.y_embed
        
        x_embed_restricted = torch.zeros_like(x_embed)
        y_embed_restricted = torch.zeros_like(y_embed)

        for key_rep in key_reps:
            orth_rep = self.group.irreps[key_rep].orth_rep

            coefs_x = orth_rep.T @ x_embed
            coefs_y = orth_rep.T @ y_embed

            x_embed_cont = orth_rep @ coefs_x
            y_embed_cont = orth_rep @ coefs_y

            x_embed_restricted += x_embed_cont
            y_embed_restricted += y_embed_cont

        x_embed_restricted = x_embed_restricted.unsqueeze(1) #(group.order, 1, hidden)
        y_embed_restricted = y_embed_restricted.unsqueeze(0) # (1, group.order, hidden)

        hidden = self.compute_hidden_from_embeds(x_embed_restricted, y_embed_restricted, model)

        logits = hidden @ model.W_U

        loss = loss_fn(logits, self.all_labels).item()
        return loss

    def total_excluded_loss(self, model, key_reps):

        x_embed = model.x_embed
        y_embed = model.y_embed
        
        x_embed_excluded = x_embed
        y_embed_excluded = y_embed

        for key_rep in key_reps:
            orth_rep = self.group.irreps[key_rep].orth_rep

            coefs_x = orth_rep.T @ x_embed
            coefs_y = orth_rep.T @ y_embed

            x_embed_cont = orth_rep @ coefs_x
            y_embed_cont = orth_rep @ coefs_y

            x_embed_excluded -= x_embed_cont
            y_embed_excluded -= y_embed_cont

        x_embed_excluded = x_embed_excluded.unsqueeze(1) #(group.order, 1, hidden)
        y_embed_excluded = y_embed_excluded.unsqueeze(0) # (1, group.order, hidden)

        hidden = self.compute_hidden_from_embeds(x_embed_excluded, y_embed_excluded, model)
        logits = hidden @ model.W_U

        loss = loss_fn(logits, self.all_labels).item()
        return loss

    def sum_of_squared_weights(self, model):
        """
        Compute the sum of squared weights on the whole model.

        Args:
            model (nn.Module)

        Returns:
            float: sum of squared weights on entire group
        """

        sum_of_square_weights = 0
        sum_of_square_weights += torch.sum(model.W_x**2)
        sum_of_square_weights += torch.sum(model.W_y**2)
        sum_of_square_weights += torch.sum(model.W_U**2)

        # hacky
        if hasattr(model, 'W'):
            sum_of_square_weights += torch.sum(model.W**2)

        return sum_of_square_weights






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




