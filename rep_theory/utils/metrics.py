from utils.plotting import fft2d
from utils.models import loss_fn
import torch
import torch.nn.functional as F
import sys

class Metrics():
    """
    A class to track metrics during training and testing.
    """
    def __init__(self, group, training, track_metrics, train_data = None, train_labels=None, test_data=None, test_labels=None, shuffled_indices=None, cfg={}, only_x_embed=False):
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
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.group = group
        self.track_metrics = track_metrics
        all_data, _ = group.get_all_data()
        self.all_data = all_data[:, :2]
        self.all_labels = all_data[:, 2]
        self.cfg = cfg
        self.shuffled_indices = shuffled_indices
        self.no_internals = False
        self.only_x_embed = only_x_embed 

        # get the indices of train data in all data
        #values, indices = torch.topk(((self.all_data.t() == self.train_data.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
        #indices = indices[values!=0]
        #self.train_indices = indices
        self.train_indices = self.shuffled_indices[:len(self.train_data)]

        if track_metrics:
            for rep_name in self.group.irreps.keys():
                self.group.irreps[rep_name].hidden_reps_x = self.group.irreps[rep_name].rep[self.all_data[:, 0]].reshape(self.group.order**2, -1)
                self.group.irreps[rep_name].hidden_reps_y = self.group.irreps[rep_name].rep[self.all_data[:, 1]].reshape(self.group.order**2, -1)
                self.group.irreps[rep_name].hidden_reps_xy = self.group.irreps[rep_name].rep[self.all_labels].reshape(self.group.order**2, -1)
                self.group.irreps[rep_name].hidden_reps_x_orth = torch.linalg.qr(self.group.irreps[rep_name].hidden_reps_x)[0]
                self.group.irreps[rep_name].hidden_reps_y_orth = torch.linalg.qr(self.group.irreps[rep_name].hidden_reps_y)[0]
                self.group.irreps[rep_name].hidden_reps_xy_orth = torch.linalg.qr(self.group.irreps[rep_name].hidden_reps_xy)[0]

    


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



        

    def determine_key_reps(self, model):
        """
        Determine via logit similarity of a final model which reps are key.
        """
        self.cfg['key_reps'] = []
        all_logits = self.get_all_logits(model)
        for rep_name in self.group.non_trivial_irreps.keys():
            if self.logit_trace_similarity(all_logits, self.group.irreps[rep_name].logit_trace_tensor_cube) > 0.005:
                self.cfg['key_reps'].append(rep_name)
        return self.cfg['key_reps']

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
            test_logits = self.get_test_logits(model)
            metrics = self.get_standard_metrics(test_logits, train_logits, train_loss)

        if self.only_x_embed:
            for rep_name in self.group.irreps.keys():
                metrics[f'percent_x_embed_{rep_name}_rep'] = self.percent_total_embed(model, self.group.irreps[rep_name].orth_rep)[0]
            return metrics

        if self.track_metrics:
            # losses
            all_logits = self.get_all_logits(model)
            metrics['all_loss'] = self.loss_all(all_logits)

            # totals
            percent_logits_explained = 0
            percent_hidden_explained = 0
            percent_hidden_xy_explained = 0
            percent_x_embed_explained = 0
            percent_y_embed_explained = 0
            percent_unembed_explained = 0
            sims={}

            # all reps
            for rep_name in self.group.irreps.keys():
                if rep_name != 'trivial':
                    sim = self.logit_trace_similarity(all_logits, self.group.irreps[rep_name].logit_trace_tensor_cube)
                    percent_logits_explained += sim**2
                    sims[rep_name] = sim
                    metrics[f'logit_{rep_name}_rep_trace_similarity'] = sim
                    metrics[f'logit_excluded_loss_{rep_name}_rep'], metrics[f'logit_restricted_loss_{rep_name}_rep'] = self.logit_excluded_and_restricted_loss(all_logits, sim, self.group.irreps[rep_name].logit_trace_tensor_cube)
                
                if not self.no_internals:
                    metrics[f'percent_x_embed_{rep_name}_rep'], metrics[f'percent_y_embed_{rep_name}_rep'] = self.percent_total_embed(model, self.group.irreps[rep_name].orth_rep)
                    metrics[f'percent_unembed_{rep_name}_rep']  = self.percent_unembed(model, self.group.irreps[rep_name].orth_rep)
                    metrics[f'percent_hidden_x_{rep_name}_rep'], metrics[f'percent_hidden_y_{rep_name}_rep'], metrics[f'percent_hidden_xy_{rep_name}_rep'], metrics[f'total_percent_hidden_{rep_name}_rep']= self.percent_hidden(model, rep_name)
                    metrics[f'hidden_excluded_loss_{rep_name}_rep'], metrics[f'hidden_restricted_loss_{rep_name}_rep'] = self.hidden_excluded_and_restricted_loss(model, self.group.irreps[rep_name].hidden_reps_xy_orth)

            # total 
            metrics['percent_logits_explained'] = percent_logits_explained
            metrics['total_logit_excluded_loss'], metrics['total_logit_restricted_loss'] = self.total_logit_excluded_and_restricted_loss(all_logits, self.cfg['key_reps'], sims)
            if not self.no_internals:
                metrics['total_hidden_excluded_loss'], metrics['total_hidden_restricted_loss'] = self.total_hidden_excluded_and_restricted_loss(model, self.cfg['key_reps'])
                metrics['test_loss_restricted_loss_ratio'] = metrics['test_loss']/metrics['total_hidden_restricted_loss']
                metrics['sum_of_squared_weights'] = self.sum_of_squared_weights(model)

            # key reps
            for rep_name in self.cfg['key_reps']:
                percent_hidden_explained += metrics[f'total_percent_hidden_{rep_name}_rep']
                percent_hidden_xy_explained += metrics[f'percent_hidden_xy_{rep_name}_rep']
                percent_x_embed_explained += metrics[f'percent_x_embed_{rep_name}_rep']
                percent_y_embed_explained += metrics[f'percent_y_embed_{rep_name}_rep']
                percent_unembed_explained += metrics[f'percent_unembed_{rep_name}_rep']

            metrics['percent_hidden_explained'] = percent_hidden_explained
            metrics['percent_hidden_xy_explained'] = percent_hidden_xy_explained
            metrics['percent_x_embed_explained'] = percent_x_embed_explained
            metrics['percent_y_embed_explained'] = percent_y_embed_explained
            metrics['percent_unembed_explained'] = percent_unembed_explained        

        return metrics


    def get_hidden(self, model):
        """ 
        Get the final MLP neuron activations for all data points
        """
        if 'OneLayerMLP' in model.__class__.__name__ :
            logits, activations = model.run_with_cache(self.all_data, return_cache_object=False)
            hidden = activations['hidden'] 
        elif model.__class__.__name__ == 'Transformer':
            # if all data is big, split into n sections, and stitch back together
            if self.all_data.shape[0] <= 15000:
                logits, activations = model.run_with_cache(self.all_data)
                hidden = activations["post", 0, "mlp"][:, -1, :]
            else:
                shape = self.all_data.shape[0]
                n = 4
                hidden = []
                for i in range(n):
                    logits, activations = model.run_with_cache(self.all_data[i*shape//n:(i+1)*shape//n])
                    hidden.append(activations["post", 0, "mlp"][:, -1, :])
                hidden = torch.cat(hidden, dim=0)
        return hidden


    def get_embeds(self, model):
        """ 
        Get the embedding matrices for x and y
        """
        if 'OneLayerMLP' in model.__class__.__name__:
            embeds = model.W_x @ model.W[:model.embed_dim, :], model.W_y @ model.W[model.embed_dim:, :]
        elif model.__class__.__name__ == 'Transformer':
            embeds = model.embed.W_E[:-1], model.embed.W_E[:-1]
        return embeds

    def get_unembed(self, model):
        """ 
        Get the unembedding matrix
        """
        if 'OneLayerMLP' in model.__class__.__name__:
            unembed = model.W_U
        elif model.__class__.__name__ == 'Transformer':
            unembed = model.blocks[0].mlp.W_out @ model.unembed.W_U
        return unembed


    def hidden_to_logits(self, hidden, model):
        """ 
        Convert hidden activations to logits via the correct unembed
        """
        if 'OneLayerMLP' in model.__class__.__name__:
            return hidden @ model.W_U
        elif model.__class__.__name__ == 'Transformer':
            return hidden @ model.blocks[0].mlp.W_out @ model.unembed.W_U
        else:
            raise NotImplementedError

    def logit_trace_similarity(self, logits, trace_cube):
        """
        Compute cosine similarity between true logits and logits computed via tr(\rho(x)\rho(y)\rho(z^-1))

        Args:
            logits (torch.tensor): (batch, group.order) tensor of logits
            trace_cube (torch.tensor): (group.order, group.order, group.order) tensor of tr(\rho(x)\rho(y)\rho(z^-1))

        Returns:
            float: mean cosine similarity over batch
        """
        centered_logits = logits - logits.mean(dim=-1, keepdim=True)
        centered_logits = centered_logits.reshape(-1)
        trace = trace_cube.reshape(-1)
        sim = F.cosine_similarity(centered_logits, trace, dim=0)
        return sim

    def logit_excluded_and_restricted_loss(self, logits, sim, cube):
        """ 
        Restrict at the logit level by excluding and restricting to individual characters - this metric is not used in the paper
        """
        centered_logits = logits - logits.mean(dim=-1, keepdim=True)
        centered_logits = centered_logits.reshape(-1)
        trace = cube.reshape(-1)
        unit_trace = trace / torch.norm(trace)
        norm_logits = torch.norm(centered_logits)
        excluded_logits = centered_logits - sim * norm_logits * unit_trace
        restricted_logits = sim * norm_logits * unit_trace
        excluded_logits = excluded_logits.reshape(-1, self.group.order)
        excluded_logits = excluded_logits[self.train_indices]
        excluded_loss = loss_fn(excluded_logits, self.train_labels).item()
        restricted_loss = loss_fn(restricted_logits.reshape(-1, self.group.order), self.all_labels).item()
        return excluded_loss, restricted_loss

    def total_logit_excluded_and_restricted_loss(self, logits, key_reps, sims):
        """ 
        Restrict at the logit level by excluding and restricting to key characters - this metric is not used in the paper
        """
        centered_logits = logits - logits.mean(dim=-1, keepdim=True)
        centered_logits = centered_logits.reshape(-1)
        norm_logits = torch.norm(centered_logits)
        
        key_rep_logits = torch.zeros_like(centered_logits)
        for rep_name in key_reps:
            trace = self.group.irreps[rep_name].logit_trace_tensor_cube.reshape(-1)
            unit_trace = trace / torch.norm(trace)

            key_rep_logits += sims[rep_name] * norm_logits * unit_trace

        excluded_logits = centered_logits - key_rep_logits
        excluded_logits = excluded_logits.reshape(-1, self.group.order)
        excluded_logits = excluded_logits[self.train_indices]
        restricted_logits = key_rep_logits

        excluded_loss = loss_fn(excluded_logits, self.train_labels).item()
        restricted_loss = loss_fn(restricted_logits.reshape(-1, self.group.order), self.all_labels).item()

        return excluded_loss, restricted_loss

    def percent_unembed(self, model, orth_rep):
        """
        Compute the percent of the unembed represented by the representation.

        Args:
            model (nn.Module): neural network
            orth_rep (torch.tensor): orthonormal representation

        Returns:
            (float, float): (total percent, standard deviation over matrix elements of orthonomal representation)

        """
        W_U = self.get_unembed(model)
        norm_U = W_U.pow(2).sum()
        coefs_U = orth_rep.T @ W_U.T
        conts_U = coefs_U.pow(2).sum(-1) / norm_U
        return conts_U.sum()

    def percent_total_embed(self, model, orth_rep):
        """
        Compute the percent of the total embedding represented by the representation. Total embedding is the matmul of the embedding and the linear layer.

        Args:
            model (nn.Module): neural network
            orth_rep (torch.tensor): orthonormal representation

        Returns:
            (float, float, float, float): (total percent x, total percent y)
        """
        x_embed, y_embed = self.get_embeds(model)

        norm_x = x_embed.pow(2).sum()
        norm_y = y_embed.pow(2).sum()

        coefs_x = orth_rep.T @ x_embed
        coefs_y = orth_rep.T @ y_embed

        conts_x = coefs_x.pow(2).sum(-1) / norm_x
        conts_y = coefs_y.pow(2).sum(-1) / norm_y

        return conts_x.sum(), conts_y.sum()
        
    def percent_hidden(self, model, rep_name):
        """
        Compute the percent of the total hidden representation represented by the representation matrices \rho(xy).

        Args:
            model (nn.Module): neural network
            hidden_reps_xy (torch.tensor): orthonormal hidden representations \rho(xy)

        Returns:
            (float, float): (total percent, standard deviation over matrix elements of orthonomal representation)
        """
        hidden = self.get_hidden(model)
        hidden = hidden - hidden.mean(dim=0, keepdim=True)

        hidden_norm = hidden.pow(2).sum()

        hidden_reps_x_orth = self.group.irreps[rep_name].hidden_reps_x_orth
        hidden_reps_y_orth = self.group.irreps[rep_name].hidden_reps_y_orth
        hidden_reps_xy_orth = self.group.irreps[rep_name].hidden_reps_xy_orth
        coefs_x = hidden_reps_x_orth.T @ hidden
        coefs_y = hidden_reps_y_orth.T @ hidden
        coefs_xy = hidden_reps_xy_orth.T @ hidden
        x_conts = coefs_x.pow(2).sum() / hidden_norm
        y_conts = coefs_y.pow(2).sum() / hidden_norm
        xy_conts = coefs_xy.pow(2).sum() / hidden_norm
        total_conts = x_conts + y_conts + xy_conts

        return x_conts, y_conts, xy_conts, total_conts

    def hidden_excluded_and_restricted_loss(self, model, hidden_reps_xy_orth):
        """ 
        Restrict or exclude reps rho(ab) from the hidden layer and compute the loss on the restricted and excluded parts of the hidden layer.
        """
        hidden = self.get_hidden(model)
        
        coefs_xy = hidden_reps_xy_orth.T @ hidden
        hidden_xy = hidden_reps_xy_orth @ coefs_xy

        hidden_xy_restricted = hidden_xy
        hidden_xy_excluded = hidden - hidden_xy

        logits_restricted = self.hidden_to_logits(hidden_xy_restricted, model)
        logits_excluded = self.hidden_to_logits(hidden_xy_excluded, model)

        restricted_loss = loss_fn(logits_restricted, self.all_labels).item()
        excluded_loss = loss_fn(logits_excluded[self.train_indices], self.train_labels).item()

        return excluded_loss, restricted_loss

    
    def total_hidden_excluded_and_restricted_loss(self, model, key_reps):
        """ 
        Restrict or exclude all key reps rho(ab) from the hidden layer and compute the loss on the restricted and excluded parts of the hidden layer.
        """
        hidden = self.get_hidden(model)

        hidden_restricted = torch.zeros_like(hidden)
        for rep_name in key_reps:
            coefs_xy = self.group.irreps[rep_name].hidden_reps_xy_orth.T @ hidden
            hidden_xy = self.group.irreps[rep_name].hidden_reps_xy_orth @ coefs_xy
            hidden_restricted += hidden_xy

        hidden_excluded = hidden - hidden_restricted

        logits_restricted = self.hidden_to_logits(hidden_restricted, model)
        logits_excluded = self.hidden_to_logits(hidden_excluded, model)

        restricted_loss = loss_fn(logits_restricted, self.all_labels).item()
        excluded_loss = loss_fn(logits_excluded[self.train_indices], self.train_labels).item()

        return excluded_loss, restricted_loss

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

    def sum_of_squared_weights(self, model):
        """
        Compute the sum of squared weights on the whole model.

        Args:
            model (nn.Module)

        Returns:
            float: sum of squared weights on entire group
        """

        sum_of_square_weights = 0

        if model.__class__.__name__ == "OneLayerMLP":
            sum_of_square_weights += torch.sum(model.W_x**2)
            sum_of_square_weights += torch.sum(model.W_y**2)
            sum_of_square_weights += torch.sum(model.W_U**2)
            # hacky
            if hasattr(model, 'W'):
                sum_of_square_weights += torch.sum(model.W**2)

        elif model.__class__.__name__ == "Transformer":
            for name, param in model.named_parameters():
                if 'weight' in name:
                    sum_of_square_weights += torch.sum(param**2)


        return sum_of_square_weights

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

    def get_all_logits(self, model):
        """
        Compute logits for all data.

        Args:
            model(nn.Module): neural network

        Returns:
            torch.tensor: (batch, group.order) tensor of logits for all data
        """
        return model(self.all_data)
    
    def get_test_logits(self, model):
        """
        Compute logits for test data.

        Args:
            model(nn.Module): neural network

        Returns:
            torch.tensor: (batch, group.order) tensor of logits for test data
        """
        return model(self.test_data)

