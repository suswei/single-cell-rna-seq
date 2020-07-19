import collections
from typing import Iterable
import scipy
import torch
from torch import nn as nn
from torch.distributions import Normal

from scvi.models.utils import one_hot

import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_in: int, n_out: int, n_cat_list: Iterable[int] = None,
                 n_layers: int = 1, n_hidden: int = 128, dropout_rate: float = 0.1, use_batch_norm=True):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Linear(n_in + sum(self.n_cat_list), n_out),
                nn.BatchNorm1d(n_out, momentum=.01, eps=0.001) if use_batch_norm else None,
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None))
             for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                                    for o in one_hot_cat_list]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                 n_cat_list: Iterable[int] = None, n_layers: int = 1,
                 n_hidden: int = 128, dropout_rate: float = 0.1):
        super().__init__()

        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))  # (computational stability safeguard)torch.clamp(, -5, 5)
        latent = self.reparameterize(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                 n_cat_list: Iterable[int] = None, n_layers: int = 1,
                 n_hidden: int = 128):
        super().__init__()
        self.px_decoder = FCLayers(n_in=n_input, n_out=n_hidden,
                                   n_cat_list=n_cat_list, n_layers=n_layers,
                                   n_hidden=n_hidden, dropout_rate=0)

        # mean gamma
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor,
                *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int, n_cat_list: Iterable[int] = None, n_layers: int = 1,
                 n_hidden: int = 128):
        super().__init__()
        self.decoder = FCLayers(n_in=n_input, n_out=n_hidden,
                                n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=0)

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v

class MINE_Net(nn.Module):
    def __init__(self, xy_dim, n_latents, unbiased_loss):
        super(MINE_Net, self).__init__()
        self.xy_dim = xy_dim
        self.unbiased_loss = unbiased_loss

        modules = [nn.Linear(xy_dim, n_latents[0]), nn.ReLU()]

        prev_layer = n_latents[0]
        for layer in n_latents[1:]:
            modules.append(nn.Linear(prev_layer, layer))
            modules.append(nn.ReLU())
            prev_layer = layer

        modules.append(nn.Linear(prev_layer, 1))
        self.linears = nn.Sequential(*modules)

        if self.unbiased_loss:
            self.ma_et = None
            self.ma_rate = 0.001

    def forward(self, xy, x_shuffle, x_n_dim):
        h = self.linears(xy)
        y = xy[:, x_n_dim:]
        xy_2 = torch.cat((x_shuffle, y), 1)
        h2 = self.linears(xy_2)
        return h, h2

class MINE_Net2(nn.Module):
    def __init__(self, xy_dim, n_latents,activation_fun, unbiased_loss):
        # activation_fun could be 'ReLU', 'ELU', 'Leaky_ReLU'
        # unbiased_loss: True or False. Whether to use unbiased loss or not
        super(MINE_Net2, self).__init__()
        self.xy_dim = xy_dim
        self.unbiased_loss = unbiased_loss

        if activation_fun=='ReLU':
            modules = [nn.Linear(xy_dim, n_latents[0]), nn.ReLU()]
        elif activation_fun=="ELU":
            modules = [nn.Linear(xy_dim, n_latents[0]), nn.ELU()]
        elif activation_fun=='Leaky_ReLU':
            modules = [nn.Linear(xy_dim, n_latents[0]), nn.LeakyReLU(0.2)]

        prev_layer = n_latents[0]
        for layer in n_latents[1:]:
            modules.append(nn.Linear(prev_layer, layer))
            if activation_fun == 'ReLU':
                modules.append(nn.ReLU())
            elif activation_fun == "ELU":
                modules.append(nn.ELU())
            elif activation_fun == 'Leaky_ReLU':
                modules.append(nn.LeakyReLU(0.2))
            prev_layer = layer

        modules.append(nn.Linear(prev_layer, 1))
        self.linears = nn.Sequential(*modules)

        if self.unbiased_loss:
            self.ma_et = None
            self.ma_rate = 0.001

    def forward(self, x, y):
        h = self.linears(x)
        h2 = self.linears(y)
        return h, h2

class MINE_Net3(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layers, activation_fun, unbiased_loss, initial):
        # activation_fun could be 'ReLU', 'ELU', 'Leaky_ReLU'
        # unbiased_loss: True or False. Whether to use unbiased loss or not
        # initial of weights: 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform','kaiming_normal','orthogonal','sparse'
        # 'orthogonal', 'sparse' are not proper in our case
        super().__init__()
        self.activation_fun = activation_fun
        self.unbiased_loss = unbiased_loss

        layers_dim = [input_dim] + [n_hidden]*n_layers + [1]
        self.layers = nn.Sequential(collections.OrderedDict(
            [('layer{}'.format(i),
                nn.Linear(n_in, n_out)) for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
             ]))

        for i in range(len(layers_dim)-1):
            if initial == 'normal':
                nn.init.normal_(self.layers[i].weight, std=0.02)
                nn.init.constant_(self.layers[i].bias, 0)
            elif initial == 'xavier_uniform':
                nn.init.xavier_uniform_(self.layers[i].weight)
                nn.init.zeros_(self.layers[i].bias)
            elif initial == 'xavier_normal':
                nn.init.xavier_normal_(self.layers[i].weight, gain=1.0)
            elif initial == 'kaiming_uniform':
                if isinstance(self.layers[i], nn.Linear):
                    nn.init.kaiming_uniform_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                    nn.init.constant_(self.layers[i].bias, 0.0)
            elif initial == 'kaiming_normal':
                if isinstance(self.layers[i], nn.Linear):
                    nn.init.kaiming_normal_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                    nn.init.constant_(self.layers[i].bias, 0.0)
            elif initial == 'orthogonal':
                nn.init.orthogonal_(self.layers[i].weight)
                nn.init.zeros_(self.layers[i].bias)
            elif initial == 'sparse':
                nn.init.sparse_(self.layers[i].weight, sparsity=0.1)
                nn.init.zeros_(self.layers[i].bias)

        if self.unbiased_loss:
            self.ma_et = None
            self.ma_rate = 0.001

    def forward(self, input):
        for one_layer in self.layers[0:-1]:
            if self.activation_fun == 'ReLU':
                input_next = F.relu(one_layer(input))
            elif self.activation_fun == "ELU":
                input_next = F.elu(one_layer(input))
            elif self.activation_fun == 'Leaky_ReLU':
                input_next = F.leaky_relu(one_layer(input),negative_slope=2e-1)
            input = input_next
        output = self.layers[-1](input)
        return output

class MINE_Net4(nn.Module):
    def __init__(self, input_dim, n_latents, activation_fun, unbiased_loss, initial, save_path, data_loader, drop_out, net_name, min, max):
        # activation_fun could be 'ReLU', 'ELU', 'Leaky_ReLU'
        # unbiased_loss: True or False. Whether to use unbiased loss or not
        # initial of weights: 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform','kaiming_normal','orthogonal','sparse'
        # 'orthogonal', 'sparse' are not proper in our case
        super().__init__()
        self.activation_fun = activation_fun
        self.unbiased_loss = unbiased_loss
        self.n_hidden_layers = len(n_latents)
        self.save_path = save_path
        self.data_loader = data_loader
        self.name = net_name
        self.min = min
        self.max = max

        layers_dim = [input_dim] + n_latents + [1]
        self.layers = nn.Sequential(collections.OrderedDict(
            [('layer{}'.format(i),
                nn.Linear(n_in, n_out)) for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
             ]))

        for i in range(len(layers_dim)-1):
            if initial == 'normal':
                nn.init.normal_(self.layers[i].weight, std=0.02)
                nn.init.constant_(self.layers[i].bias, 0)
            elif initial == 'xavier_uniform':
                nn.init.xavier_uniform_(self.layers[i].weight)
                nn.init.zeros_(self.layers[i].bias)
            elif initial == 'xavier_normal':
                nn.init.xavier_normal_(self.layers[i].weight, gain=1.0)
            elif initial == 'kaiming_uniform':
                if isinstance(self.layers[i], nn.Linear):
                    nn.init.kaiming_uniform_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                    nn.init.constant_(self.layers[i].bias, 0.0)
            elif initial == 'kaiming_normal':
                if isinstance(self.layers[i], nn.Linear):
                    nn.init.kaiming_normal_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                    nn.init.constant_(self.layers[i].bias, 0.0)
            elif initial == 'orthogonal':
                nn.init.orthogonal_(self.layers[i].weight)
                nn.init.zeros_(self.layers[i].bias)
            elif initial == 'sparse':
                nn.init.sparse_(self.layers[i].weight, sparsity=0.1)
                nn.init.zeros_(self.layers[i].bias)

        self.bn1 = nn.BatchNorm1d(num_features=n_latents[0])
        self.dropout = nn.Dropout(p=drop_out)
        if self.unbiased_loss:
            self.ma_et = None
            self.ma_rate = 0.001

    def forward(self, input):
        for one_layer in self.layers[0:-1]:
            if self.activation_fun == 'ReLU':
                input_next = self.dropout(self.bn1(F.relu(one_layer(input))))
            elif self.activation_fun == "ELU":
                input_next = self.dropout(self.bn1(F.elu(one_layer(input))))
            elif self.activation_fun == 'Leaky_ReLU':
                input_next = self.dropout(self.bn1(F.leaky_relu(one_layer(input),negative_slope=2e-1)))
            input = input_next
        output = self.layers[-1](input)
        return output


class MINE_Net5(nn.Module):

    def __init__(self, xy_dim, hidden_size=128):
        super().__init__()
        self.fc1_sample = nn.Linear(xy_dim, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.ma_et = None
        self.ma_rate = 0.001

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        x_sample = self.fc1_sample(input)
        x = F.leaky_relu(x_sample + self.fc1_bias, negative_slope=2e-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x

class Classifier_Net(nn.Module):
    def __init__(self, input_dim, n_latents, activation_fun, initial, save_path, data_loader, drop_out,net_name, min, max):
        # activation_fun could be 'ReLU', 'ELU', 'Leaky_ReLU'
        # initial: could be 'None','normal', 'xavier_uniform', 'kaiming'
        super().__init__()
        self.activation_fun = activation_fun
        self.n_hidden_layers = len(n_latents)
        self.save_path = save_path
        self.data_loader = data_loader
        self.name = net_name
        self.min = min
        self.max = max

        layers_dim = [input_dim] + n_latents + [1]
        self.layers = nn.Sequential(collections.OrderedDict(
            [('layer{}'.format(i),
                nn.Linear(n_in, n_out)) for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
             ]))

        if initial in ['xavier_uniform', 'xavier_normal', 'kaiming_uniform','kaiming_normal']:
            for i in range(len(layers_dim)-1):
                if initial == 'normal':
                    nn.init.normal_(self.layers[i].weight, std=0.02)
                    nn.init.constant_(self.layers[i].bias, 0)
                elif initial == 'xavier_uniform':
                    nn.init.xavier_uniform_(self.layers[i].weight)
                    nn.init.zeros_(self.layers[i].bias)
                elif initial == 'xavier_normal':
                    nn.init.xavier_normal_(self.layers[i].weight, gain=1.0)
                elif initial == 'kaiming_uniform':
                    if isinstance(self.layers[i], nn.Linear):
                        nn.init.kaiming_uniform_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                        nn.init.constant_(self.layers[i].bias, 0.0)
                elif initial == 'kaiming_normal':
                    if isinstance(self.layers[i], nn.Linear):
                        nn.init.kaiming_normal_(self.layers[i].weight)#recommended to use only with 'relu' or 'leaky_relu' (default)
                        nn.init.constant_(self.layers[i].bias, 0.0)

        self.bn1 = nn.BatchNorm1d(num_features=n_latents[0])
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, input):
        for one_layer in self.layers[0:-1]:
            if self.activation_fun == 'ReLU':
                input_next = self.dropout(self.bn1(F.relu(one_layer(input))))
            elif self.activation_fun == "ELU":
                input_next = self.dropout(self.bn1(F.elu(one_layer(input))))
            elif self.activation_fun == 'Leaky_ReLU':
                input_next = self.dropout(self.bn1(F.leaky_relu(one_layer(input),negative_slope=2e-1)))
            input = input_next
        logit = torch.sigmoid(self.layers[-1](input_next))
        return logit


#Nearest neighbor method is not recommended for back-propogation, one reason is the for loop in it.

# discrete_continuous_info(d, c) estimates the mutual information between a
# discrete vector 'd' and a continuous vector 'c' using
# nearest-neighbor statistics.  Similar to the estimator described by
# Kraskov et. al. ("Estimating Mutual Information", PRE 2004)
# Each vector in c & d is stored as a column in an array:
# c.shape = (vector dimension, # of samples).
import numpy as np
import math
def discrete_continuous_info(d, c, k:int = 3, base: float = 2):
    #this function is transformed from the discrete_continuous_info.m from the paper
    # 'Mutual information between discrete and continuous data sets'.

    # First, bin the continuous data 'c' according to the discrete symbols 'd'
    # d and c are tensors
    first_symbol = []
    symbol_IDs = d.shape[1]*[0]
    c_split = []
    cs_indices = []
    num_d_symbols = 0

    for c1 in range(d.shape[1]):

        symbol_IDs[c1] = num_d_symbols + 1

        for c2 in range(num_d_symbols):
            if d[:,c1] == d[:, first_symbol[c2]]:
                symbol_IDs[c1] = c2
                break
        if symbol_IDs[c1] > num_d_symbols:
            num_d_symbols = num_d_symbols + 1
            first_symbol = first_symbol + [c1]
            c_split = c_split + [torch.transpose(c[:,c1][np.newaxis,:],0,1)]
            cs_indices = cs_indices + [Variable(torch.from_numpy(np.array([[c1]])).type(torch.FloatTensor),requires_grad=True)]
        else:
            c_split[symbol_IDs[c1]-1] = torch.cat((c_split[symbol_IDs[c1]-1], torch.transpose(c[:,c1][np.newaxis,:],0,1)), dim=1)
            cs_indices[symbol_IDs[c1]-1] = torch.cat((cs_indices[symbol_IDs[c1]-1], Variable(torch.from_numpy(np.array([[c1]])).type(torch.FloatTensor),requires_grad=True)), dim=1)

    # Second, compute the neighbor statistic for each data pair (c, d) using
    # the binned c_split list

    m_tot = 0
    av_psi_Nd = 0
    V = d.shape[1]*[0]
    all_c_distances = c.shape[1]*[0]
    psi_ks = 0

    for c_bin in range(num_d_symbols):
        one_k = min(k, c_split[c_bin].shape[1]-1)
        if one_k > 0:
            c_distances = c_split[c_bin].shape[1]*[0]
            c_split_one = c_split[c_bin]
            for pivot in range(c_split[c_bin].shape[1]):
                # find the radius of our volume using only those samples with
                # the particular value of the discrete symbol 'd'
                for cv in range(c_split[c_bin].shape[1]):
                    vector_diff = c_split_one[:,cv][np.newaxis,:] - c_split_one[:,pivot][np.newaxis,:]
                    c_distances[cv] = torch.sqrt(torch.mm(vector_diff, torch.transpose(vector_diff,0,1)))
                eps_over_2 = sorted(c_distances)[one_k]

                for cv in range(c.shape[1]):
                    vector_diff = c[:,cv][np.newaxis,:]-c_split_one[:,pivot][np.newaxis,:]
                    all_c_distances[cv] = torch.sqrt(torch.mm(vector_diff, torch.transpose(vector_diff,0,1)))

                m =  max(len(list(filter(lambda x: x <= eps_over_2, all_c_distances)))-1,0)
                m_tot = m_tot + scipy.special.digamma(m)
                V[cs_indices[c_bin][0,pivot].int()] = (2*eps_over_2)**(d.shape[0])
        else:
            m_tot = m_tot + scipy.special.digamma(num_d_symbols*2)

        p_d = (c_split[c_bin].shape[1])/(d.shape[1])
        av_psi_Nd = av_psi_Nd + p_d*scipy.special.digamma(p_d*(d.shape[1]))
        psi_ks = psi_ks + p_d*scipy.special.digamma(max(one_k, 1))

    f = (scipy.special.digamma(d.shape[1]) - av_psi_Nd + psi_ks - m_tot/(d.shape[1]))/math.log(base)
    return f

def Sample_From_Aggregated_Posterior(qz_m, qz_v, batch_index, batch1_ratio, nsamples_z):
    # nsamples_z: the number of z taken from the aggregated posterior distribution of z
    # qz_v is the variance or covariance matrix for z

    qz_m_batch0 = qz_m[(batch_index[:, 0] == 0).nonzero().squeeze(1)]
    qz_m_batch1 = qz_m[(batch_index[:, 0] == 1).nonzero().squeeze(1)]

    qz_v_batch0 = qz_v[(batch_index[:, 0] == 0).nonzero().squeeze(1)]
    qz_v_batch1 = qz_v[(batch_index[:, 0] == 1).nonzero().squeeze(1)]

    bernoulli = Bernoulli(torch.tensor([batch1_ratio]))  # batch1_ratio is the probability to sample batch 1
    categorical0 = Categorical(torch.tensor([1 / qz_m_batch0.shape[0]] * qz_m_batch0.shape[0]))
    categorical1 = Categorical(torch.tensor([1 / qz_m_batch1.shape[0]] * qz_m_batch1.shape[0]))

    batch_tensor = torch.empty(0, 1)
    z_tensor = torch.empty(0, qz_m.shape[1])
    for i in range(nsamples_z):
        batch = bernoulli.sample()
        if batch.item() == 0:
            category_index = categorical0.sample()
            z = MultivariateNormal(qz_m_batch0[category_index.item(), :], torch.diag(qz_v_batch0[category_index.item(), :])).sample()
        else:
            category_index = categorical1.sample()
            z = MultivariateNormal(qz_m_batch1[category_index.item(), :], torch.diag(qz_v_batch1[category_index.item(), :])).sample()

        batch_tensor = torch.cat((batch_tensor, batch.reshape(1, 1)), 0)
        z_tensor = torch.cat((z_tensor, z.reshape(1, qz_m.shape[1])), 0)

    return batch_tensor, z_tensor
