import collections
from typing import Iterable

import torch
from torch import nn as nn
from torch.distributions import Normal

from scvi.models.utils import one_hot

import torch.nn.functional as F

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
    """
    Takes two inputs nuisance and z, implements T_\theta(nuisance,z) \in \mathbb R function in MINE paper
    Typically applied with nuisance representing either batch or library size and z representing latent code
    """

    def __init__(self,n_input_nuisance,n_input_z,n_hidden_z,n_layers_z):
        super(MINE_Net, self).__init__()
        self.nn_nuisance = nn.Linear(n_input_nuisance, 1)
        self.nn_z = FCLayers(n_in=n_input_z, n_out=1,
                                n_layers=n_layers_z,
                                n_hidden=n_hidden_z, dropout_rate=0)

    def forward(self, nuisance, z):
        h = F.relu(self.nn_nuisance(nuisance)+self.nn_z(z))
        return h


class MINE_Net2(nn.Module):
    """
    Takes two inputs nuisance and z, implements T_\theta(nuisance,z) \in \mathbb R function in MINE paper
    Typically applied with nuisance representing either batch or library size and z representing latent code
    """

    def __init__(self,n_input_nuisance,n_input_z,n_hidden_z,n_layers_z):
        super(MINE_Net2, self).__init__()
        self.nn_nuisance = FCLayers(n_in=n_input_nuisance, n_out=1,
                                n_layers=n_layers_z,
                                n_hidden=n_hidden_z, dropout_rate=0)
        self.nn_z = FCLayers(n_in=n_input_z, n_out=1,
                                n_layers=n_layers_z,
                                n_hidden=n_hidden_z, dropout_rate=0)

    def forward(self, nuisance, z):
        h = F.relu(self.nn_nuisance(nuisance)+self.nn_z(z))
        return h


class MINE_Net3(nn.Module):
    def __init__(self,n_input_nuisance, n_input_z, H):
        super(MINE_Net3, self).__init__()
        self.fc1 = nn.Linear(n_input_nuisance, H)
        self.fc2 = nn.Linear(n_input_z, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2


class MINE_Net4(nn.Module):
    def __init__(self, xy_dim, n_latents):
        super(MINE_Net4, self).__init__()
        self.xy_dim = xy_dim

        modules = [nn.Linear(xy_dim, n_latents[0]), nn.ReLU()]

        prev_layer = n_latents[0]
        for layer in n_latents[1:]:
            modules.append(nn.Linear(prev_layer, layer))
            modules.append(nn.ReLU())
            prev_layer = layer

        modules.append(nn.Linear(prev_layer, 1))
        self.linears = nn.Sequential(*modules)

    def forward(self, xy, x_shuffle, x_n_dim):
        h = self.linears(xy)
        y = xy[:, x_n_dim:]
        xy_2 = torch.cat((x_shuffle, y), 1)
        h2 = self.linears(xy_2)
        return h, h2


# discrete_continuous_info(d, c) estimates the mutual information between a
# discrete vector 'd' and a continuous vector 'c' using
# nearest-neighbor statistics.  Similar to the estimator described by
# Kraskov et. al. ("Estimating Mutual Information", PRE 2004)
# Each vector in c & d is stored as a column in an array:
# c.shape = (vector length, # samples).
import numpy as np
import math
import scipy
def discrete_continuous_info(d, c, k:int = 3, base: float = 2):
    # First, bin the continuous data 'c' according to the discrete symbols 'd'
    # d and c are array
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
            c_split = c_split + [np.array([c[:,c1]]).transpose()]
            cs_indices = cs_indices + [np.array([[c1]])]
        else:
            c_split[symbol_IDs[c1]-1] = np.concatenate((c_split[symbol_IDs[c1]-1], np.array([c[:,c1]]).transpose()), axis=1)
            cs_indices[symbol_IDs[c1]-1] = np.concatenate((cs_indices[symbol_IDs[c1]-1], np.array([[c1]])), axis=1)

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
            c_split_one = np.asmatrix(c_split[c_bin])
            for pivot in range(c_split[c_bin].shape[1]):
                # find the radius of our volume using only those samples with
                # the particular value of the discrete symbol 'd'
                for cv in range(c_split[c_bin].shape[1]):
                    vector_diff = c_split_one[:,cv] - c_split_one[:,pivot]
                    c_distances[cv] = math.sqrt(np.matmul(vector_diff.transpose(),vector_diff)[0,0])
                eps_over_2 = sorted(c_distances)[one_k]
                c_matrix = np.asmatrix(c)

                for cv in range(c.shape[1]):
                    vector_diff = c_matrix[:,cv]-c_split_one[:,pivot]
                    all_c_distances[cv] = math.sqrt(np.matmul(vector_diff.transpose(),vector_diff)[0,0])

                m =  max(len(list(filter(lambda x: x <= eps_over_2, all_c_distances)))-1,0)
                m_tot = m_tot + scipy.special.digamma(m)
                V[cs_indices[c_bin][0,pivot]] = (2*eps_over_2)**(d.shape[0])
        else:
            m_tot = m_tot + scipy.special.digamma(num_d_symbols*2)

        p_d = (c_split[c_bin].shape[1])/(d.shape[1])
        av_psi_Nd = av_psi_Nd + p_d*scipy.special.digamma(p_d*(d.shape[1]))
        psi_ks = psi_ks + p_d*scipy.special.digamma(max(one_k, 1))

    f = (scipy.special.digamma(d.shape[1]) - av_psi_Nd + psi_ks - m_tot/(d.shape[1]))/math.log(base)
    return f
