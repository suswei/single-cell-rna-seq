import collections
from typing import Iterable
import scipy
import torch
from torch import nn as nn
from torch.distributions import Normal

from scvi.models.utils import one_hot
import math
import torch.nn.functional as F
from torch.autograd import Variable
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

#Nearest neighbor method is not applicable for back-propogation.
def Nearest_Neighbor_Estimate(discrete, continuous, k:int = 3):

    continuous_expand0 = continuous.unsqueeze(0).expand(int(continuous.size(0)), int(continuous.size(0)), int(continuous.size(1)))
    continuous_expand1 = continuous.unsqueeze(1).expand(int(continuous.size(0)), int(continuous.size(0)), int(continuous.size(1)))
    L2_distance = ((continuous_expand0-continuous_expand1)**2).sum(2)

    discrete_expand0 = discrete.unsqueeze(0).expand(int(discrete.size(0)), int(discrete.size(0)), int(discrete.size(1)))
    discrete_expand1 = discrete.unsqueeze(1).expand(int(discrete.size(0)), int(discrete.size(0)), int(discrete.size(1)))
    match_index = torch.eq(discrete_expand0, discrete_expand1).sum(2)
    number_same_category = match_index.sum(1) - 1

    match_index_sorted = torch.gather(match_index, 1, torch.argsort(L2_distance, dim=1)) #1 means by row

    index_cum = torch.cumsum(match_index_sorted,dim=1)
    cum_index = (index_cum == k+1).type(torch.FloatTensor)
    reverse_index = torch.arange(cum_index.shape[1], 0, -1).type(torch.FloatTensor)
    tmp = cum_index * reverse_index
    number_samecategory_d = torch.argmax(tmp, 1)

    constant = torch.digamma(torch.tensor([continuous.size(0)]).type(torch.FloatTensor)) + torch.digamma(torch.tensor([k]).type(torch.FloatTensor))
    NN_estimator = constant - torch.mean(torch.digamma(number_same_category.type(torch.FloatTensor))) -  torch.mean(torch.digamma(number_samecategory_d.type(torch.FloatTensor)))
    return NN_estimator.item()
'''
#code is referred to https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
import torch.nn as nn
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.median(L2_distance.data)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.sqrt(torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX))
        return loss
'''

import torch.nn as nn
import numpy as np
from scipy.special import gamma
"""
Return the mmd score between a pair of observations
Notes:
Reimplementation in pytorch of the Information Constraints on Auto-Encoding Variational Bayes
https://github.com/romain-lopez/HCV/blob/master/scVI/scVI.py
"""

class MMD_loss(nn.Module):
    def __init__(self):
        super(MMD_loss, self).__init__()
        return
    def bandwidth(self, d):
        """
        in the case of Gaussian random variables and the use of a RBF kernel,
        this can be used to select the bandwidth according to the median heuristic
        """
        gz = 2 * gamma(0.5 * (d + 1)) / gamma(0.5 * d)
        return 1. / (2. * gz ** 2)

    def K(self,x1, x2, gamma=1.):
        x1_expand = x1.unsqueeze(0).expand(int(x2.size(0)), int(x1.size(0)), int(x1.size(1)))
        x2_expand = x2.unsqueeze(1).expand(int(x2.size(0)), int(x1.size(0)), int(x2.size(1)))
        dist_table = x1_expand - x2_expand

        return torch.transpose(torch.exp(-gamma * ((dist_table ** 2).sum(2))),0,1)

    def forward(self, x1, x2):

        d1 = x1.size()[1]
        d2 = x2.size()[1]

        x1x1 = self.K(x1, x1, gamma=self.bandwidth(d1))
        x2x2 = self.K(x2, x2, gamma=self.bandwidth(d2))
        x1x2 = self.K(x1, x2, gamma=self.bandwidth(d1)) #in our case d1=d2

        return torch.sqrt(torch.mean(x1x1) - 2 * torch.mean(x1x2) + torch.mean(x2x2))

def EmpiricalMI_From_Aggregated_Posterior(qz_m, qz_v, batch_index, batch_ratio, nsamples):
    # nsamples_z: the number of z taken from the aggregated posterior distribution of z
    # qz_v is the variance or covariance matrix for z
    # batch_ratio: if there are two batches, and p(batch=0)=0.6, then batch ratio is [0.6, 0.4]

    batch_categorical = Categorical(torch.tensor(batch_ratio))
    log_density_ratio_list = []
    for i in range(nsamples):

        batch = batch_categorical.sample()
        qz_m_onebatch = qz_m[(batch_index[:, 0] == batch.item()).nonzero().squeeze(1)]
        qz_v_onebatch = qz_v[(batch_index[:, 0] == batch.item()).nonzero().squeeze(1)]

        z_categorical = Categorical(torch.tensor([1 / qz_m_onebatch.shape[0]] * qz_m_onebatch.shape[0]))
        z_category_index = z_categorical.sample()
        z = MultivariateNormal(qz_m_onebatch[z_category_index.item(), :],torch.diag(qz_v_onebatch[z_category_index.item(), :])).sample()

        # use log-sum-exp trick to calculate log_density_ratio
        componentwise_log_prob = {}
        component_num = []
        for j in range(len(batch_ratio)):
            qz_m_onebatch = qz_m[(batch_index[:, 0] == j).nonzero().squeeze(1)]
            qz_v_onebatch = qz_v[(batch_index[:, 0] == j).nonzero().squeeze(1)]
            component_num += [qz_m_onebatch.shape[0]]
            componentwise_log_prob.update({'batch{}'.format(j):[MultivariateNormal(qz_m_onebatch[k, :],torch.diag(qz_v_onebatch[k, :])).log_prob(z).item() for k in range(qz_m_onebatch.shape[0])]})

        prob_z = 0
        for j in range(len(batch_ratio)):
            componentwise_log_prob_onebatch = componentwise_log_prob['batch{}'.format(j)]
            max_onebatch = max(componentwise_log_prob_onebatch)
            prob_z += (1/component_num[j])*batch_ratio[j]*math.exp(max_onebatch)*sum([math.exp(ele - max_onebatch) for ele in componentwise_log_prob_onebatch])
            if j==batch:
                log_prob_z_s = -math.log(component_num[j])+ max_onebatch + math.log(sum([math.exp(ele - max_onebatch) for ele in componentwise_log_prob_onebatch]))

        log_prob_z = math.log(prob_z)
        log_density_ratio = log_prob_z_s - log_prob_z
        log_density_ratio_list += [log_density_ratio]
    empirical_MI = sum(log_density_ratio_list)/len(log_density_ratio_list)

    return empirical_MI
