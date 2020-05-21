# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence as kl

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from torch.autograd import Variable

import torch.nn.functional as F


from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI, MINE_Net4, discrete_continuous_info, Sample_From_Aggregated_Posterior
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True



# VAE model
class VAE_MI(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log variational distribution
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    :param n_hidden_z, n_layers_z: MINE network parameter for latent code

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(self, n_input: int, n_batch: int = 0, n_labels: int = 0,
                 n_hidden: int = 128, n_latent: int = 10, n_layers_encoder: int = 1,
                 n_layers_decoder: int = 1,
                 dropout_rate: float = 0.1, dispersion: str = "gene",
                 log_variational: bool = True, reconstruction_loss: str = "zinb",
                 n_hidden_z: int = 5, n_layers_z: int = 10,
                 MI_estimator: str = 'NN', Adv_MineNet4_architecture: list=[32,16], MIScale: int=0,
                 nsamples_z: int=200, adv: bool=False, adv_minibatch_MI: float=0, save_path: str='None',
                 minibatch_index: int=0, mini_ELBO: int=10000, max_ELBO: int=1000000, minibatch_number: int=60,
                 std: bool=False, MIScale_index: int=1):
        super().__init__()
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent = n_latent
        self.n_latent_layers = 1  # not sure what this is for, no usages?

        self.n_hidden_z = n_hidden_z
        self.n_layers_z = n_layers_z
        self.MI_estimator = MI_estimator
        self.Adv_MineNet4_architecture = Adv_MineNet4_architecture
        self.MIScale = MIScale
        self.nsamples_z = nsamples_z
        self.adv = adv
        self.adv_minibatch_MI = adv_minibatch_MI
        self.save_path = save_path
        self.minibatch_index = minibatch_index
        self.minibatch_number = minibatch_number
        self.mini_ELBO = mini_ELBO
        self.max_ELBO = max_ELBO
        self.std = std
        self.MIScale_index = MIScale_index

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(n_input, n_latent, n_layers=n_layers_encoder, n_hidden=n_hidden, dropout_rate=dropout_rate)
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(n_input, 1, n_layers=n_layers_encoder, n_hidden=n_hidden, dropout_rate=dropout_rate)
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(n_latent, n_input, n_cat_list=[n_batch], n_layers=n_layers_decoder, n_hidden=n_hidden)

    def get_latents(self, x, y=None):
        r""" returns the result of ``sample_from_posterior_z`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z

    def sample_from_posterior_l(self, x):
        r""" samples the tensor of library sizes from the posterior
        #doesn't really sample, returns the tensor of the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :return: tensor of shape ``(batch_size, 1)``
        :rtype: :py:class:`torch.Tensor`
        """
        if self.log_variational:
            x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of predicted frequencies of expression

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param n_samples: number of samples
        :return: tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[0]

    def get_sample_rate(self, x, batch_index=None, y=None, n_samples=1):
        r"""Returns the tensor of means of the negative binomial distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param y: tensor of cell-types labels with shape ``(batch_size, n_labels)``
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param n_samples: number of samples
        :return: tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        :rtype: :py:class:`torch.Tensor`
        """
        return self.inference(x, batch_index=batch_index, y=y, n_samples=n_samples)[2]

    def _reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        qz_m, qz_v, z = self.z_encoder(sample_batch)
        batch_index = fixed_batch * torch.ones_like(sample_batch[:, [0]])
        library = 4. * torch.ones_like(sample_batch[:, [0]])
        px_scale, _, _, _ = self.decoder('gene', z, library, batch_index)
        return px_scale


    def inference(self, x, batch_index=None, y=None, n_samples=1, nsamples_z=1000):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        ql_m, ql_v, library = self.l_encoder(x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index, y)
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        # calculate mutual information(MI) using MINE_Net, code from MasanoriYamada/Mine_pytorch
        z_shuffle = np.random.permutation(z.detach().numpy())
        z_shuffle = Variable(torch.from_numpy(z_shuffle).type(torch.FloatTensor), requires_grad=True)
        batch_index = Variable(batch_index.type(torch.FloatTensor), requires_grad=True)

        #n_input_nuisance = batch_index.shape[1]
        #n_input_z = z.shape[1]
        #minenet = MINE_Net(n_input_nuisance,n_input_z,self.n_hidden_z,self.n_layers_z)
        #pred_xz = minenet(batch_index, z) #pred_xz has the dimension [128,1], because the batch_size for each minibatch is 128
        #pred_x_z = minenet(batch_index, z_shuffle) #pred_xz has the dimension [128,1], because the batch_size for each minibatch is 128
        '''
        if self.adv == False:
            #z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v, batch_index, self.nsamples_z)
            #batch0_indices = np.array([[0] * (z_batch0.shape[0])])
            #batch1_indices = np.array([[1] * (z_batch1.shape[0])])
            # calculate mutual information(MI) using MINE_Net4
            if self.MI_estimator=='Mine_Net4': #modify this if part, use the samples from posterior distribution
                batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.ravel())})
                batch_dummy = pd.get_dummies(batch_dataframe['batch']).values
                batch_dummy = Variable(batch_dummy.type(torch.FloatTensor), requires_grad=True)
                z_batch = torch.cat([z, batch_dummy],dim=1)
                self.minenet = MINE_Net4(z_batch.shape[-1], self.MineNet4_architecture)
                pred_xz, pred_x_z = self.minenet(xy=z_batch, x_shuffle=z_shuffle, x_n_dim=z.shape[-1])
            elif self.MI_estimator=='NN':
            # calculate mutual information(MI) using nearest neighbor method
                #batch_array = np.append(batch0_indices, batch1_indices, axis=1)
                #z_array = np.append(z_batch0, z_batch1, axis=0).transpose()
                #predicted_mutual_info = discrete_continuous_info(d=batch_array, c=z_array)
                batch_index_array = np.array(batch_index.detach().numpy().transpose())
                z_array = z.detach().numpy().transpose()
                library_array = np.array(library.detach().numpy().transpose())
                batch_index_array_tensor = Variable(torch.from_numpy(batch_index_array).type(torch.FloatTensor), requires_grad=True)
                z_array_tensor = Variable(torch.from_numpy(z_array).type(torch.FloatTensor), requires_grad=True)
                library_array_tensor = Variable(torch.from_numpy(library_array).type(torch.FloatTensor), requires_grad=True)
                l_z_array_tensor = torch.cat((library_array_tensor,z_array_tensor),dim=0)
                predicted_mutual_info = discrete_continuous_info(d=batch_index_array_tensor, c=l_z_array_tensor)
            #elif self.MI_estimator=='aggregated_posterior':
                #z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
                #z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
                #self.minenet = MINE_Net4_2(z_batch0_tensor.shape[-1], self.MineNet4_architecture)
                #pred_xz, pred_x_z = self.minenet(x=z_batch0_tensor, y=z_batch1_tensor)

            #TODO: have another MINE net for library depth

            if self.MI_estimator in ['Mine_Net4','aggregated_posterior']:
                return px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library, pred_xz, pred_x_z
            elif self.MI_estimator=='NN':
                return px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library, predicted_mutual_info
        '''
        return px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variancess of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        '''
        if self.adv == False:
            if self.MI_estimator == 'MI':
                px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library, pred_xz, pred_x_z = self.inference(x, batch_index, y, nsamples_z=self.nsamples_z)
            elif self.MI_estimator=='NN':
                px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library, predicted_mutual_info = self.inference(x, batch_index, y, nsamples_z=self.nsamples_z)
        '''
        px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library = self.inference(x, batch_index, y, nsamples_z=self.nsamples_z)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1) #kl_divergence_z: dimension [128]
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)#kl_divergence_l: dimension [128]

        reconst_loss = self._reconstruction_loss(x, px_rate, px_r, px_dropout) # reconst_loss: dimension [128]
        '''
        if self.adv == False:
            # calculate Mutual information(MI) loss
            if self.MI_estimator == 'MI':
                MIloss = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z))) #MIloss: dimension: [1]
            elif self.MI_estimator=='NN':
                MIloss = predicted_mutual_info  # MIloss: dimension: [1]

            print('kl_divergence_z: {}, kl_divergence_l: {}, reconst_loss: {}, MI loss: {}, scaled MI loss: {}'.format(kl_divergence_z.mean(),kl_divergence_l.mean(),reconst_loss.mean(), MIloss, self.MIScale*MIloss))

            # TODO: should return kl_divergence_z and MIloss separately, in current state same penalty term is applied to them
            return reconst_loss + kl_divergence_l+ kl_divergence_z, self.MIScale*MIloss
        '''

        return reconst_loss + kl_divergence_l, kl_divergence_z
