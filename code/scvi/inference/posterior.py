from abc import abstractmethod
import copy

import numpy as np
import pandas as pd
import scipy
import torch
import os
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kde, entropy
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, RandomSampler

from code.scvi import compute_log_likelihood, compute_marginal_log_likelihood


class SequentialSubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)

def _init_fn(worker_id):
    np.random.seed(int(0))
# put worker_init_fn=_init_fn in the DataLoader() function

class Posterior:
    r"""The functional data unit. A `Posterior` instance is instanciated with a model and a gene_dataset, and
    as well as additional arguments that for Pytorch's `DataLoader`. A subset of indices can be specified, for
    purpose such as splitting the data into train/test or labelled/unlabelled (for semi-supervised learning).
    Each trainer instance of the `Trainer` class can therefore have multiple `Posterior` instances to train a model.
    A `Posterior` instance also comes with many methods or utilities for its corresponding data.


    :param model: Number of input genes
    :param gene_dataset: Number of batches
    :param shuffle: Number of labels
    :param indices: Number of nodes per hidden layer
    :param use_cuda: Dimensionality of the latent space
    :param data_loader_kwargs: Number of hidden layers used for encoder and decoder NNs

    Examples:

    Let's instanciate a `trainer`, with a gene_dataset and a model

        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels, use_cuda=True)
        >>> trainer = UnsupervisedTrainer(vae, gene_dataset)
        >>> trainer.train(n_epochs=50)

    A `UnsupervisedTrainer` instance has two `Posterior` attributes: `train_set` and `test_set`
    For this subset of the original gene_dataset instance, we can examine the differential expression,
    log_likelihood, entropy batch mixing, ... or display the TSNE of the data in the latent space through the
    scVI model

        >>> trainer.train_set.differential_expression_stats()
        >>> trainer.train_set.ll()
        >>> trainer.train_set.entropy_batch_mixing()
        >>> trainer.train_set.show_t_sne(n_samples=1000, color_by='labels')

    """

    def __init__(self, model, gene_dataset, shuffle=False, indices=None, use_cuda=True, data_loader_kwargs=dict()):
        '''
        When added to annotation, has a private name attribute
        '''
        self.model = model
        self.gene_dataset = gene_dataset
        self.to_monitor = []
        self.use_cuda = use_cuda

        if indices is not None and shuffle:
            raise ValueError('indices is mutually exclusive with shuffle')
        if indices is None:
            if shuffle:
                sampler = RandomSampler(gene_dataset)
            else:
                sampler = SequentialSampler(gene_dataset)
        else:
            if hasattr(indices, 'dtype') and indices.dtype is np.dtype('bool'):
                indices = np.where(indices)[0].ravel()
            sampler = SubsetRandomSampler(indices)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        if hasattr(gene_dataset, 'collate_fn'):
            self.data_loader_kwargs.update({'collate_fn': gene_dataset.collate_fn})
        self.data_loader_kwargs.update({'sampler': sampler})
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)


    @abstractmethod
    def accuracy(self, verbose=False):
        pass

    accuracy.mode = 'max'

    @property
    def indices(self):
        if hasattr(self.data_loader.sampler, 'indices'):
            return self.data_loader.sampler.indices
        else:
            return np.arange(len(self.gene_dataset))

    def __iter__(self):
        return map(self.to_cuda, iter(self.data_loader))

    def to_cuda(self, tensors):
        return [t.cuda() if self.use_cuda else t for t in tensors]

    def update(self, data_loader_kwargs):
        posterior = copy.copy(self)
        posterior.data_loader_kwargs = copy.copy(self.data_loader_kwargs)
        posterior.data_loader_kwargs.update(data_loader_kwargs)
        posterior.data_loader = DataLoader(self.gene_dataset, **posterior.data_loader_kwargs)
        return posterior

    def sequential(self, batch_size=128):
        return self.update({'batch_size': batch_size, 'sampler': SequentialSubsetSampler(indices=self.indices)})

    def corrupted(self):
        return self.update({'collate_fn': self.gene_dataset.collate_fn_corrupted})

    def uncorrupted(self):
        return self.update({'collate_fn': self.gene_dataset.collate_fn})

    def ll(self, verbose=False):
        ll = compute_log_likelihood(self.model, self)
        if verbose:
            print("LL : %.4f" % ll)
        return ll

    ll.mode = 'min'

    def marginal_ll(self, verbose=False, n_mc_samples=1000):
        ll = compute_marginal_log_likelihood(self.model, self, n_mc_samples)
        if verbose:
            print("True LL : %.4f" % ll)
        return ll

    @torch.no_grad()
    def get_latent(self, sample=False):
        latent = []
        batch_indices = []
        labels = []
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            if not sample:
                if self.model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)
                latent += [self.model.z_encoder(sample_batch)[0].cpu()]
            else:
                latent += [self.model.sample_from_posterior_z(sample_batch).cpu()]
            batch_indices += [batch_index.cpu()]
            labels += [label.cpu()]
        return np.array(torch.cat(latent)), np.array(torch.cat(batch_indices)), np.array(torch.cat(labels)).ravel()

    @torch.no_grad()
    def entropy_batch_mixing(self, verbose=False, **kwargs):
        if self.gene_dataset.n_batches == 2:
            latent, batch_indices, labels = self.get_latent()
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            if verbose:
                print("Entropy batch mixing :", be_score)
            return be_score

    entropy_batch_mixing.mode = 'max'

    @torch.no_grad()
    def differential_expression_stats(self, M_sampling=100):
        """
        Output average over statistics in a symmetric way (a against b)
        forget the sets if permutation is True
        :param vae: The generative vae and encoder network
        :param data_loader: a data loader for a particular dataset
        :param M_sampling: number of samples
        :return: A 1-d vector of statistics of size n_genes
        """
        px_scales = []
        all_labels = []
        batch_size = max(self.data_loader_kwargs['batch_size'] // M_sampling, 2)  # Reduce batch_size on GPU
        if len(self.gene_dataset) % batch_size == 1:
            batch_size += 1
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array((self.model.get_sample_scale(
                    sample_batch, batch_index=batch_index, y=labels, n_samples=M_sampling)
                         ).cpu())]

            # Align the sampling
            if M_sampling > 1:
                px_scales[-1] = (px_scales[-1].transpose((1, 0, 2))).reshape(-1, px_scales[-1].shape[-1])
            all_labels += [np.array((labels.repeat(1, M_sampling).view(-1, 1)).cpu())]

        px_scales = np.concatenate(px_scales)
        all_labels = np.concatenate(all_labels).ravel()  # this will be used as boolean

        return px_scales, all_labels

    @torch.no_grad()
    def sample_scale_from_batch(self, n_samples, batchid=None, selection=None):
        px_scales = []
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        else:
            if selection.dtype is np.dtype('bool'):
                selection = np.asarray(np.where(selection)[0].ravel())
        for i in batchid:
            idx = np.random.choice(np.arange(len(self.gene_dataset))[selection], n_samples)
            sampler = SubsetRandomSampler(idx)
            self.data_loader_kwargs.update({'sampler': sampler})
            self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)
            px_scales.append(self.get_harmonized_scale(i))
        sampler = RandomSampler(self.gene_dataset)
        self.data_loader_kwargs.update({'sampler': sampler})
        self.data_loader = DataLoader(self.gene_dataset, **self.data_loader_kwargs)
        px_scales = np.concatenate(px_scales)
        return px_scales

    @torch.no_grad()
    def differential_expression_score(self, idx1, idx2, batchid1=None, batchid2=None,
                                      genes=None, n_samples=None, M_permutation=None, all_stats=True,
                                      sample_pairs=True):
        if n_samples is None:
            n_samples = 5000
        if M_permutation is None:
            M_permutation = 10000
        if batchid1 is None:
            batchid1 = np.arange(self.gene_dataset.n_batches)
        if batchid2 is None:
            batchid2 = np.arange(self.gene_dataset.n_batches)
        px_scale1 = self.sample_scale_from_batch(selection=idx1, batchid=batchid1, n_samples=n_samples)
        px_scale2 = self.sample_scale_from_batch(selection=idx2, batchid=batchid2, n_samples=n_samples)
        px_scale_mean1 = px_scale1.mean(axis=0)
        px_scale_mean2 = px_scale2.mean(axis=0)
        px_scale = np.concatenate((px_scale1, px_scale2), axis=0)
        all_labels = np.concatenate((np.repeat(0, len(px_scale1)), np.repeat(1, len(px_scale2))), axis=0)
        if genes is not None:
            px_scale = px_scale[:, self.gene_dataset._gene_idx(genes)]
        bayes1 = get_bayes_factors(px_scale, all_labels, cell_idx=0, M_permutation=M_permutation,
                                   permutation=False, sample_pairs=sample_pairs)
        if all_stats is True:
            bayes1_permuted = get_bayes_factors(px_scale, all_labels, cell_idx=0, M_permutation=M_permutation,
                                                permutation=True, sample_pairs=sample_pairs)
            bayes2 = get_bayes_factors(px_scale, all_labels, cell_idx=1, M_permutation=M_permutation,
                                       permutation=False, sample_pairs=sample_pairs)
            bayes2_permuted = get_bayes_factors(px_scale, all_labels, cell_idx=1, M_permutation=M_permutation,
                                                permutation=True, sample_pairs=sample_pairs)
            mean1, mean2, nonz1, nonz2, norm_mean1, norm_mean2 = \
                self.gene_dataset.raw_counts_properties(idx1, idx2)
            res = pd.DataFrame([bayes1, bayes1_permuted, bayes2, bayes2_permuted,
                                mean1, mean2, nonz1, nonz2, norm_mean1, norm_mean2,
                                px_scale_mean1, px_scale_mean2],
                               index=['bayes1', 'bayes1_permuted', 'bayes2', 'bayes2_permuted',
                                      'mean1', 'mean2', 'nonz1', 'nonz2', 'norm_mean1', 'norm_mean2',
                                      'scale1', 'scale2'],
                               columns=self.gene_dataset.gene_names).T
            res = res.sort_values(by=['bayes1'], ascending=False)
            return res
        else:
            return(bayes1)

    @torch.no_grad()
    def one_vs_all_degenes(self, subset=None, cell_labels=None, min_cells=10,
                           n_samples=None, M_permutation=None, output_file=False,
                           save_dir='./', filename='one2all'):
        if cell_labels is not None:
            if len(cell_labels) != len(self.gene_dataset):
                raise ValueError(" the length of cell_labels have to be the same as the number of cells")
        if (cell_labels is None) and not hasattr(self.gene_dataset, 'cell_types'):
            raise ValueError("If gene_dataset is not annotated with labels and cell types,"
                             " then must provide cell_labels")
        # Input cell_labels take precedence over cell type label annotation in dataset
        elif (cell_labels is not None):
            cluster_id = np.unique(cell_labels[cell_labels >= 0])
            # Can make cell_labels < 0 to filter out cells when computing DE
        else:
            cluster_id = self.gene_dataset.cell_types
            cell_labels = self.gene_dataset.labels.ravel()
        de_res = []
        de_cluster = []
        for i, x in enumerate(cluster_id):
            if subset is None:
                idx1 = (cell_labels == i)
                idx2 = (cell_labels != i)
            else:
                idx1 = (cell_labels == i) * subset
                idx2 = (cell_labels != i) * subset
            if np.sum(idx1) > min_cells and np.sum(idx2) > min_cells:
                de_cluster.append(x)
                res = self.differential_expression_score(idx1=idx1, idx2=idx2, M_permutation=M_permutation,
                                                         n_samples=n_samples, sample_pairs=False)
                res['clusters'] = np.repeat(x, len(res.index))
                de_res.append(res)
        if output_file:  # store as an excel spreadsheet
            writer = pd.ExcelWriter(save_dir + 'differential_expression.%s.xlsx' % filename, engine='xlsxwriter')
            for i, x in enumerate(de_cluster):
                de_res[i].to_excel(writer, sheet_name=str(x))
            writer.close()
        return de_res, de_cluster

    def within_cluster_degenes(self, cell_labels=None, min_cells=10, states=[], batch1=None, batch2=None, subset=None,
                               n_samples=None, M_permutation=None, output_file=False,
                               save_dir='./', filename='within_cluster'):
        if len(self.gene_dataset) != len(states):
            raise ValueError(" the length of states have to be the same as the number of cells")
        if cell_labels is not None:
            if len(cell_labels) != len(self.gene_dataset):
                raise ValueError(" the length of cell_labels have to be the same as the number of cells")
        if (cell_labels is None) and not hasattr(self.gene_dataset, 'cell_types'):
            raise ValueError("If gene_dataset is not annotated with labels and cell types,"
                             " then must provide cell_labels")
        # Input cell_labels take precedence over cell type label annotation in dataset
        elif (cell_labels is not None):
            cluster_id = np.unique(cell_labels[cell_labels >= 0])
            # Can make cell_labels < 0 to filter out cells when computing DE
        else:
            cluster_id = self.gene_dataset.cell_types
            cell_labels = self.gene_dataset.labels.ravel()
        de_res = []
        de_cluster = []
        states = np.asarray([1 if x else 0 for x in states])
        nstates = np.asarray([0 if x else 1 for x in states])
        for i, x in enumerate(cluster_id):
            if subset is None:
                idx1 = (cell_labels == i) * states
                idx2 = (cell_labels == i) * nstates
            else:
                idx1 = (cell_labels == i) * subset * states
                idx2 = (cell_labels == i) * subset * nstates
            if np.sum(idx1) > min_cells and np.sum(idx2) > min_cells:
                de_cluster.append(x)
                res = self.differential_expression_score(idx1=idx1, idx2=idx2,
                                                         batchid1=batch1, batchid2=batch2, M_permutation=M_permutation,
                                                         n_samples=n_samples)
                res['clusters'] = np.repeat(x, len(res.index))
                de_res.append(res)
        if output_file:  # store as an excel spreadsheet
            writer = pd.ExcelWriter(save_dir + 'differential_expression.%s.xlsx' % filename, engine='xlsxwriter')
            for i, x in enumerate(de_cluster):
                de_res[i].to_excel(writer, sheet_name=str(x))
            writer.close()
        return de_res, de_cluster

    @torch.no_grad()
    def imputation(self, n_samples=1):
        imputed_list = []
        for tensors in self:
            sample_batch, _, _, batch_index, labels = tensors
            px_rate = self.model.get_sample_rate(sample_batch, batch_index=batch_index, y=labels, n_samples=n_samples)
            imputed_list += [np.array(px_rate.cpu())]
        imputed_list = np.concatenate(imputed_list)
        return imputed_list.squeeze()

    @torch.no_grad()
    def generate(self, n_samples=100, genes=None):  # with n_samples>1 return original list/ otherwose sequential
        '''
        Return original_values as y and generated as x (for posterior density visualization)
        :param n_samples:
        :param genes:
        :return:
        '''
        original_list = []
        posterior_list = []
        batch_size = 128  # max(self.data_loader_kwargs['batch_size'] // n_samples, 2)  # Reduce batch_size on GPU
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, batch_index, labels = tensors
            px_dispersion, px_rate = self.model.inference(sample_batch, batch_index=batch_index, y=labels,
                                                          n_samples=n_samples)[1:3]

            p = (px_rate / (px_rate + px_dispersion)).cpu()
            r = px_dispersion.cpu()
            #
            l_train = np.random.gamma(r, p / (1 - p))
            X = np.random.poisson(l_train)
            # '''
            # In numpy (shape, scale) => (concentration, rate), with scale = p /(1 - p)
            # rate = (1 - p) / p  # = 1/scale # used in pytorch
            # l_train = Gamma(r, rate).sample()  # assert Gamma(r, rate).mean = px_rate
            # posterior = Poisson(l_train).sample()
            # '''
            original_list += [np.array(sample_batch.cpu())]
            posterior_list += [X]  # [np.array(posterior.cpu())]##

            if genes is not None:
                posterior_list[-1] = posterior_list[-1][:, :, self.gene_dataset._gene_idx(genes)]
                original_list[-1] = original_list[-1][:, self.gene_dataset._gene_idx(genes)]

            posterior_list[-1] = np.transpose(posterior_list[-1], (1, 2, 0))

        return np.concatenate(posterior_list, axis=0), np.concatenate(original_list, axis=0)

    @torch.no_grad()
    def generate_parameters(self):
        dropout_list = []
        mean_list = []
        dispersion_list = []
        for tensors in self.sequential(1000):
            sample_batch, _, _, batch_index, labels = tensors
            px_dispersion, px_rate, px_dropout = self.model.inference(sample_batch, batch_index=batch_index, y=labels,
                                                                      n_samples=1)[1:4]

            dispersion_list += [np.repeat(np.array(px_dispersion.cpu())[np.newaxis, :], px_rate.size(0), axis=0)]
            mean_list += [np.array(px_rate.cpu())]
            dropout_list += [np.array(px_dropout.cpu())]

        return np.concatenate(dropout_list), np.concatenate(mean_list), np.concatenate(dispersion_list)

    @torch.no_grad()
    def get_stats(self, verbose=True):
        libraries = []
        for tensors in self.sequential(batch_size=128):
            x, local_l_mean, local_l_var, batch_index, y = tensors
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library = \
                self.model.inference(x, batch_index, y)
            libraries += [np.array(library.cpu())]
        libraries = np.concatenate(libraries)
        return libraries.ravel()

    @torch.no_grad()
    def get_harmonized_scale(self, fixed_batch):
        px_scales = []
        fixed_batch = float(fixed_batch)
        for tensors in self:
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            px_scales += [self.model.scale_from_z(sample_batch, fixed_batch).cpu()]
        return np.concatenate(px_scales)

    @torch.no_grad()
    def get_sample_scale(self):
        px_scales = []
        for tensors in self:
            sample_batch, _, _, batch_index, labels = tensors
            px_scales += [
                np.array((self.model.get_sample_scale(
                    sample_batch, batch_index=batch_index, y=labels, n_samples=1)
                         ).cpu())]
        return np.concatenate(px_scales)

    @torch.no_grad()
    def imputation_list(self, n_samples=1):
        original_list = []
        imputed_list = []
        batch_size = 10000  # self.data_loader_kwargs['batch_size'] // n_samples
        for tensors, corrupted_tensors in zip(self.uncorrupted().sequential(batch_size=batch_size),
                                              self.corrupted().sequential(batch_size=batch_size)):
            batch = tensors[0]
            actual_batch_size = batch.size(0)
            dropout_batch, _, _, batch_index, labels = corrupted_tensors
            px_rate = self.model.get_sample_rate(dropout_batch, batch_index=batch_index, y=labels, n_samples=n_samples)

            indices_dropout = torch.nonzero(batch - dropout_batch)
            if indices_dropout.size() != torch.Size([0]):
                i = indices_dropout[:, 0]
                j = indices_dropout[:, 1]

                batch = batch.unsqueeze(0).expand((n_samples, batch.size(0), batch.size(1)))
                original = np.array(batch[:, i, j].view(-1).cpu())
                imputed = np.array(px_rate[..., i, j].view(-1).cpu())

                cells_index = np.tile(np.array(i.cpu()), n_samples)

                original_list += [original[cells_index == i] for i in range(actual_batch_size)]
                imputed_list += [imputed[cells_index == i] for i in range(actual_batch_size)]
            else:
                original_list = np.array([])
                imputed_list = np.array([])
        return original_list, imputed_list

    @torch.no_grad()
    def imputation_score(self, verbose=False, original_list=None, imputed_list=None, n_samples=1):
        if original_list is None or imputed_list is None:
            original_list, imputed_list = self.imputation_list(n_samples=n_samples)
            if len(original_list) == 0:
                print("No difference between corrupted dataset and uncorrupted dataset")
                return 0
        return np.median(np.abs(np.concatenate(original_list) - np.concatenate(imputed_list)))

    @torch.no_grad()
    def imputation_benchmark(self, n_samples=8, verbose=False, show_plot=True, title_plot='imputation', save_path=''):
        original_list, imputed_list = self.imputation_list(n_samples=n_samples)
        # Median of medians for all distances
        median_score = self.imputation_score(original_list=original_list, imputed_list=imputed_list)

        # Mean of medians for each cell
        imputation_cells = []
        for original, imputed in zip(original_list, imputed_list):
            has_imputation = len(original) and len(imputed)
            imputation_cells += [np.median(np.abs(original - imputed)) if has_imputation else 0]
        mean_score = np.mean(imputation_cells)

        if verbose:
            print("\nMedian of Median: %.4f\nMean of Median for each cell: %.4f" % (median_score, mean_score))

        plot_imputation(np.concatenate(original_list), np.concatenate(imputed_list), show_plot=show_plot,
                        title=os.path.join(save_path, title_plot))
        return original_list, imputed_list

    @torch.no_grad()
    def knn_purity(self, verbose=False):
        latent, _, labels = self.get_latent()
        score = knn_purity(latent, labels)
        if verbose:
            print("KNN purity score :", score)
        return score

    knn_purity.mode = 'max'

    @torch.no_grad()
    def clustering_scores(self, verbose=True, prediction_algorithm='knn'):
        if self.gene_dataset.n_labels > 1:
            latent, _, labels = self.get_latent()
            if prediction_algorithm == 'knn':
                labels_pred = KMeans(self.gene_dataset.n_labels, n_init=200).fit_predict(latent)  # n_jobs>1 ?
            elif prediction_algorithm == 'gmm':
                gmm = GMM(self.gene_dataset.n_labels)
                gmm.fit(latent)
                labels_pred = gmm.predict(latent)

            asw_score = silhouette_score(latent, labels)
            nmi_score = NMI(labels, labels_pred)
            ari_score = ARI(labels, labels_pred)
            uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
            if verbose:
                print("Clustering Scores:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f" %
                      (asw_score, nmi_score, ari_score, uca_score))
            return asw_score, nmi_score, ari_score, uca_score

    @torch.no_grad()
    def nn_overlap_score(self, verbose=True, **kwargs):
        '''
        Quantify how much the similarity between cells in the mRNA latent space resembles their similarity at the
        protein level. Compute the overlap fold enrichment between the protein and mRNA-based cell 100-nearest neighbor
        graph and the Spearman correlation of the adjacency matrices.
        '''
        if hasattr(self.gene_dataset, 'adt_expression_clr'):
            latent, _, _ = self.sequential().get_latent()
            protein_data = self.gene_dataset.adt_expression_clr[self.indices]
            spearman_correlation, fold_enrichment = nn_overlap(latent, protein_data, **kwargs)
            if verbose:
                print("Overlap Scores:\nSpearman Correlation: %.4f\nFold Enrichment: %.4f" %
                      (spearman_correlation, fold_enrichment))
            return spearman_correlation, fold_enrichment

    @torch.no_grad()
    def show_t_sne(self, n_samples=1000, color_by='', save_name='', latent=None, batch_indices=None,
                   labels=None, n_batch=None):
        # If no latent representation is given
        if latent is None:
            latent, batch_indices, labels = self.get_latent(sample=True)
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        if color_by == 'scalar':
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1], c=labels.ravel())
        else:
            if n_batch is None:
                n_batch = self.gene_dataset.n_batches
            if color_by == 'batches' or color_by == 'labels':
                indices = batch_indices.ravel() if color_by == 'batches' else labels.ravel()
                n = n_batch if color_by == 'batches' else self.gene_dataset.n_labels
                if hasattr(self.gene_dataset, 'cell_types') and color_by == 'labels':
                    plt_labels = self.gene_dataset.cell_types
                else:
                    plt_labels = [str(i) for i in range(n)]
                plt.figure(figsize=(10, 10))
                colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#95d0fc', '#029386', '#f97306','#96f97b',
                          '#c20078', '#ffff14', '#04d9ff', '#929591', '#bf77f6', '#00ffff', '#13eac9', '#6e750e','#06470c', '#d1b26f',
                          '#000000', '#ff028d', '#ffb07c', '#8e82fe', '#8f1402', '#658b38', '#fac205','#5b7c99', '#be0119', '#cdc50a']
                rgb_values = sns.color_palette(colors)
                # Map label to RGB
                color_map = dict(zip(plt_labels, rgb_values))
                for i, label in zip(range(n), plt_labels):
                    plt.scatter(latent[indices == i, 0], latent[indices == i, 1], label=label, c=np.array([color_map[label]]))
                if len(plt_labels)>10:
                    plt.legend(prop = {'size': 5})
                else:
                    plt.legend()
            elif color_by == 'batches and labels':
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                batch_indices = batch_indices.ravel()
                for i in range(n_batch):
                    axes[0].scatter(latent[batch_indices == i, 0], latent[batch_indices == i, 1], label=str(i))
                axes[0].set_title("batch coloring", fontsize=25)
                axes[0].axis("off")
                axes[0].legend()

                indices = labels.ravel()
                if hasattr(self.gene_dataset, 'cell_types'):
                    plt_labels = self.gene_dataset.cell_types
                else:
                    n = self.gene_dataset.n_labels
                    plt_labels = [str(i) for i in range(n)]

                colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000', '#95d0fc', '#029386', '#f97306','#96f97b',
                          '#c20078', '#ffff14', '#04d9ff', '#929591', '#bf77f6', '#00ffff', '#13eac9', '#6e750e','#06470c', '#d1b26f',
                          '#000000', '#ff028d', '#ffb07c', '#8e82fe', '#8f1402', '#658b38', '#fac205','#5b7c99', '#be0119', '#cdc50a']
                rgb_values = sns.color_palette(colors)
                # Map label to RGB
                color_map = dict(zip(plt_labels, rgb_values))
                for i, cell_type in zip(range(self.gene_dataset.n_labels), plt_labels):
                    axes[1].scatter(latent[indices == i, 0], latent[indices == i, 1], label=cell_type, c=np.array([color_map[cell_type]]))
                if len(plt_labels) > 10:
                    axes[1].legend(prop = {'size': 6})
                else:
                    axes[1].legend()
                axes[1].set_title("cell type coloring", fontsize=25)
                axes[1].axis("off")
        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)

    @staticmethod
    def apply_t_sne(latent, n_samples=1000):
        idx_t_sne = np.random.permutation(len(latent))[:n_samples] if n_samples else np.arange(len(latent))
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        return latent, idx_t_sne

    def raw_data(self):
        """
        Returns raw data for classification
        """
        return self.gene_dataset.X[self.indices], self.gene_dataset.labels[self.indices].ravel()


def entropy_from_indices(indices):
    return entropy(np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)))


def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0
    for t in range(n_pools):
        indices = np.random.choice(np.arange(latent_space.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    return score / float(n_pools)


def get_bayes_factors(px_scale, all_labels, cell_idx, other_cell_idx=None, genes_idx=None,
                      M_permutation=10000, permutation=False, sample_pairs=True):
    '''
    Returns a list of bayes factor for all genes
    :param px_scale: The gene frequency array for all cells (might contain multiple samples per cells)
    :param all_labels: The labels array for the corresponding cell types
    :param cell_idx: The first cell type population to consider. Either a string or an idx
    :param other_cell_idx: (optional) The second cell type population to consider. Either a string or an idx
    :param M_permutation: The number of permuted samples.
    :param permutation: Whether or not to permute.
    :return:
    '''
    idx = (all_labels == cell_idx)
    idx_other = (all_labels == other_cell_idx) if other_cell_idx is not None else (all_labels != cell_idx)
    if genes_idx is not None:
        px_scale = px_scale[:, genes_idx]
    sample_rate_a = px_scale[idx].reshape(-1, px_scale.shape[1])
    sample_rate_b = px_scale[idx_other].reshape(-1, px_scale.shape[1])

    # agregate dataset
    samples = np.vstack((sample_rate_a, sample_rate_b))

    if sample_pairs is True:
        # prepare the pairs for sampling
        list_1 = list(np.arange(sample_rate_a.shape[0]))
        list_2 = list(sample_rate_a.shape[0] + np.arange(sample_rate_b.shape[0]))
        if not permutation:
            # case1: no permutation, sample from A and then from B
            u, v = np.random.choice(list_1, size=M_permutation), np.random.choice(list_2, size=M_permutation)
        else:
            # case2: permutation, sample from A+B twice
            u, v = (np.random.choice(list_1 + list_2, size=M_permutation),
                    np.random.choice(list_1 + list_2, size=M_permutation))

        # then constitutes the pairs
        first_set = samples[u]
        second_set = samples[v]
    else:
        first_set = sample_rate_a
        second_set = sample_rate_b
    res = np.mean(first_set >= second_set, 0)
    res = np.log(res + 1e-8) - np.log(1 - res + 1e-8)
    return res

def plot_imputation(original, imputed, show_plot=True, title="Imputation"):
    y = imputed
    x = original

    ymax = 10
    mask = x < ymax
    x = x[mask]
    y = y[mask]

    mask = y < ymax
    x = x[mask]
    y = y[mask]

    l_minimum = np.minimum(x.shape[0], y.shape[0])

    x = x[:l_minimum]
    y = y[:l_minimum]

    data = np.vstack([x, y])

    plt.figure(figsize=(5, 5))

    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])

    nbins = 50

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins * 1j, 0:ymax:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title(title, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(y[:, np.newaxis], x, rcond=-1)
    linspace = np.linspace(0, ymax)
    plt.plot(linspace, a * linspace, color='black')

    plt.plot(linspace, linspace, color='black', linestyle=":")
    if show_plot:
        plt.show()
    plt.savefig(title + '.png')


def nn_overlap(X1, X2, k=100):
    '''
    Compute the overlap between the k-nearest neighbor graph of X1 and X2 using Spearman correlation of the
    adjacency matrices.
    '''
    assert len(X1) == len(X2)
    n_samples = len(X1)
    k = min(k, n_samples - 1)
    nne = NearestNeighbors(n_neighbors=k + 1)  # "n_jobs=8
    nne.fit(X1)
    kmatrix_1 = nne.kneighbors_graph(X1) - scipy.sparse.identity(n_samples)
    nne.fit(X2)
    kmatrix_2 = nne.kneighbors_graph(X2) - scipy.sparse.identity(n_samples)

    # 1 - spearman correlation from knn graphs
    spearman_correlation = scipy.stats.spearmanr(kmatrix_1.A.flatten(), kmatrix_2.A.flatten())[0]
    # 2 - fold enrichment
    set_1 = set(np.where(kmatrix_1.A.flatten() == 1)[0])
    set_2 = set(np.where(kmatrix_2.A.flatten() == 1)[0])
    fold_enrichment = len(set_1.intersection(set_2)) * n_samples ** 2 / (float(len(set_1)) * len(set_2))
    return spearman_correlation, fold_enrichment


def unsupervised_clustering_accuracy(y, y_pred):
    """
    Unsupervised Clustering Accuracy
    """
    assert len(y_pred) == len(y)
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    row_ind, col_ind = linear_assignment(cost_matrix)
    ind = np.concatenate((row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)), axis=1)
    return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

def knn_purity(latent, label, n_neighbors=30):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: label[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - label.reshape(-1, 1)) == 0).mean(axis=1)
    res = [np.mean(scores[label == i]) for i in np.unique(label)]  # per cell-type purity

    return np.mean(res)


def proximity_imputation(real_latent1, normed_gene_exp_1, real_latent2, k=4):
    knn = KNeighborsRegressor(k, weights='distance')
    y = knn.fit(real_latent1, normed_gene_exp_1).predict(real_latent2)
    return y
