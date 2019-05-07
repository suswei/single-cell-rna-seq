n_epochs_all = None
show_plot = True



import os
os.getcwd()
if not os.path.exists('./data/2019-04-11'):
    os.makedirs('./data/2019-04-11')

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import *
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch

def barplot_list(data, alg, title, save=None, interest=0, prog=False, figsize=None):
    ind = np.arange(len(alg))  # the x locations for the groups
    width = 0.25  # the width of the bars
    if figsize is None:
        fig = plt.figure()

    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if len(data[0]) == 3:
        width = 0.25  # the width of the bars

    else:
        width = 0.15

    rects = []
    color = ["r", "g", "y", "b", "purple"]
    if prog:
        color = ['darkred', "red", "tomato", "salmon"]
    for i in range(len(data[0])):
        rects.append(ax.barh(ind + i * width, data[:, i], width, color=color[i]))

    anchor_param = (0.8, 0.8)
    leg_rec = [x[0] for x in rects]
    leg_lab = ('ASW', 'NMI', 'ARI', "UCA", "BE")
    if prog:
        leg_lab = ["2", "3", "4", "7"]
    ax.legend(leg_rec, leg_lab[:len(data[0])])

    # add some text for labels, title and axes ticks
    ax.set_xlabel(title)
    ax.set_yticks(ind + width)
    ax.set_yticklabels(alg)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)


retina_dataset = RetinaDataset(save_path=save_path)
# TODO: HUI: why won't pbmc load? : delete line 131 in dataset.py because self._X has already been subsetted.
pbmc_dataset = PbmcDataset(save_path=save_path)
# TODO: HUI: weird error with calling clustering_scores on hemato trained vae :
#            The error also happens when use the original SCVI, the error is ValueError: n_samples=3614 should be >= n_clusters=3979
#            The reason for the error is that there is something wrong with the 'labels' and 'n_labels' attributes for the hemato_dataset
# hemato_dataset = HematoDataset()


#datasets_dict = {'retina': retina_dataset}
datasets_dict = {'retina': retina_dataset, 'pbmc': pbmc_dataset}
# TODO: HUI: make sure the code below runs with pbmc and hemato. also are there other built-in datasets we can use?
#            smfish_dataset: 4462 cells, 33 genes, 1 batch, 6 cell types
#            brain_large_dataset: 1303182 cells, 720 genes, 1 batch, 1 cell type
#            cortex_dataset: 3005 cells, 558 genes, 1 batch, 7 cell types
#            pbmc_dataset: 11990 cells, 3346 genes, 2 batches, 9 cell types
#            retina_dataset: 19829 cells, 13166 genes, 2 batches, 15 cell types
#            hemato_dataset: 4016 cells, 7397 genes, 1 batch, 3979 cell types, I think hemato dataset can not be used.
#            cbmc_dataset: 8617 cells, 20400 genes, 1 batch, 1 cell type
#            brain_small_dataset: 9128 cells, 27998 genes, 1 batch, 10 cell typess


for key,dataset in datasets_dict.items():


    # Training
    n_epochs=200 if n_epochs_all is None else n_epochs_all
    lr=0.0005
    use_batches=True
    use_cuda=False
    train_size = 0.2

    # different models

    vae_mine = VAE_MINE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    # TODO: sweep over kl penalty parameter
    # trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5, kl=100 )
    trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5, kl=1)
    trainer_vae_mine.train(n_epochs=n_epochs, lr=lr)

    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    trainer_vae = UnsupervisedTrainer(vae, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
    trainer_vae.train(n_epochs=n_epochs, lr=lr)

    # visualize results
    n_samples_tsne = 1000

    trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-04-11\\trainset_tsne_SCVI+MINE_{}'.format(key))
    plt.show()
    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='.\\result\\2019-04-11\\trainset_tsne_SCVI_{}'.format(key))
    plt.show()

    trainer_vae_mine.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-04-11\\testset_tsne_SCVI+MINE_{}'.format(key))
    plt.show()
    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='.\\result\\2019-04-11\\testset_tsne_SCVI_{}'.format(key))
    plt.show()

    # clustering_scores() -- these metrics measure clustering performance
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    # entropy_batch_mixing() -- this metric measures batch effect
    #   entropy batch mixing (be, higher is better meaning less batch effect)

    train_results = np.empty((0, 5), int)
    print('scVI: train set')
    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()
    train_results = np.append(train_results, np.array([[asw, nmi, ari, uca, be]]), axis=0)
    print('scVI+MINE: train set')
    asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
    be = trainer_vae_mine.train_set.entropy_batch_mixing()
    train_results = np.append(train_results, np.array([[asw, nmi, ari, uca, be]]), axis=0)
    alg = ["scVI", "scVI+MINE"]
    barplot_list(train_results, alg, 'Clustering metrics train set {}'.format(key), save='.\\result\\2019-04-11\\trainset_clustering_metrics_{}'.format(key))
    plt.show()


    test_results = np.empty((0, 5), int)
    print('scVI: test set')
    asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
    be = trainer_vae.test_set.entropy_batch_mixing()
    test_results = np.append(test_results, np.array([[asw, nmi, ari, uca, be]]), axis=0)
    print('scVI+MINE: test set')
    asw, nmi, ari, uca = trainer_vae_mine.test_set.clustering_scores()
    be = trainer_vae_mine.test_set.entropy_batch_mixing()
    test_results = np.append(test_results, np.array([[asw, nmi, ari, uca, be]]), axis=0)
    alg = ["scVI", "scVI+MINE"]
    barplot_list(test_results, alg, 'Clustering metrics test set{}'.format(key), save='.\\result\\2019-04-11\\testset_clustering_metrics_{}'.format(key))
    plt.show()
