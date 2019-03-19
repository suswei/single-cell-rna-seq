n_epochs_all = None
show_plot = True



import os
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


retina_dataset = RetinaDataset()
# TODO: why won't pbmc load?
# pbmc_dataset = PbmcDataset()
# TODO: weird error with calling clustering_scores on hemato trained vae
# hemato_dataset = HematoDataset()


datasets_dict = {'retina': retina_dataset}

for key,dataset in datasets_dict.items():


    # Training
    n_epochs=200 if n_epochs_all is None else n_epochs_all
    lr=0.0005
    use_batches=True
    use_cuda=False
    train_size = 0.9

    # different models
    vae_mine = VAE_MINE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    # TODO: sweep over kl penalty parameter
    # trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5, kl=100 )
    trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
    trainer_vae_mine.train(n_epochs=n_epochs, lr=lr)

    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    trainer_vae = UnsupervisedTrainer(vae, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
    trainer_vae.train(n_epochs=n_epochs, lr=lr)

    # visualize results
    n_samples_tsne = 1000
    trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='tsne_SCVI+MINE_{}'.format(key))
    plt.show()
    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='tsne_SCVI_{}'.format(key))
    plt.show()

    # TODO: figure out which of these should be high if batch effect is removed successfully, presumably some of these are about clustering the cell types correctly
    # clustering_scores():
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    # entropy_batch_mixing():
    #   entropy batch mixing (be, higher is better)

    results = np.empty((0, 5), int)

    print('scVI')
    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()
    results = np.append(results, np.array([[asw, nmi, ari, uca, be]]), axis=0)

    print('scVI+MINE')
    asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
    be = trainer_vae_mine.train_set.entropy_batch_mixing()
    results = np.append(results, np.array([[asw, nmi, ari, uca, be]]), axis=0)

    alg = ["scVI", "scVI+MINE"]

    barplot_list(results, alg, 'Clustering metrics {}'.format(key), save = 'clustering_metrics_{}'.format(key))
    plt.show()
