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

dataset_names = ['hemato', 'pbmc','retina']

for dataset_name in dataset_names:

    if dataset_name == 'retina':
        dataset = RetinaDataset()
    elif dataset_name == 'pbmc':
        dataset = PbmcDataset()
    elif dataset_name == 'hemato':
        dataset = HematoDataset()

    # Training
    n_epochs=200 if n_epochs_all is None else n_epochs_all
    lr=0.0005
    use_batches=True
    use_cuda=True
    train_size = 0.9

    # different models
    vae_mine = VAE_MINE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
    trainer_vae_mine.train(n_epochs=n_epochs, lr=lr)

    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
    trainer_vae = UnsupervisedTrainer(vae, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
    trainer_vae.train(n_epochs=n_epochs, lr=lr)

    # visualize results
    n_samples_tsne = 1000
    trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels')
    plt.show()
    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels')
    plt.show()

    # clustering_scores():
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    # entropy_batch_mixing():
    #   entropy batch mixing (be, , higher is better)

    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()
    results = np.array([asw, nmi, ari, uca, be])

    asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
    be = trainer_vae_mine.train_set.entropy_batch_mixing()
    results.append([asw, nmi, ari, uca, be])

    alg = ["scVI", "scVI+MINE"]

    barplot_list(results, alg, 'Clustering metrics %s'.format(dataset_name))
    plt.show()
