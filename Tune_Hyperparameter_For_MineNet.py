#!/usr/bin/env bash
n_epochs_all = None
show_plot = True



import os
os.chdir(r'D:\UMelb\PhD_Projects\Project1_Modify_SCVI')
save_path = 'data/2019-05-01/'

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

n_hidden_z_array = np.asarray([10,30])#I already use the default n_hidden_z = 5 to run, and the result is not improved much. Therefore, I use bigger two bigger values, one is small, the other is big
n_layers_z_array = np.asarray([3,10])
MineLoss_Scale_array = np.asarray([1000,5000,10000,50000,100000])
n_hidden_z_Mesh, n_layers_z_Mesh, MineLoss_Scale_Mesh = np.meshgrid(n_hidden_z_array, n_layers_z_array, MineLoss_Scale_array)

pbmc_dataset = PbmcDataset(save_path=save_path)

datasets_dict = {'pbmc': pbmc_dataset}

for i in range(n_hidden_z_Mesh.shape[0]):
    for j in range(n_hidden_z_Mesh.shape[1]):
        for k in range(n_hidden_z_Mesh.shape[2]):
             n_hidden_z = n_hidden_z_Mesh[i,j,k]
             n_layers_z = n_layers_z_Mesh[i,j,k]
             MineLoss_Scale = MineLoss_Scale_Mesh[i,j,k]

             for key,dataset in datasets_dict.items():
                    # Training
                    n_epochs=400 if n_epochs_all is None else n_epochs_all
                    lr=0.0005
                    use_batches=True
                    use_cuda=False
                    train_size = 0.6

                    # different models

                    vae_mine = VAE_MINE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches,n_hidden_z = n_hidden_z, n_layers_z = n_layers_z, MineLoss_Scale = MineLoss_Scale)
                    # TODO: sweep over kl penalty parameter
                    # trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5, kl=100 )
                    trainer_vae_mine = UnsupervisedTrainer(vae_mine, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5, kl=1)
                    trainer_vae_mine.train(n_epochs=n_epochs, lr=lr)

                    vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
                    trainer_vae = UnsupervisedTrainer(vae, dataset, train_size=train_size, use_cuda=use_cuda, frequency=5)
                    trainer_vae.train(n_epochs=n_epochs, lr=lr)

                    # visualize results
                    n_samples_tsne = 1000

                    trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-01\\trainset_tsne_SCVI+MINE_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
                    plt.show()
                    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='.\\result\\2019-05-01\\trainset_tsne_SCVI_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
                    plt.show()

                    trainer_vae_mine.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-01\\testset_tsne_SCVI+MINE_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
                    plt.show()
                    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='.\\result\\2019-05-01\\testset_tsne_SCVI_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
                    plt.show()

                    # clustering_scores() -- these metrics measure clustering performance
                    #   silhouette width (asw, higher is better),
                    #   normalised mutual information (nmi, higher is better),
                    #   adjusted rand index (ari, higher is better),
                    #   unsupervised clustering accuracy (uca)
                    #   entropy_batch_mixing() -- this metric measures batch effect
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
                    barplot_list(train_results, alg, 'Clustering metrics train set {}'.format(key), save='.\\result\\2019-05-01\\trainset_clustering_metrics_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
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
                    barplot_list(test_results, alg, 'Clustering metrics test set{}'.format(key), save='.\\result\\2019-05-01\\testset_clustering_metrics_{}_Hidden{}_layers{}_MineLossScale{}'.format(key,n_hidden_z,n_layers_z,MineLoss_Scale))
                    plt.show()
