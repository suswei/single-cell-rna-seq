n_epochs_all = None
show_plot = True

import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, RetinaDataset
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch

retina_dataset = RetinaDataset()

# Training
n_epochs=200 if n_epochs_all is None else n_epochs_all
lr=0.0005
use_batches=True
use_cuda=True

vae = VAE_mine(retina_dataset.nb_genes, n_batch=retina_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae,
                              retina_dataset,
                              train_size=0.1,
                              use_cuda=use_cuda,
                              frequency=5)

trainer.train(n_epochs=n_epochs, lr=lr)

n_samples_tsne = 1000
trainer.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels')


asw, nmi, ari, uca = trainer.train_set.clustering_scores()
be = trainer.train_set.entropy_batch_mixing()

# In scVI-reproducibility
# asw = 0.36126289
# ari = 0.83652908297624851
# nmi = 0.61592297966666154
# be = 0.61157293916
basw = 0.0185152 # not computed here

retina_data = np.array([[0.21954027735402515, 0.58685860355428821, 0.36856986097377797, 0.192647888467, 0.0624201044534],\
        [0.34417779654030661, 0.64755125928853896, 0.51136144259466121, 0.597048161687, 0.0409665873338],\
        [0.44694698300835106, 0.794144880698191, 0.47583467153476738, 0.34069311837, 0.019781812657], \
        [asw,ari,nmi,be,basw]])
alg = ["PCA", "SIMLR", "Combat + PCA", "scVI"]

barplot_list(retina_data, alg, 'Clustering metrics (RETINA)')
