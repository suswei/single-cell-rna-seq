# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:54:30 2019

@author: huli2
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

os.getcwd()
if not os.path.exists('./data/2019-04-11'):
    os.makedirs('./data/2019-04-11')

save_path = 'data/2019-04-11/'

from scvi.dataset import *
from scvi.models import *
from scvi.inference import UnsupervisedTrainer

hemato_dataset = HematoDataset(save_path=os.path.join(save_path, 'HEMATO/'))

n_epochs_all = None
show_plot = True

n_epochs = 200 if n_epochs_all is None else n_epochs_all
lr = 0.0004
use_batches = False
use_cuda = True

# train the model
hemato_vae = VAE(hemato_dataset.nb_genes, n_batch=hemato_dataset.n_batches * use_batches)
hemato_trainer = UnsupervisedTrainer(hemato_vae,
                                     hemato_dataset,
                                     train_size=0.9,
                                     use_cuda=use_cuda,
                                     frequency=5)
hemato_trainer.train(n_epochs=n_epochs, lr=lr)

asw, nmi, ari, uca = hemato_trainer.train_set.clustering_scores()

smfish_dataset = SmfishDataset(save_path=save_path)
brain_large_dataset = BrainLargeDataset(save_path=save_path)
cortex_dataset = CortexDataset(save_path=save_path)

pbmc_dataset = PbmcDataset(save_path=save_path)

retina_dataset = RetinaDataset(save_path=save_path)

cbmc_dataset = CbmcDataset(save_path=os.path.join(save_path, "citeSeq/"))

brain_small_dataset = BrainSmallDataset(save_path=save_path)
