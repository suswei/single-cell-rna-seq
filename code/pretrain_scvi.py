# Run 100 Monte Carlo Samples for each dataset, as scVI is variable for each run.
# For mouse marrow dataset, the hyperparameter tried n_layers for scvi=2, n_hidden=128, n_latent=10, reconstruction_loss=zinb, dropout_rate=0.1, lr=0.001, n_epochs=250, training_size-0.8
# For Pbmc dataset, the hyperparameter tried n_layers_for_scvi=1, n_latent=256, n_latent=14, dropout_rate=0.5, lr=0.01, n_epochs=170, training_size=0.8.
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scvi.dataset import *
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.models.modules import MINE_Net4_3, Classifier_Net
from scvi.inference import UnsupervisedTrainer
import torch
from torch.autograd import Variable
import itertools
from tqdm import tqdm


def main(dataset_name, nuisance_variable, adv_model, taskid):
    # taskid is just any integer from 0 to 99
    # dataset_name could be 'muris_tabula', 'pbmc'
    # nuisance_variable could be 'batch'
    # MI_estimator could be 'Mine_Net4', 'NN' (NN stands for nearest neighbor), 'aggregated_posterior'

    if not os.path.exists('./data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('./data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))
    if not os.path.exists('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))

    if dataset_name == 'muris_tabula' and nuisance_variable == 'batch' and adv_model == 'MI':
        hyperparameter_config = {
            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'pre_n_epochs': [100]
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    data_save_path = './data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)
    result_save_path = './result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)

    if dataset_name == 'muris_tabula':
        dataset1 = TabulaMuris('facs', save_path=data_save_path)
        dataset2 = TabulaMuris('droplet', save_path=data_save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    elif dataset_name == 'pbmc':
        gene_dataset = PbmcDataset(save_path=data_save_path)
    elif dataset_name == 'retina':
        gene_dataset = RetinaDataset(save_path=data_save_path)

    key, value = zip(*hyperparameter_experiments[0].items())
    n_layers_encoder = value[0]
    n_layers_decoder = value[1]
    n_hidden = value[2]
    n_latent = value[3]
    dropout_rate = value[4]
    reconstruction_loss = value[5]
    use_batches = value[6]
    use_cuda = value[7]
    train_size = value[8]
    pre_n_epochs = value[9]

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100), dtype=np.uint32)

    taskid = int(taskid)
    desired_seed = int(desired_seeds[0, taskid])

    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                    n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder, n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,
                    reconstruction_loss=reconstruction_loss, MI_estimator=adv_model, adv=False, save_path='None',
                    std=False, mini_ELBO=12000, max_ELBO=16000)  # mini_ELBO=15000, max_ELBO=20000
    trainer_vae_MI = UnsupervisedTrainer(vae_MI, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)

    vae_MI_file_path = '%s/%s_%s_taskid%s_VaeMI.pth' % (data_save_path, dataset_name, nuisance_variable, taskid)
    # Pretrain trainer_vae_MI.vae_MI when adv=False
    trainer_vae_MI.train(n_epochs=pre_n_epochs, lr=1e-3)
    torch.save(trainer_vae_MI.model.state_dict(), vae_MI_file_path)

# Run the actual program
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
