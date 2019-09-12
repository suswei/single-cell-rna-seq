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
from scvi.models.modules import MINE_Net4_3, discrete_continuous_info, Sample_From_Aggregated_Posterior
from scvi.inference import UnsupervisedTrainer
import torch
from torch.autograd import Variable
import itertools

def main(taskid, dataset_name, nuisance_variable, config_id):
    # taskid is just any integer from 0 to 99
    # dataset_name could be 'muris_tabula', 'pbmc'
    # nuisance_variable could be 'batch'
    # MI_estimator could be 'Mine_Net4', 'NN' (NN stands for nearest neighbor), 'aggregated_posterior'

    if not os.path.exists('../data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('../data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))
    if not os.path.exists('../result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('../result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))

    if dataset_name == 'muris_tabula' and nuisance_variable == 'batch':
        hyperparameter_config = {
            'n_layers_encoder': [2],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [5e-3,1e-4,1e-5,5e-6,1e-6],
            'n_epochs': [1500],
            'nsamples_z': [200],
            'adv': [False],
        }
    elif dataset_name == 'pbmc' and nuisance_variable == 'batch':
        hyperparameter_config = {
            'n_layers_encoder': [1],
            'n_layers_decoder': [1],
            'n_hidden': [256],
            'n_latent': [14],
            'dropout_rate': [0.5],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'MIScale': [200, 500, 800, 1000, 2000, 5000, 10000, 100000],
            'train_size': [0.8],
            'lr': [0.01],
            'adv_lr': [0.001],
            'n_epochs': [170],
            'nsamples_z': [200],
            'adv': [False]
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    data_save_path = '../data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)
    result_save_path = '../result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)

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

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100), dtype=np.uint32)

    taskid = int(taskid)
    desired_seed = int(desired_seeds[0, taskid])

    n_samples_tsne = 1000
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be'])

    config_id = int(config_id)

    key, value = zip(*hyperparameter_experiments[config_id].items())
    n_layers_encoder = value[0]
    n_layers_decoder = value[1]
    n_hidden = value[2]
    n_latent = value[3]
    dropout_rate = value[4]
    reconstruction_loss = value[5]
    use_batches = value[6]
    use_cuda = value[7]
    train_size = value[8]
    lr = value[9]  # 0.0005
    n_epochs = value[10]  # 500
    nsamples_z = value[11]
    adv = value[12]

    if not os.path.exists('../result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/scviconfig%s' % (dataset_name, config_id)):
        os.makedirs('../result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/scviconfig%s' % (dataset_name, config_id))

    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
              n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder,
              n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,
              reconstruction_loss=reconstruction_loss, nsamples_z=nsamples_z, adv=adv, save_path=result_save_path+'/scviconfig%s/'%(config_id))
    trainer_vae = UnsupervisedTrainer(vae, gene_dataset, train_size=train_size, seed=desired_seed,use_cuda=use_cuda, frequency=5, kl=1)

    vae_file_path = '%s/%s_%s_config%s_Vae.pk1' % (data_save_path, dataset_name, nuisance_variable, config_id)

    if os.path.isfile(vae_file_path):
        trainer_vae.model.load_state_dict(torch.load(vae_file_path))
        trainer_vae.model.eval()
    else:
        reconst_loss_list, clustermetrics_trainingprocess = trainer_vae.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer_vae.model.state_dict(), vae_file_path)
        ll_train_set = trainer_vae.history["ll_train_set"]
        ll_test_set = trainer_vae.history["ll_test_set"]
        x = np.linspace(0, 500, (len(ll_train_set)))

        fig = plt.figure(figsize=(14, 7))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        if dataset_name=='muris_tabula':
           plt.ylim(13000, 25000)
        elif dataset_name=='pbmc':
           plt.ylim(1150, 1600)
        plt.title("Blue for training error and orange for testing error")

        fig1_path = '%s/scviconfig%s/training_testing_error_SCVI_%s_%s_config%s.png'%(result_save_path,config_id, dataset_name,nuisance_variable, config_id)
        fig.savefig(fig1_path)
        plt.close(fig)
    clustermetrics_trainingprocess.to_csv('%s/scviconfig%s/%s_%s_config%s_clustermetrics_duringtraining.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id),index=None, header=True)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(reconst_loss_list))], [np.mean(i) for i in reconst_loss_list])
    plt.ylim(12000, 60000)
    plt.title("reconst_loss_%s_%s_config%s"%(dataset_name, nuisance_variable, config_id))
    fig1_path = '%s/scviconfig%s/reconst_loss_%s_%s_config%s.png' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id)
    fig.savefig(fig1_path)
    plt.close(fig)

    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/config%s/trainset_tsne_SCVI+MI_%s_%s_config%s' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id))
    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/config%s/testset_tsne_SCVI+MI_%s_%s_config%s' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id))

    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()

    label = '%s_%s_config%s_Vae_trainset' % (dataset_name, nuisance_variable, config_id)

    intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

    asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
    be = trainer_vae.test_set.entropy_batch_mixing()

    label = '%s_%s_config%s_Vae_testset' % (dataset_name, nuisance_variable, config_id)
    intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)
    clustering_metric.to_csv('%s/scviconfig%s/%s_%s_config%s_ClusterMetric.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id), index=None, header=True)

# Run the actual program
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
