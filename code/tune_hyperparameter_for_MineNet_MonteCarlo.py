#Run 100 Monte Carlo Samples for each dataset, as scVI is variable for each run.
#For mouse marrow dataset, the hyperparameter tried n_layers for scvi=2, n_hidden=128, n_latent=10, reconstruction_loss=zinb, dropout_rate=0.1, lr=0.001, n_epochs=250, training_size-0.8
#For Pbmc dataset, the hyperparameter tried n_layers_for_scvi=1, n_latent=256, n_latent=14, dropout_rate=0.5, lr=0.01, n_epochs=170, training_size=0.8.
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scvi.dataset import *
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch
import itertools

def main(taskid):
    if not os.path.exists('data/tune_hyperparameter_for_MineNet/pbmc'):
        os.makedirs('data/tune_hyperparameter_for_MineNet/pbmc')
    if not os.path.exists('result/tune_hyperparameter_for_MineNet/pbmc'):
        os.makedirs('result/tune_hyperparameter_for_MineNet/pbmc')

    hyperparameter_config = {
        'dataset_name': ['Pbmc'],
        'nuisance_factor': ['batch'],
        'MineLoss_Scale': [200, 500, 800, 1000, 2000, 5000,10000,100000]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    save_path = './data/tune_hyperparameter_for_MineNet/pbmc'

    gene_dataset = PbmcDataset(save_path=save_path)
    #gene_dataset = RetinaDataset(save_path=save_path)
    #dataset1 = TabulaMuris('facs',save_path = save_path)
    #dataset2 = TabulaMuris('droplet',save_path = save_path)
    #dataset1.subsample_genes(dataset1.nb_genes)
    #dataset2.subsample_genes(dataset2.nb_genes)
    #gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100),dtype=np.uint32)

    taskid = int(taskid[0])
    desired_seed = int(desired_seeds[0,taskid])

    n_epochs_all = None
    show_plot = True
    n_epochs = 170 if n_epochs_all is None else n_epochs_all
    lr = 0.01
    use_batches = True
    use_cuda = False
    train_size = 0.8

    n_samples_tsne = 1000
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be'])

    for n_layer in [1]:
        for i in range(len(hyperparameter_experiments)):
            key, value = zip(*hyperparameter_experiments[i].items())
            dataset_name = value[0]
            nuisance_variable = value[1]
            MineLoss_Scale = value[2]

            vae_mine = VAE_MINE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels, n_hidden=256, n_latent=14, n_layers = n_layer,dropout_rate = 0.5, MineLoss_Scale=MineLoss_Scale)
            trainer_vae_mine = UnsupervisedTrainer(vae_mine, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda,frequency=5, kl=1)
            vae_mine_file_path = '%s/%s_%s_n_layers%s_MineLossScale%s_sample%s_VaeMine.pk1'%(save_path, dataset_name, nuisance_variable, n_layer, MineLoss_Scale, taskid)

            if os.path.isfile(vae_mine_file_path):
                trainer_vae_mine.model.load_state_dict(torch.load(vae_mine_file_path))
                trainer_vae_mine.model.eval()
            else:
                trainer_vae_mine.train(n_epochs=n_epochs, lr=lr)
                torch.save(trainer_vae_mine.model.state_dict(), vae_mine_file_path)
                ll_train_set = trainer_vae_mine.history["ll_train_set"]
                ll_test_set = trainer_vae_mine.history["ll_test_set"]
                x = np.linspace(0, 500, (len(ll_train_set)))

                fig = plt.figure(figsize=(14, 7))
                plt.plot(x, ll_train_set)
                plt.plot(x, ll_test_set)
                plt.ylim(1150, 1600)
                plt.title("Blue for training error and orange for testing error")

                fig1_path = 'result/tune_hyperparameter_for_MineNet/pbmc/training_testing_error_SCVI+MINE_{}_{}_n_layers{}_sample{}_MineLossScale{}.png'.format(dataset_name,nuisance_variable, n_layer, taskid, MineLoss_Scale)
                fig.savefig(fig1_path)
                plt.close(fig)

            trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='result/tune_hyperparameter_for_MineNet/pbmc/trainset_tsne_SCVI+MINE_{}_{}_n_layers{}_sample{}_MineLossScale{}'.format(dataset_name,nuisance_variable, n_layer, taskid, MineLoss_Scale))
            trainer_vae_mine.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/tune_hyperparameter_for_MineNet/pbmc/testset_tsne_SCVI+MINE_{}_{}_n_layers{}_sample{}_MineLossScale{}'.format(dataset_name,nuisance_variable, n_layer, taskid, MineLoss_Scale))

            asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
            be = trainer_vae_mine.train_set.entropy_batch_mixing()
            label = '%s_%s_n_layers%s_sample%s_MineLossScale%s_VaeMine_trainset'%(dataset_name, nuisance_variable, n_layer, taskid, MineLoss_Scale)
            intermediate_dataframe1 = pd.DataFrame.from_dict({'Label':[label],'asw':[asw],'nmi':[nmi],'ari':[ari],'uca':[uca],'be':[be]})
            clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

            asw, nmi, ari, uca = trainer_vae_mine.test_set.clustering_scores()
            be = trainer_vae_mine.test_set.entropy_batch_mixing()
            label = '%s_%s_n_layers%s_sample%s_MineLossScale%s_VaeMine_testset'%(dataset_name, nuisance_variable, n_layer, taskid, MineLoss_Scale)
            intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
            clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)
            clustering_metric.to_csv('result/tune_hyperparameter_for_MineNet/pbmc/%s_%s_n_layers%s_sample%s_ClusterMetric.csv' % (dataset_name, nuisance_variable, n_layer, taskid), index=None, header=True)

        vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels, n_hidden=256, n_latent=14, n_layers = n_layer,dropout_rate = 0.5)
        trainer_vae = UnsupervisedTrainer(vae, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5)
        vae_file_path = '%s/%s_%s_n_layers%s_sample%s_Vae.pk1'%(save_path,dataset_name, nuisance_variable, n_layer, taskid)

        if os.path.isfile(vae_file_path):
            trainer_vae.model.load_state_dict(torch.load(vae_file_path))
            trainer_vae.model.eval()
        else:
            trainer_vae.train(n_epochs=n_epochs, lr=lr)
            torch.save(trainer_vae.model.state_dict(), vae_file_path)

        ll_train_set = trainer_vae.history["ll_train_set"]
        ll_test_set = trainer_vae.history["ll_test_set"]
        x = np.linspace(0, 500, (len(ll_train_set)))

        fig = plt.figure(figsize=(14, 7))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        plt.ylim(1150, 1600)
        plt.title("Blue for training error and orange for testing error")

        fig2_path = 'result/tune_hyperparameter_for_MineNet/pbmc/training_testing_error_SCVI_{}_{}_n_layers{}_sample{}.png'.format(dataset_name, nuisance_variable, n_layer, taskid)
        fig.savefig(fig2_path)
        plt.close(fig)

        trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/tune_hyperparameter_for_MineNet/pbmc/trainset_tsne_SCVI_{}_{}_n_layers{}_sample{}'.format(dataset_name,nuisance_variable, n_layer, taskid))
        trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/tune_hyperparameter_for_MineNet/pbmc/testset_tsne_SCVI_{}_{}_n_layers{}_sample{}'.format(dataset_name,nuisance_variable, n_layer, taskid))

        #   clustering_scores() -- these metrics measure clustering performance
        #   silhouette width (asw, higher is better),
        #   normalised mutual information (nmi, higher is better),
        #   adjusted rand index (ari, higher is better),
        #   unsupervised clustering accuracy (uca)
        #   entropy_batch_mixing() -- this metric measures batch effect
        #   entropy batch mixing (be, higher is better meaning less batch effect)


        asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
        be = trainer_vae.train_set.entropy_batch_mixing()
        label = '%s_%s_n_layers%s_sample%s_Vae_trainset' % (dataset_name, nuisance_variable, n_layer, taskid)
        intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

        asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
        be = trainer_vae.test_set.entropy_batch_mixing()
        label = '%s_%s_n_layers%s_sample%s_Vae_testset' % (dataset_name, nuisance_variable,n_layer, taskid)
        intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)

        clustering_metric.to_csv('result/tune_hyperparameter_for_MineNet/pbmc/%s_%s_n_layers%s_sample%s_ClusterMetric.csv'%(dataset_name, nuisance_variable, n_layer, taskid), index=None, header=True)

# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1:])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
