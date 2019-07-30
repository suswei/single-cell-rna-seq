#Run 100 Monte Carlo Samples for each dataset, as scVI is variable for each run.
#For mouse marrow dataset, the hyperparameter tried n_layers for scvi=2, n_hidden=128, n_latent=10, reconstruction_loss=zinb, dropout_rate=0.1, lr=0.001, n_epochs=250, training_size-0.8
#For Pbmc dataset, the hyperparameter tried n_layers_for_scvi=1, n_latent=256, n_latent=14, dropout_rate=0.5, lr=0.01, n_epochs=170, training_size=0.8.
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
from scvi.models.modules import discrete_continuous_info, Sample_From_Aggregated_Posterior
from scvi.inference import UnsupervisedTrainer
import torch
from torch.autograd import Variable
import itertools

def main(taskid, dataset_name, nuisance_variable, MI_estimator):

    # taskid is just any integer from 0 to 99
    # dataset_name could be 'muris_tabula', 'pbmc'
    # nuisance_variable could be 'batch'
    # MI_estimator could be 'Mine_Net4', 'NN' (NN stands for nearest neighbor), 'aggregated_posterior'

    if not os.path.exists('data/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name)):
        os.makedirs('data/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name))
    if not os.path.exists('result/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name)):
        os.makedirs('result/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name))

    if dataset_name=='muris_tabula' and nuisance_variable=='batch':
        hyperparameter_config = {
            'n_layers_encoder': [2],
            'n_layers_decoder': [2],
            'n_hidden' : [128],
            'n_latent' : [10],
            'dropout_rate' : [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'MIScale': [500, 1000, 5000, 10000, 100000, 1000000,10000000],
            'train_size': [0.8],
            'lr': [0.001],
            'n_epochs' : [250],
        }
    elif dataset_name=='pbmc' and nuisance_variable=='batch':
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
            'n_epochs': [170],
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    data_save_path = './data/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name)
    result_save_path = './result/tune_hyperparameter_for_SCVI_MI/%s'%(dataset_name)

    if dataset_name=='muris_tabula':
        dataset1 = TabulaMuris('facs',save_path = data_save_path)
        dataset2 = TabulaMuris('droplet',save_path = data_save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    elif dataset_name=='pbmc':
        gene_dataset = PbmcDataset(save_path=data_save_path)
    elif dataset_name=='retina':
        gene_dataset = RetinaDataset(save_path=data_save_path)

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100),dtype=np.uint32)

    taskid = int(taskid)
    desired_seed = int(desired_seeds[0,taskid])

    n_samples_tsne = 1000
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be','MILoss'])

    for i in range(len(hyperparameter_experiments)):
        key, value = zip(*hyperparameter_experiments[i].items())
        n_layers_encoder = value[0]
        n_layers_decoder = value[1]
        n_hidden = value[2]
        n_latent = value[3]
        dropout_rate = value[4]
        reconstruction_loss =  value[5]
        use_batches = value[6]
        use_cuda = value[7]
        MIScale = value[8]
        train_size = value[9]
        lr = value[10]
        n_epochs = value[11]

        vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels, n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder = n_layers_encoder, n_layers_decoder=n_layers_decoder, dropout_rate = dropout_rate, reconstruction_loss=reconstruction_loss, MI_estimator=MI_estimator, MIScale=MIScale)
        trainer_vae_MI = UnsupervisedTrainer(vae_MI, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda,frequency=5, kl=1)
        vae_MI_file_path = '%s/%s_%s_MIScale%s_sample%s_VaeMI.pk1'%(data_save_path, dataset_name, nuisance_variable, MIScale, taskid)

        if os.path.isfile(vae_MI_file_path):
            trainer_vae_MI.model.load_state_dict(torch.load(vae_MI_file_path))
            trainer_vae_MI.model.eval()
        else:
            trainer_vae_MI.train(n_epochs=n_epochs, lr=lr)
            torch.save(trainer_vae_MI.model.state_dict(), vae_MI_file_path)
            ll_train_set = trainer_vae_MI.history["ll_train_set"]
            ll_test_set = trainer_vae_MI.history["ll_test_set"]
            x = np.linspace(0, 500, (len(ll_train_set)))

            fig = plt.figure(figsize=(14, 7))
            plt.plot(x, ll_train_set)
            plt.plot(x, ll_test_set)
            if dataset_name=='muris_tabula':
               plt.ylim(13000, 25000)
            elif dataset_name=='pbmc':
               plt.ylim(1150, 1600)
            plt.title("Blue for training error and orange for testing error")

            fig1_path = '%s/training_testing_error_SCVI+MI_%s_%s__MIScale%s_sample%s.png'%(result_save_path, dataset_name,nuisance_variable, MIScale, taskid )
            fig.savefig(fig1_path)
            plt.close(fig)

        trainer_vae_MI.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/trainset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s'%(result_save_path, dataset_name,nuisance_variable, MIScale, taskid))
        trainer_vae_MI.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='%s/testset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s'%(result_save_path, dataset_name,nuisance_variable, MIScale, taskid))

        asw, nmi, ari, uca = trainer_vae_MI.train_set.clustering_scores()
        be = trainer_vae_MI.train_set.entropy_batch_mixing()

        latent, batch_indices, labels = trainer_vae_MI.train_set.get_latent(sample=False)
        x_ = trainer_vae_MI.train_set.gene_dataset._X.toarray()
        if trainer_vae_MI.model.log_variational:
            x_ = np.log(1 + x_)
        qz_m, qz_v, z = trainer_vae_MI.model.z_encoder(x_, None)
        z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v, trainer_vae_MI.train_set.gene_dataset.batch_indices,trainer_vae_MI.model.nsamples_z)
        batch0_indices = np.array([[0] * (z_batch0.shape[0])])
        batch1_indices = np.array([[1] * (z_batch1.shape[0])])
        if MI_estimator == 'Mine_Net4': ##this if part should be modified
            latent_tensor = Variable(latent.type(torch.FloatTensor), requires_grad=True)
            latent_shuffle = np.random.permutation(latent)
            latent_shuffle = Variable(torch.from_numpy(latent_shuffle).type(torch.FloatTensor), requires_grad=True)
            batch_tensor = Variable(batch_indices.type(torch.FloatTensor), requires_grad=True)
            latent_batch = torch.cat([latent_tensor, batch_tensor], dim=1)
            pred_xz, pred_x_z = trainer_vae_MI.model.minenet(xy=latent_batch, x_shuffle=latent_shuffle, x_n_dim=latent.shape[-1])
            predicted_mutual_info = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
        elif MI_estimator == 'NN':
            batch_array = np.append(batch0_indices, batch1_indices, axis=1)
            z_array = np.append(z_batch0, z_batch1, axis=0).transpose()
            predicted_mutual_info = discrete_continuous_info(d=batch_array, c=z_array)
        elif MI_estimator == 'aggregated_posterior':
            z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
            z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
            pred_xz, pred_x_z = trainer_vae_MI.model.minenet(x=z_batch0_tensor, y=z_batch1_tensor)
            predicted_mutual_info = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))

        label = '%s_%s_MIScale%s_sample%s_VaeMI_trainset'%(dataset_name, nuisance_variable, MIScale, taskid)

        intermediate_dataframe1 = pd.DataFrame.from_dict({'Label':[label],'asw':[asw],'nmi':[nmi],'ari':[ari],'uca':[uca],'be':[be],'MILoss':[predicted_mutual_info]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

        asw, nmi, ari, uca = trainer_vae_MI.test_set.clustering_scores()
        be = trainer_vae_MI.test_set.entropy_batch_mixing()

        latent, batch_indices, labels = trainer_vae_MI.test_set.get_latent(sample=False)
        x_ = trainer_vae_MI.test_set.gene_dataset._X.toarray()
        x_ = Variable(torch.from_numpy(x_).type(torch.FloatTensor), requires_grad=True)
        if trainer_vae_MI.model.log_variational:
            x_ = torch.log(1 + x_)
        qz_m, qz_v, z = trainer_vae_MI.model.z_encoder(x_, None)
        batch_indices_tensor = Variable(torch.from_numpy(trainer_vae_MI.test_set.gene_dataset.batch_indices).type(torch.FloatTensor), requires_grad=True)
        z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v,batch_indices_tensor,trainer_vae_MI.model.nsamples_z)
        batch0_indices = np.array([[0] * (z_batch0.shape[0])])
        batch1_indices = np.array([[1] * (z_batch1.shape[0])])
        if MI_estimator == 'Mine_Net4': ##this if part should be modified
            latent_tensor = Variable(latent.type(torch.FloatTensor), requires_grad=True)
            latent_shuffle = np.random.permutation(latent)
            latent_shuffle = Variable(torch.from_numpy(latent_shuffle).type(torch.FloatTensor), requires_grad=True)
            batch_tensor = Variable(batch_indices.type(torch.FloatTensor), requires_grad=True)
            latent_batch = torch.cat([latent_tensor, batch_tensor], dim=1)
            pred_xz, pred_x_z = trainer_vae_MI.model.minenet(xy=latent_batch, x_shuffle=latent_shuffle, x_n_dim=latent.shape[-1])
            predicted_mutual_info = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
        elif MI_estimator == 'NN':
            batch_array = np.append(batch0_indices,batch1_indices, axis=1)
            z_array = np.append(z_batch0, z_batch1, axis=0).transpose()
            predicted_mutual_info = discrete_continuous_info(d=batch_array, c=z_array)
        elif MI_estimator == 'aggregated_posterior':
            z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
            z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
            pred_xz, pred_x_z = trainer_vae_MI.model.minenet(x=z_batch0_tensor, y=z_batch1_tensor)
            predicted_mutual_info = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))

        label = '%s_%s_MIScale%s_sample%s_VaeMI_testset'%(dataset_name, nuisance_variable, MIScale, taskid)
        intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be],'MILoss':[predicted_mutual_info]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)
        clustering_metric.to_csv('%s/%s_%s_sample%s_ClusterMetric.csv' % (result_save_path, dataset_name, nuisance_variable, taskid), index=None, header=True)

    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels, n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder = n_layers_encoder, n_layers_decoder=n_layers_decoder, dropout_rate = dropout_rate, reconstruction_loss=reconstruction_loss)
    trainer_vae = UnsupervisedTrainer(vae, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5)
    vae_file_path = '%s/%s_%s_sample%s_Vae.pk1'%(data_save_path,dataset_name, nuisance_variable, taskid)

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
    if dataset_name=='muris_tabula':
       plt.ylim(13000, 25000)
    elif dataset_name=='pbmc':
       plt.ylim(1150, 1600)
    plt.title("Blue for training error and orange for testing error")

    fig2_path = '%s/training_testing_error_SCVI_%s_%s_sample%s.png'%(result_save_path, dataset_name, nuisance_variable, taskid)
    fig.savefig(fig2_path)
    plt.close(fig)

    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='%s/trainset_tsne_SCVI_%s_%s_sample%s'%(result_save_path, dataset_name,nuisance_variable, taskid))
    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='%s/testset_tsne_SCVI_%s_%s_sample%s'%(result_save_path, dataset_name,nuisance_variable, taskid))

    #   clustering_scores() -- these metrics measure clustering performance
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    #   entropy_batch_mixing() -- this metric measures batch effect
    #   entropy batch mixing (be, higher is better meaning less batch effect)

    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()

    x_ = trainer_vae.train_set.gene_dataset._X.toarray()
    if trainer_vae.model.log_variational:
        x_ = np.log(1 + x_)
    qz_m, qz_v, z = trainer_vae.model.z_encoder(x_, None)
    z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v,trainer_vae.train_set.gene_dataset.batch_indices,trainer_vae.model.nsamples_z)
    batch0_indices = np.array([[0] * (z_batch0.shape[0])])
    batch1_indices = np.array([[1] * (z_batch1.shape[0])])
    if MI_estimator == 'Mine_Net4':
        predicted_mutual_info = None
    elif MI_estimator == 'NN':
        batch_array = np.append(batch0_indices, batch1_indices, axis=1)
        z_array = np.append(z_batch0, z_batch1, axis=0).transpose()
        predicted_mutual_info = discrete_continuous_info(d=batch_array, c=z_array)
    elif MI_estimator == 'aggregated_posterior':
        predicted_mutual_info = None

    label = '%s_%s_sample%s_Vae_trainset' % (dataset_name, nuisance_variable, taskid)
    intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be],'MILoss':[predicted_mutual_info]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

    asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
    be = trainer_vae.test_set.entropy_batch_mixing()

    x_ = trainer_vae.test_set.gene_dataset._X.toarray()
    if trainer_vae.model.log_variational:
        x_ = np.log(1 + x_)
    qz_m, qz_v, z = trainer_vae.model.z_encoder(x_, None)
    z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v, trainer_vae.train_set.gene_dataset.batch_indices,trainer_vae.model.nsamples_z)
    batch0_indices = np.array([[0] * (z_batch0.shape[0])])
    batch1_indices = np.array([[1] * (z_batch1.shape[0])])
    if MI_estimator == 'Mine_Net4':
        predicted_mutual_info = None
    elif MI_estimator == 'NN':
        batch_array = np.append(batch0_indices, batch1_indices, axis=1)
        z_array = np.append(z_batch0, z_batch1, axis=0).transpose()
        predicted_mutual_info = discrete_continuous_info(d=batch_array, c=z_array)
    elif MI_estimator == 'aggregated_posterior':
        predicted_mutual_info = None

    label = '%s_%s_sample%s_Vae_testset' % (dataset_name, nuisance_variable,taskid)
    intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be],'MILoss':[predicted_mutual_info]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)

    clustering_metric.to_csv('%s/%s_%s_sample%s_ClusterMetric.csv'%(result_save_path, dataset_name, nuisance_variable, taskid), index=None, header=True)

# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
