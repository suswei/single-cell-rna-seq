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

def main(taskid, dataset_name, nuisance_variable, config_id):
    # taskid is just any integer from 0 to 99
    # dataset_name could be 'muris_tabula', 'pbmc'
    # nuisance_variable could be 'batch'
    # MI_estimator could be 'Mine_Net4', 'NN' (NN stands for nearest neighbor), 'aggregated_posterior'

    if not os.path.exists('./data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('./data/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))
    if not os.path.exists('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name)):
        os.makedirs('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config' % (dataset_name))

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
            'MIScale': [0.2, 0.5, 0.8], #0.2, 0.5, 0.8
            'train_size': [0.8],
            'lr': [1e-3, 5e-3, 1e-4], #1e-3, 5e-3, 1e-4
            'adv_lr': [5e-4, 1e-8], #5e-4, 1e-8
            'n_epochs': [2500], #2500
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'adv_epochs': [100], #100
            'change_adv_epochs': [5], #5
            'activation_fun': ['ELU','Leaky_ReLU'],  # activation_fun could be 'ReLU', 'ELU', 'Leaky_ReLU' , 'Leaky_ReLU'
            'unbiased_loss': [True],  # unbiased_loss: True or False. Whether to use unbiased loss or not
            'initial': ['xavier_normal1'], # initial: could be 'None', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform','kaiming_normal', 'orthogonal', 'sparse' ('orthogonal', 'sparse' are not proper in our case)
            'adv_model' : ['MI','Classifier'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2]
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

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100), dtype=np.uint32)

    taskid = int(taskid)
    desired_seed = int(desired_seeds[0, taskid])

    n_samples_tsne = 1000
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be', 'MILoss'])

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
    MIScale = value[8]
    train_size = value[9]
    lr = value[10]  # 0.0005
    adv_lr = value[11]
    n_epochs = value[12]  # 500
    nsamples_z = value[13]
    adv = value[14]
    Adv_Net_architecture = value[15]
    adv_epochs = value[16]
    change_adv_epochs = value[17]
    activation_fun = value[18]
    unbiased_loss = value[19]
    initial = value[20]
    adv_model = value[21]
    optimiser = value[22]
    adv_drop_out = value[23]

    if not os.path.exists('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/config%s' % (dataset_name, config_id)):
        os.makedirs('./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/config%s' % (dataset_name, config_id))

    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder,
              n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,reconstruction_loss=reconstruction_loss, nsamples_z=nsamples_z, adv=False,
              save_path='None')
    trainer_vae = UnsupervisedTrainer(vae, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)
    vae_ELBO_list = trainer_vae.train(n_epochs=n_epochs, lr=lr)

    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                    n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder,
                    n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate, reconstruction_loss=reconstruction_loss,
                    MI_estimator=adv_model, MIScale=MIScale, nsamples_z=nsamples_z, adv=adv,
                    Adv_MineNet4_architecture=Adv_Net_architecture, save_path=result_save_path+'/config%s/'%(config_id),
                    mini_ELBO=min(vae_ELBO_list), max_ELBO=np.percentile(vae_ELBO_list, 90))
    trainer_vae_MI = UnsupervisedTrainer(vae_MI, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)
    trainer_vae_MI_adv = UnsupervisedTrainer(vae_MI, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1, batch_size=256)

    if adv == True:
        if adv_model == 'MI':
            advnet = MINE_Net4_3(input_dim=vae_MI.n_latent + 1, n_latents=Adv_Net_architecture,
                                  activation_fun=activation_fun, unbiased_loss=unbiased_loss, initial=initial,
                                  save_path='./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/config%s/' % (dataset_name, config_id),
                                  data_loader=trainer_vae_MI_adv, drop_out = adv_drop_out, net_name = adv_model)
        elif adv_model == 'Classifier':
            advnet = Classifier_Net(input_dim=vae_MI.n_latent + 1, n_latents=Adv_Net_architecture, activation_fun=activation_fun, initial=initial,
                                  save_path='./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/config%s/' % (dataset_name, config_id),
                                  data_loader=trainer_vae_MI_adv, drop_out = adv_drop_out, net_name = adv_model)
        adv_optimizer = torch.optim.Adam(advnet.parameters(), lr=adv_lr)
        trainer_vae_MI.adv_model = advnet
        trainer_vae_MI.adv_optimizer = adv_optimizer
        trainer_vae_MI.adv_epochs = adv_epochs
        trainer_vae_MI.change_adv_epochs = change_adv_epochs

    vae_MI_file_path = '%s/%s_%s_config%s_VaeMI.pk1' % (data_save_path, dataset_name, nuisance_variable, config_id)
    adv_MI_file_path = '%s/%s_%s_config%s_advMI.pk1' % (data_save_path, dataset_name, nuisance_variable, config_id)

    if os.path.isfile(vae_MI_file_path) and os.path.isfile(adv_MI_file_path):
        trainer_vae_MI.model.load_state_dict(torch.load(vae_MI_file_path))
        trainer_vae_MI.model.eval()
        trainer_vae_MI.adv_model.load_state_dict(torch.load(adv_MI_file_path))
        trainer_vae_MI.adv_model.eval()
    else:
        ELBO_list, penalty_loss_list, activation_mean, activation_var = trainer_vae_MI.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer_vae_MI.model.state_dict(), vae_MI_file_path)
        torch.save(trainer_vae_MI.adv_model.state_dict(), adv_MI_file_path)
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

        fig1_path = '%s/config%s/training_testing_error_SCVI+MI_%s_%s_config%s.png'%(result_save_path,config_id, dataset_name,nuisance_variable, config_id)
        fig.savefig(fig1_path)
        plt.close(fig)

    layers = {'layers': ['layer2'] + ['layer%s'%((k+1)*10-1) for k in range(int(trainer_vae_MI.adv_model.n_hidden_layers / 10))]}
    activation_mean_pd = pd.concat([pd.DataFrame.from_dict(layers),pd.DataFrame(data=activation_mean, columns=['epoch%s'%(i*10) for i in range(int((n_epochs-1)/10)+1)])],axis=1)
    activation_var_pd = pd.concat([pd.DataFrame.from_dict(layers),pd.DataFrame(data=activation_var, columns=['epoch%s'%(i*10) for i in range(int((n_epochs-1)/10)+1)])],axis=1)
    activation_mean_pd.to_csv('%s/config%s/%s_%s_config%s_activationmean.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id),index=None, header=True)
    activation_var_pd.to_csv('%s/config%s/%s_%s_config%s_activationvar.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id),index=None, header=True)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(ELBO_list))], [np.mean(i) for i in ELBO_list])
    plt.ylim(12000, 60000)
    plt.title("reconst_loss_%s_%s_config%s"%(dataset_name, nuisance_variable, config_id))
    fig1_path = '%s/config%s/reconst_loss_%s_%s_config%s.png' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id)
    fig.savefig(fig1_path)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(ELBO_list))], penalty_loss_list)
    plt.title("MI_loss_%s_%s_config%s" % (dataset_name, nuisance_variable, config_id))
    fig1_path = '%s/config%s/MI_%s_%s_config%s.png' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id)
    fig.savefig(fig1_path)
    plt.close(fig)

    trainer_vae_MI.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/config%s/trainset_tsne_SCVI+MI_%s_%s_config%s' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id))
    trainer_vae_MI.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/config%s/testset_tsne_SCVI+MI_%s_%s_config%s' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id))

    asw, nmi, ari, uca = trainer_vae_MI.train_set.clustering_scores()
    be = trainer_vae_MI.train_set.entropy_batch_mixing()

    # latent, batch_indices, labels = trainer_vae_MI.train_set.get_latent(sample=False)
    x_ = trainer_vae_MI.train_set.gene_dataset._X.toarray()
    if trainer_vae_MI.model.log_variational:
        x_ = np.log(1 + x_)
    x_ = Variable(torch.from_numpy(x_).type(torch.FloatTensor), requires_grad=True)
    qz_m, qz_v, z = trainer_vae_MI.model.z_encoder(x_, None)
    ql_m, ql_v, library = trainer_vae_MI.model.l_encoder(x_)
    batch_indices_tensor = Variable(torch.from_numpy(trainer_vae_MI.train_set.gene_dataset.batch_indices).type(torch.FloatTensor),requires_grad=True)

    batch_index_adv_list = np.ndarray.tolist(batch_indices_tensor.detach().numpy())
    z_batch0_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
    z_batch1_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
    l_batch0_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
    l_batch1_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
    l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
    l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)
    pred_xz = trainer_vae_MI.adv_model(input=l_z_batch0_tensor)
    pred_x_z = trainer_vae_MI.adv_model(input=l_z_batch1_tensor)
    predicted_mutual_info = (torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))).detach().cpu().numpy()

    label = '%s_%s_config%s_VaeMI_trainset' % (dataset_name, nuisance_variable, config_id)

    intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be], 'MILoss': [predicted_mutual_info]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

    asw, nmi, ari, uca = trainer_vae_MI.test_set.clustering_scores()
    be = trainer_vae_MI.test_set.entropy_batch_mixing()

    # latent, batch_indices, labels = trainer_vae_MI.test_set.get_latent(sample=False)
    x_ = trainer_vae_MI.test_set.gene_dataset._X.toarray()
    if trainer_vae_MI.model.log_variational:
        x_ = np.log(1 + x_)
    x_ = Variable(torch.from_numpy(x_).type(torch.FloatTensor), requires_grad=True)
    qz_m, qz_v, z = trainer_vae_MI.model.z_encoder(x_, None)
    batch_indices_tensor = Variable(
        torch.from_numpy(trainer_vae_MI.test_set.gene_dataset.batch_indices).type(torch.FloatTensor),
        requires_grad=True)

    batch_index_adv_list = np.ndarray.tolist(batch_indices_tensor.detach().numpy())
    z_batch0_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
    z_batch1_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
    l_batch0_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
    l_batch1_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
    l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
    l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)
    pred_xz = trainer_vae_MI.adv_model(input=l_z_batch0_tensor)
    pred_x_z = trainer_vae_MI.adv_model(input=l_z_batch1_tensor)
    predicted_mutual_info = (torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))).detach().cpu().numpy()

    label = '%s_%s_config%s_VaeMI_testset' % (dataset_name, nuisance_variable, config_id)
    intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be],'MILoss': [predicted_mutual_info]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)
    clustering_metric.to_csv('%s/config%s/%s_%s_config%s_ClusterMetric.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id), index=None, header=True)

# Run the actual program
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
