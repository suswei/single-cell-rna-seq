# Run 100 Monte Carlo Samples for each dataset, as scVI is variable for each run.
# For mouse marrow dataset, the hyperparameter tried n_layers for scvi=2, n_hidden=128, n_latent=10, reconstruction_loss=zinb, dropout_rate=0.1, lr=0.001, n_epochs=250, training_size-0.8
# For Pbmc dataset, the hyperparameter tried n_layers_for_scvi=1, n_latent=256, n_latent=14, dropout_rate=0.5, lr=0.01, n_epochs=170, training_size=0.8.


import os
import sys
import argparse
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scvi.dataset import *
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.models.modules import MINE_Net3, Classifier_Net
from scvi.inference import UnsupervisedTrainer
import torch
from torch.autograd import Variable
from tqdm import tqdm

def main( ):

    parser = argparse.ArgumentParser(description='Pareto_Front')

    parser.add_argument('--taskid', type=int, default=1000 + randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--dataset_name', type=str, default='muris_tabula',
                        help='the name of the dataset')

    parser.add_argument('--nuisance_variable', type=str, default='batch',
                        help='the name of nuisance variable')


    #for scVI
    parser.add_argument('--n_layers_encoder', type=int, default=2,
                        help='number of hidden layers for encoder in scVI')

    parser.add_argument('--n_layers_decoder', type=int, default=2,
                        help='number of hidden layers for decoder in scVI')

    parser.add_argument('--n_hidden', type=int, default=128,
                        help='number of hidden nodes for each hidden layer in both encoder and decoder in scVI')

    parser.add_argument('--n_latent', type=int, default=10,
                        help='dimension for latent vector z')

    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout rate for encoder in scVI')

    parser.add_argument('--reconstruction_loss', type=str, default='zinb',
                        help='the generative model used to estimate loss')

    parser.add_argument('--use_batches', type=bool, default=True,
                        help='whether to use batches or not in scVI')

    parser.add_argument('--train_size', type=float, default=0.8,
                        help='the ratio to split training and testing data set')

    parser.add_argument('--std', type=bool, default=True,
                        help='whether to standardize loss from scVI and MINE')

    parser.add_argument('--max_neg_ELBO', type=float, default=17000,
                        help='the maximum value used to standardize loss of scVI')

    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for scVI')

    parser.add_argument('--pre_n_epochs', type=int, default=100,
                        help='number of epochs to pretrain scVI')

    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs to train scVI and MINE')

    parser.add_argument('--nsamples_z', type=int, default=200,
                        help='number of z sampled from aggregated posterior for nearest neighbor method')


    #for MINE
    parser.add_argument('--conf_estimator', type=str, default='MINE',
                       help='the method used to estimate MI or KL_divergence')

    parser.add_argument('--adv_n_latents', type=int, default=128,
                       help='the number of hidden nodes in each hidden layer for MINE')

    parser.add_argument('--adv_n_layers', type=int, default=10,
                        help='the number of hidden layers for MINE')

    parser.add_argument('--adv_pre_epochs', type=int, default=100,
                        help='number of epochs to pretrain MINE')

    parser.add_argument('--adv_epochs', type=int, default=2,
                        help='number of epochs to train MINE for each minibatch when scVI is trained')

    parser.add_argument('--activation_fun', type=str, default='Leaky_ReLU',
                        help='the activation function used for MINE')

    parser.add_argument('--unbiased_loss', type=bool, default=True,
                        help='whether to use unbiased loss or not in MINE')

    parser.add_argument('--batchsize', type=int, default=128,
                        help='batch size for MINE training')

    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs in MINE training')

    parser.add_argument('--adv_lr', type=float, default=5e-4,
                        help='learning rate in MINE training')

    parser.add_argument('--lambda', type=float, default=0,
                        help='the scale for the standardized loss of MINE')


    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before logging training status')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))

    if args.activation_fun == 'Leaky_ReLU':
        args.w_initial = 'kaiming_normal'
    elif args.activation_fun in ['ReLU', 'ELU']:
        args.w_initial = 'normal'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_save_path = './data/pareto_front_scVI_MINE/%s' % (args.dataset_name)
    if not os.path.exists('./data/pareto_front_scVI_MINE/%s' % (args.dataset_name)):
        os.makedirs('./data/pareto_front_scVI_MINE/%s' % (args.dataset_name))

    if args.dataset_name == 'muris_tabula':
        dataset1 = TabulaMuris('facs', save_path=data_save_path)
        dataset2 = TabulaMuris('droplet', save_path=data_save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    elif args.dataset_name == 'pbmc':
        gene_dataset = PbmcDataset(save_path=data_save_path)
    elif args.dataset_name == 'retina':
        gene_dataset = RetinaDataset(save_path=data_save_path)

    #get the seed to split training and testing dataset
    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100), dtype=np.uint32)
    seed_idx =  args.taskid - int(args.taskid/100)*100
    desired_seed = int(desired_seeds[0, seed_idx])

    result_save_path = './result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder_type, args.taskid)
    if not os.path.exists('./result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder_type, args.taskid)):
        os.makedirs('./result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder_type, args.taskid))

    #vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder,
    #          n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,reconstruction_loss=reconstruction_loss, nsamples_z=nsamples_z, adv=False,
    #         save_path='None')
    #trainer_vae = UnsupervisedTrainer(vae, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)
    #vae_ELBO_list = trainer_vae.train(n_epochs=n_epochs, lr=lr)
    '''
    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                    n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder, n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,
                    reconstruction_loss=reconstruction_loss, MI_estimator=adv_model, adv=False, save_path='None',
                    std=False, mini_ELBO=12000, max_ELBO=max_reconst)  # mini_ELBO=15000, max_ELBO=20000
    trainer_vae_MI = UnsupervisedTrainer(vae_MI, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)

    vae_MI_file_path = '%s/%s_%s_taskid%s_VaeMI.pth' % (data_save_path, dataset_name, nuisance_variable, taskid)
    # Pretrain trainer_vae_MI.vae_MI when adv=False
    trainer_vae_MI.train(n_epochs=pre_n_epochs, lr=1e-3)
    torch.save(trainer_vae_MI.model.state_dict(), vae_MI_file_path)
    '''
    vae_MI_file_path = '%s/%s_%s_taskid%s_VaeMI.pth' % (data_save_path, dataset_name, nuisance_variable, taskid)


    MIScale_index = jobid%10
    MIScale = MIScales[MIScale_index]
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be', 'std_penalty', 'std_ELBO', 'penalty_fully'])

    vae_MI2 = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                    n_hidden=n_hidden, n_latent=n_latent, n_layers_encoder=n_layers_encoder, n_layers_decoder=n_layers_decoder, dropout_rate=dropout_rate,
                    reconstruction_loss=reconstruction_loss, MI_estimator=adv_model, MIScale=MIScale, nsamples_z=nsamples_z, adv=adv,
                    Adv_MineNet4_architecture=Adv_Net_architecture, save_path=result_save_path + '/rep%s/' % (taskid),
                    std=std, mini_ELBO=12000, max_ELBO=max_reconst, MIScale_index=MIScale_index)  # mini_ELBO=15000, max_ELBO=20000
    trainer_vae_MI2 = UnsupervisedTrainer(vae_MI2, gene_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5, kl=1)
    trainer_vae_MI2.model.load_state_dict(torch.load(vae_MI_file_path))
    #trainer_vae_MI2.model.adv = adv
    #trainer_vae_MI2.model.std = std

    trainer_vae_MI2_adv = UnsupervisedTrainer(vae_MI2, gene_dataset, train_size=train_size, seed=desired_seed,use_cuda=use_cuda, frequency=5, kl=1, batch_size=256)

    if adv == True:
        if adv_model == 'MI':
            advnet = MINE_Net4_3(input_dim=vae_MI2.n_latent + 1, n_latents=Adv_Net_architecture,
                                  activation_fun=activation_fun, unbiased_loss=unbiased_loss, initial=initial,
                                  save_path='./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/rep%s/' % (dataset_name, taskid),
                                  data_loader=trainer_vae_MI2_adv, drop_out = adv_drop_out, net_name = adv_model, min=-0.075, max=0.05)
        elif adv_model == 'Classifier':
            advnet = Classifier_Net(input_dim=vae_MI2.n_latent + 1, n_latents=Adv_Net_architecture, activation_fun=activation_fun, initial=initial,
                                  save_path='./result/tune_hyperparameter_for_SCVI_MI/%s/choose_config/rep%s/' % (dataset_name, taskid),
                                  data_loader=trainer_vae_MI2_adv, drop_out = adv_drop_out, net_name = adv_model, min=0.2, max=6)
        trainer_vae_MI2.adv_model = advnet
        trainer_vae_MI2.adv_criterion = torch.nn.BCELoss(reduction='mean')
        trainer_vae_MI2.adv_optimizer = torch.optim.Adam(advnet.parameters(), lr=adv_lr)
        trainer_vae_MI2.adv_epochs = pre_adv_epochs
        trainer_vae_MI2.change_adv_epochs = adv_epochs

    '''
    adv_MI_file_path = '%s/%s_%s_config%s_advMI.pk1' % (data_save_path, dataset_name, nuisance_variable, config_id)
    if os.path.isfile(vae_MI_file_path) and os.path.isfile(adv_MI_file_path):
        trainer_vae_MI2.model.load_state_dict(torch.load(vae_MI_file_path))
        trainer_vae_MI2.model.eval()
        trainer_vae_MI2.adv_model.load_state_dict(torch.load(adv_MI_file_path))
        trainer_vae_MI2.adv_model.eval()
    else:
    '''
    ELBO_list, std_ELBO_list, penalty_list, std_penalty_list = trainer_vae_MI2.train(n_epochs=n_epochs, lr=lr)
    #torch.save(trainer_vae_MI2.adv_model.state_dict(), adv_MI_file_path)
    ll_train_set = trainer_vae_MI2.history["ll_train_set"]
    ll_test_set = trainer_vae_MI2.history["ll_test_set"]
    x = np.linspace(0, 500, (len(ll_train_set)))

    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, ll_train_set)
    plt.plot(x, ll_test_set)
    if dataset_name=='muris_tabula':
       plt.ylim(13000, 25000)
    elif dataset_name=='pbmc':
       plt.ylim(1150, 1600)
    plt.title("Blue for training error and orange for testing error")

    fig1_path = '%s/rep%s/training_testing_error_SCVI+MI_%s_%s_MIScale%s.png'%(result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index)
    fig.savefig(fig1_path)
    plt.close(fig)
    '''
    layers = {'layers': ['layer2'] + ['layer%s'%((k+1)*10-1) for k in range(int(trainer_vae_MI2.adv_model.n_hidden_layers / 10))]}
    activation_mean_pd = pd.concat([pd.DataFrame.from_dict(layers),pd.DataFrame(data=activation_mean, columns=['epoch%s'%(i*10) for i in range(int((n_epochs-1)/10)+1)])],axis=1)
    activation_var_pd = pd.concat([pd.DataFrame.from_dict(layers),pd.DataFrame(data=activation_var, columns=['epoch%s'%(i*10) for i in range(int((n_epochs-1)/10)+1)])],axis=1)
    activation_mean_pd.to_csv('%s/config%s/%s_%s_config%s_activationmean.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id),index=None, header=True)
    activation_var_pd.to_csv('%s/config%s/%s_%s_config%s_activationvar.csv' % (result_save_path, config_id, dataset_name, nuisance_variable, config_id),index=None, header=True)
    '''
    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(ELBO_list))], [np.mean(i) for i in ELBO_list])
    plt.ylim(12000, 60000)
    plt.title("reconst_loss_%s_%s_MIScale%s"%(dataset_name, nuisance_variable, MIScale_index))
    fig1_path = '%s/rep%s/reconst_loss_%s_%s_MIScale%s.png' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index)
    fig.savefig(fig1_path)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(std_ELBO_list))], [np.mean(i) for i in std_ELBO_list])
    plt.title("std_reconst_loss_%s_%s_MIScale%s" % (dataset_name, nuisance_variable, MIScale_index))
    fig1_path = '%s/rep%s/std_reconst_loss_%s_%s_MIScale%s.png' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index)
    fig.savefig(fig1_path)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(penalty_list))], penalty_list)
    plt.title("penalty_%s_%s_MIScale%s" % (dataset_name, nuisance_variable, MIScale_index))
    fig1_path = '%s/rep%s/penalty_%s_%s_MIScale%s.png' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index)
    fig.savefig(fig1_path)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 7))
    plt.plot([i for i in range(len(std_penalty_list))], std_penalty_list)
    plt.title("std_penalty_%s_%s_MIScale%s" % (dataset_name, nuisance_variable, MIScale_index))
    fig1_path = '%s/rep%s/std_penalty_%s_%s_MIScale%s.png' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index)
    fig.savefig(fig1_path)
    plt.close(fig)

    minibatch_info = pd.DataFrame.from_dict({'minibatch_ELBO': ELBO_list, 'minibatch_penalty': penalty_list})
    minibatch_info.to_csv('%s/rep%s/%s_%s_MIScale%s_minibatch_info.csv' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index), index=None, header=True)

    n_samples_tsne = 1000
    trainer_vae_MI2.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/rep%s/trainset_tsne_SCVI+MI_%s_%s_MIScale%s' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index))
    trainer_vae_MI2.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='%s/rep%s/testset_tsne_SCVI+MI_%s_%s_MIScale%s' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index))

    asw, nmi, ari, uca = trainer_vae_MI2.train_set.clustering_scores()
    be = trainer_vae_MI2.train_set.entropy_batch_mixing()

    # latent, batch_indices, labels = trainer_vae_MI2.train_set.get_latent(sample=False)
    '''
     x_ = trainer_vae_MI2.test_set.gene_dataset._X.toarray()
     if trainer_vae_MI2.model.log_variational:
         x_ = np.log(1 + x_)
     x_ = Variable(torch.from_numpy(x_).type(torch.FloatTensor), requires_grad=True)
     qz_m, qz_v, z = trainer_vae_MI2.model.z_encoder(x_, None)
     batch_indices_tensor = Variable(torch.from_numpy(trainer_vae_MI2.test_set.gene_dataset.batch_indices).type(torch.FloatTensor),requires_grad=True)
     

    ##fully train adv_minenet
    advnet2 = MINE_Net4_3(input_dim=vae_MI.n_latent + 1, n_latents=Adv_Net_architecture,
                         activation_fun=activation_fun, unbiased_loss=unbiased_loss, initial=initial,
                         save_path='None',data_loader=trainer_vae_MI_adv, drop_out=adv_drop_out, net_name=adv_model, min=-0.3, max=0.3)
    full_adv_optimizer = torch.optim.Adam(advnet2.parameters(), lr=adv_lr)
    full_adv_epochs = 500
    for full_adv_epoch in tqdm(range(full_adv_epochs)):
        advnet2.train()
        for tensor in advnet2.data_loader.data_loaders_loop():
            sample_batch_adv, local_l_mean_adv, local_l_var_adv, batch_index_adv, _ = tensor
            if trainer_vae_MI2.model.log_variational:
                x_ = torch.log(1 + x_)
            qz_m, qz_v, z = trainer_vae_MI2.model.z_encoder(x_, None)
            ql_m, ql_v, library = trainer_vae_MI2.model.l_encoder(x_)

            batch_index_adv_list = np.ndarray.tolist(batch_index_adv.detach().numpy())
            z_batch0_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
            z_batch1_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
            l_batch0_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]],:]
            l_batch1_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]],:]
            l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
            l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)

            if (l_z_batch0_tensor.shape[0] == 0) or (l_z_batch1_tensor.shape[0] == 0):
                continue

            pred_xz = advnet2(input=l_z_batch0_tensor)
            pred_x_z = advnet2(input=l_z_batch1_tensor)

            if advnet2.unbiased_loss:
                t = pred_xz
                et = torch.exp(pred_x_z)
                if advnet2.ma_et is None:
                    advnet2.ma_et = torch.mean(et).detach().item()
                advnet2.ma_et += advnet2.ma_rate * (torch.mean(et).detach().item() - advnet2.ma_et)
                # unbiasing use moving average
                loss_adv2 = -(torch.mean(t) - (torch.log(torch.mean(et)) * torch.mean(et).detach() / advnet2.ma_et))
            else:
                loss_adv = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
                loss_adv2 = -loss_adv  # maximizing loss_adv equals minimizing -loss_adv

            full_adv_optimizer.zero_grad()
            loss_adv2.backward()
            full_adv_optimizer.step()
    advnet2.eval()
    '''
    z_tensor_train = Variable(torch.from_numpy(np.empty([0, 10], dtype=float)).type(torch.FloatTensor),requires_grad=True)
    batch_indices_list_train = []
    library_tensor_train = Variable(torch.from_numpy(np.empty([0, 1], dtype=float)).type(torch.FloatTensor),requires_grad=True)
    for tensors in trainer_vae_MI2.train_set:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        if trainer_vae_MI2.model.log_variational:
            x_ = torch.log(1 + sample_batch)
        z_tensor_train = torch.cat((z_tensor_train, trainer_vae_MI2.model.z_encoder(x_)[0].cpu()), dim=0)
        batch_indices_list_train += [batch_index.cpu()]
        library_tensor_train = torch.cat((library_tensor_train, trainer_vae_MI2.model.l_encoder(x_)[0].cpu()), dim=0)
    batch_indices_array_train = np.empty([0, 1], dtype=int)
    for i in range(len(batch_indices_list_train)):
        batch_indices_array_train = np.concatenate((batch_indices_array_train, batch_indices_list_train[i].detach().numpy()), axis=0)

    if adv_model == 'MI':
        '''
        batch_index_adv_list = np.ndarray.tolist(batch_indices_array_train)
        z_batch0_tensor_train = z_tensor_train[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]],:]
        z_batch1_tensor_train = z_tensor_train[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]],:]
        l_batch0_tensor_train = library_tensor_train[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
        l_batch1_tensor_train = library_tensor_train[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
        l_z_batch0_tensor_train = torch.cat((l_batch0_tensor_train, z_batch0_tensor_train), dim=1)
        l_z_batch1_tensor_train = torch.cat((l_batch1_tensor_train, z_batch1_tensor_train), dim=1)
        pred_xz_train = trainer_vae_MI2.adv_model(input=l_z_batch0_tensor_train)
        pred_x_z_train = trainer_vae_MI2.adv_model(input=l_z_batch1_tensor_train)
        predicted_mutual_info = (torch.mean(pred_xz_train) - torch.log(torch.mean(torch.exp(pred_x_z_train)))).detach().cpu().numpy()
        std_predicted_mutual_info = (predicted_mutual_info - (-0.02))/(0.3-(-0.02))
        '''
        l_z_joint_train = torch.cat((library_tensor_train, z_tensor_train), dim=1)
        z_shuffle_train = np.random.permutation(z_tensor_train.detach().numpy())
        z_shuffle_train = Variable(torch.from_numpy(z_shuffle_train).type(torch.FloatTensor), requires_grad=True)
        l_z_indept_train = torch.cat((library_tensor_train, z_shuffle_train), dim=1)
        pred_xz_train = trainer_vae_MI2.adv_model(input=l_z_joint_train)
        pred_x_z_train = trainer_vae_MI2.adv_model(input=l_z_indept_train)
        predicted_mutual_info = (torch.mean(pred_xz_train) - (torch.log(torch.mean(torch.exp(pred_x_z_train))) * torch.mean(torch.exp(pred_x_z_train)).detach() / trainer_vae_MI2.adv_model.ma_et)).detach().cpu().numpy()
        std_predicted_mutual_info = (predicted_mutual_info - (-0.075)) / (0.05 - (-0.075))

        advnet2 = MINE_Net4_3(input_dim=vae_MI2.n_latent + 1, n_latents=Adv_Net_architecture, activation_fun=activation_fun, unbiased_loss=unbiased_loss, initial=initial,
                              save_path='None',data_loader=trainer_vae_MI2_adv, drop_out=adv_drop_out, net_name=adv_model, min=-0.075, max=0.05)
        adv_optimizer2 = torch.optim.Adam(advnet.parameters(), lr=5e-3)
        #To fully train MineNet
        for full_epoch in tqdm(range(400)):
            advnet2.train()
            for tensor_adv in advnet2.data_loader.data_loaders_loop():
                sample_batch_adv, local_l_mean_adv, local_l_var_adv, batch_index_adv, _ = tensor_adv[0]
                x_ = sample_batch_adv
                if trainer_vae_MI2.model.log_variational:
                    x_ = torch.log(1 + x_)
                # Sampling
                qz_m, qz_v, z = trainer_vae_MI2.model.z_encoder(x_, None)
                # z = z.detach()
                ql_m, ql_v, library = trainer_vae_MI2.model.l_encoder(x_)
                '''
                z_batch0_tensor = z[(Variable(torch.LongTensor([1])) - batch_index_adv).squeeze(1).byte()]
                z_batch1_tensor = z[batch_index_adv.squeeze(1).byte()]
                l_batch0_tensor = library[(Variable(torch.LongTensor([1])) - batch_index_adv).squeeze(1).byte()]
                l_batch1_tensor = library[batch_index_adv.squeeze(1).byte()]
                l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
                l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)

                if (l_z_batch0_tensor.shape[0] == 0) or (l_z_batch1_tensor.shape[0] == 0):
                    continue

                pred_xz = advnet2(input=l_z_batch0_tensor)
                pred_x_z = advnet2(input=l_z_batch1_tensor)
                # clip pred_x_z, but not pred_xz
                pred_x_z = torch.min(pred_x_z, Variable(torch.FloatTensor([1])))
                pred_x_z = torch.max(pred_x_z, Variable(torch.FloatTensor([-1])))
                '''
                l_z_joint_tensor = torch.cat((library, z), dim=1)
                z_shuffle_tensor = np.random.permutation(z.detach().numpy())
                z_shuffle_tensor = Variable(torch.from_numpy(z_shuffle_tensor).type(torch.FloatTensor), requires_grad=True)
                l_z_indept_tensor = torch.cat((library, z_shuffle_tensor), dim=1)
                pred_xz = advnet2(input=l_z_joint_tensor)
                pred_x_z = advnet2(input=l_z_indept_tensor)
                if advnet2.unbiased_loss:
                    t = pred_xz
                    et = torch.exp(pred_x_z)
                    if advnet2.ma_et is None:
                        advnet2.ma_et = torch.mean(et).detach().item()
                    advnet2.ma_et += advnet2.ma_rate * (torch.mean(et).detach().item() - advnet2.ma_et)
                    # unbiasing use moving average
                    loss_adv2 = -(torch.mean(t) - (torch.log(torch.mean(et)) * torch.mean(et).detach() / advnet2.ma_et))
                else:
                    loss_adv = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
                    loss_adv2 = -loss_adv  # maximizing loss_adv equals minimizing -loss_adv
                adv_optimizer2.zero_grad()
                loss_adv2.backward()
                adv_optimizer2.step()
        advnet2.eval()
    '''
    pred_xz_train_fully = advnet2(input=l_z_batch0_tensor_train)
    pred_x_z_train_fully = advnet2(input=l_z_batch1_tensor_train)
    pred_x_z_train_fully = torch.min(pred_x_z_train_fully, Variable(torch.FloatTensor([1])))
    pred_x_z_train_fully = torch.max(pred_x_z_train_fully, Variable(torch.FloatTensor([-1])))
    predicted_mutual_info_fully = (torch.mean(pred_xz_train_fully) - torch.log(torch.mean(torch.exp(pred_x_z_train_fully)))).detach().cpu().numpy()
    '''
    pred_xz_train_fully = advnet2(input=l_z_joint_train)
    pred_x_z_train_fully = advnet2(input=l_z_indept_train)
    predicted_mutual_info_fully = (torch.mean(pred_xz_train_fully) - (torch.log(torch.mean(torch.exp(pred_x_z_train_fully))) * torch.mean(torch.exp(pred_x_z_train_fully)).detach() / advnet2.ma_et)).detach().cpu().numpy()

    ELBO_list_train = []
    number_samples = 0
    for tensors in trainer_vae_MI2.train_set:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        reconst_loss, kl_divergence = trainer_vae_MI2.model(sample_batch, local_l_mean, local_l_var, batch_index)
        ELBO = torch.mean(reconst_loss + kl_divergence).detach().cpu().numpy()
        ELBO_list_train += [ELBO*sample_batch.shape[0]]
        number_samples += sample_batch.shape[0]
    std_ELBO_train = (sum(ELBO_list_train)/number_samples - 12000)/(max_reconst - 12000)

    label = '%s_%s_MIScale%s_VaeMI_trainset' % (dataset_name, nuisance_variable, MIScale_index)

    if adv_model == 'MI':
        intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be], 'std_penalty': [std_predicted_mutual_info], 'std_ELBO':[std_ELBO_train], 'penalty_fully': [predicted_mutual_info_fully]})
    #elif adv_model == 'Classifier':
    #    intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be], 'std_penalty': [std_cross_entropy], 'std_ELBO':[std_ELBO_train]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

    asw, nmi, ari, uca = trainer_vae_MI2.test_set.clustering_scores()
    be = trainer_vae_MI2.test_set.entropy_batch_mixing()

    # latent, batch_indices, labels = trainer_vae_MI2.test_set.get_latent(sample=False)
    z_tensor_test = Variable(torch.from_numpy(np.empty([0,10],dtype=float)).type(torch.FloatTensor), requires_grad=True)
    batch_indices_list_test = []
    library_tensor_test = Variable(torch.from_numpy(np.empty([0,1],dtype=float)).type(torch.FloatTensor), requires_grad=True)
    for tensors in trainer_vae_MI2.test_set:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        if trainer_vae_MI2.model.log_variational:
            x_ = torch.log(1 + sample_batch)
        z_tensor_test = torch.cat((z_tensor_test, trainer_vae_MI2.model.z_encoder(x_)[0].cpu()),dim=0)
        batch_indices_list_test += [batch_index.cpu()]
        library_tensor_test = torch.cat((library_tensor_test, trainer_vae_MI2.model.l_encoder(x_)[0].cpu()),dim=0)
    batch_indices_array_test = np.empty([0, 1], dtype=int)
    for i in range(len(batch_indices_list_test)):
        batch_indices_array_test = np.concatenate((batch_indices_array_test, batch_indices_list_test[i].detach().numpy()), axis=0)

    if adv_model == 'MI':
        '''
        batch_index_adv_list = np.ndarray.tolist(batch_indices_array_test)
        z_batch0_tensor_test = z_tensor_test[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
        z_batch1_tensor_test = z_tensor_test[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
        l_batch0_tensor_test = library_tensor_test[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
        l_batch1_tensor_test = library_tensor_test[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
        l_z_batch0_tensor_test = torch.cat((l_batch0_tensor_test, z_batch0_tensor_test), dim=1)
        l_z_batch1_tensor_test = torch.cat((l_batch1_tensor_test, z_batch1_tensor_test), dim=1)
        pred_xz_test = trainer_vae_MI2.adv_model(input=l_z_batch0_tensor_test)
        pred_x_z_test = trainer_vae_MI2.adv_model(input=l_z_batch1_tensor_test)
        predicted_mutual_info = (torch.mean(pred_xz_test) - torch.log(torch.mean(torch.exp(pred_x_z_test)))).detach().cpu().numpy()
        std_predicted_mutual_info = (predicted_mutual_info - (-0.02)) / (0.3 - (-0.02))

        pred_xz_test_fully = advnet2(input=l_z_batch0_tensor_test)
        pred_x_z_test_fully = advnet2(input=l_z_batch1_tensor_test)
        pred_x_z_test_fully = torch.min(pred_x_z_test_fully, Variable(torch.FloatTensor([1])))
        pred_x_z_test_fully = torch.max(pred_x_z_test_fully, Variable(torch.FloatTensor([-1])))
        predicted_mutual_info_fully = (torch.mean(pred_xz_test_fully) - torch.log(torch.mean(torch.exp(pred_x_z_test_fully)))).detach().cpu().numpy()
        '''
        l_z_joint_test = torch.cat((library_tensor_test, z_tensor_test), dim=1)
        z_shuffle_test = np.random.permutation(z_tensor_test.detach().numpy())
        z_shuffle_test = Variable(torch.from_numpy(z_shuffle_test).type(torch.FloatTensor), requires_grad=True)
        l_z_indept_test = torch.cat((library_tensor_test, z_shuffle_test), dim=1)
        pred_xz_test = trainer_vae_MI2.adv_model(input=l_z_joint_test)
        pred_x_z_test = trainer_vae_MI2.adv_model(input=l_z_indept_test)
        predicted_mutual_info = (torch.mean(pred_xz_test) - (torch.log(torch.mean(torch.exp(pred_x_z_test))) * torch.mean(
                torch.exp(pred_x_z_test)).detach() / trainer_vae_MI2.adv_model.ma_et)).detach().cpu().numpy()
        std_predicted_mutual_info = (predicted_mutual_info - (-0.075)) / (0.05 - (-0.075))

        pred_xz_test_fully = advnet2(input=l_z_joint_test)
        pred_x_z_test_fully = advnet2(input=l_z_indept_test)
        predicted_mutual_info_fully = (torch.mean(pred_xz_test_fully) - (torch.log(torch.mean(torch.exp(pred_x_z_test_fully))) * torch.mean(torch.exp(pred_x_z_test_fully)).detach() / advnet2.ma_et)).detach().cpu().numpy()


    elif adv_model == 'Classifier':
        z_l_test = torch.cat((library_tensor_test, z_tensor_test), dim=1)
        batch_indices_tensor_test = Variable(torch.from_numpy(batch_indices_array_test).type(torch.FloatTensor), requires_grad=False)
        logit = trainer_vae_MI2.adv_model(z_l_test)
        cross_entropy = trainer_vae_MI2.adv_criterion(logit, batch_indices_tensor_test).detach().cpu().numpy()
        std_cross_entropy = (-cross_entropy - (-6)) / (-0.2 - (-6))

    ELBO_list_test = []
    number_samples = 0
    for tensors in trainer_vae_MI2.test_set:
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
        reconst_loss, kl_divergence = trainer_vae_MI2.model(sample_batch, local_l_mean, local_l_var, batch_index)
        ELBO = torch.mean(reconst_loss + kl_divergence).detach().cpu().numpy()
        ELBO_list_test += [ELBO* sample_batch.shape[0]]
        number_samples += sample_batch.shape[0]
    std_ELBO_test = (sum(ELBO_list_test) / number_samples - 12000) / (max_reconst - 12000)

    label = '%s_%s_MIScale%s_VaeMI_testset' % (dataset_name, nuisance_variable, MIScale_index)
    if adv_model == 'MI':
        intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be],'std_penalty': [std_predicted_mutual_info], 'std_ELBO': [std_ELBO_test], 'penalty_fully': [predicted_mutual_info_fully]})
    #elif adv_model == 'Classifier':
    #    intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be], 'std_penalty': [std_cross_entropy], 'std_ELBO':[std_ELBO_test]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)
    clustering_metric.to_csv('%s/rep%s/%s_%s_MIScale%s_ClusterMetric.csv' % (result_save_path, taskid, dataset_name, nuisance_variable, MIScale_index), index=None, header=True)

# Run the actual program
if __name__ == "__main__":
    main()

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
