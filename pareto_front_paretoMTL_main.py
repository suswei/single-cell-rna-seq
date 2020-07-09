import os
import argparse
from random import randint
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim

from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
from scvi.models.modules import MINE_Net3, discrete_continuous_info
import pickle

def sample1_sample2(trainer_vae, sample_batch, batch_index):
    x_ = sample_batch
    if trainer_vae.model.log_variational:
        x_ = torch.log(1 + x_)
    # Sampling
    qz_m, qz_v, z = trainer_vae.model.z_encoder(x_, None)

    batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.numpy().ravel())})
    batch_dummy = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)
    batch_dummy = Variable(batch_dummy, requires_grad=True)

    sample1 = torch.cat((z, batch_dummy), 1)  # joint
    shuffle_index = torch.randperm(z.shape[0])
    sample2 = torch.cat((z[shuffle_index], batch_dummy), 1)

    return sample1, sample2, z, batch_dummy

def obj1_train_test(trainer_vae):

    trainer_vae.cal_loss = True
    trainer_vae.cal_adv_loss = False

    obj1_minibatch_list_train, obj1_minibatch_list_test = [], []
    for tensors_list in trainer_vae.train_set:
        loss, _, _ = trainer_vae.two_loss(tensors_list)
        obj1_minibatch_list_train.append(loss.item())

    obj1_train = sum(obj1_minibatch_list_train) / len(obj1_minibatch_list_train)

    for tensors_list in trainer_vae.test_set:
        loss, _, _ = trainer_vae.two_loss(tensors_list)
        obj1_minibatch_list_test.append(loss.item())

    obj1_test = sum(obj1_minibatch_list_test) / len(obj1_minibatch_list_test)

    return obj1_train, obj1_test

def fully_MINE_after_trainerVae(trainer_vae):
    MINE_network = MINE_Net3(input_dim=10 + 2, n_hidden=128, n_layers=10,
                            activation_fun='ELU', unbiased_loss=True, initial='normal')

    MINE_optimizer = optim.Adam(MINE_network.parameters(), lr=5e-5)

    for epoch in range(200):
        MINE_network.train()
        for tensors_list in trainer_vae.data_loaders_loop():
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list[0]

            sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)
            t = MINE_network(sample1)
            et = torch.exp(MINE_network(sample2))

            #use unbiased loss
            if MINE_network.ma_et is None:
                MINE_network.ma_et = torch.mean(et).detach().item()  # detach means will not calculate gradient for ma_et, ma_et is just a number
            MINE_network.ma_et = (1 - MINE_network.ma_rate) * MINE_network.ma_et + MINE_network.ma_rate * torch.mean(et).detach().item()

            loss = -(torch.mean(t) - (1 / MINE_network.ma_et) * torch.mean(et))

            #if epoch % 40 == 0:
            #    MINE_estimator_minibatch = torch.mean(t) - torch.log(torch.mean(et))
            #    NN_estimator = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
            #    print('Epoch: {}, MINE_MI: {}, NN: {}.'.format(epoch, MINE_estimator_minibatch,NN_estimator))

            loss.backward()
            MINE_optimizer.step()
            MINE_optimizer.zero_grad()
            trainer_vae.optimizer.zero_grad()

    with torch.no_grad():
        MINE_network.eval()
        z_all_train = torch.empty(0, 10)
        batch_all_train = torch.empty(0, 2)

        for tensors_list in trainer_vae.train_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

            sample1, sample2, z, batch_dummy= sample1_sample2(trainer_vae, sample_batch, batch_index)
            z_all_train = torch.cat((z_all_train, z), 0)
            batch_all_train = torch.cat((batch_all_train, batch_dummy), 0)

        sample1_all_train = torch.cat((z_all_train, batch_all_train), 1)  # joint
        shuffle_index = torch.randperm(z_all_train.shape[0])
        sample2_all_train = torch.cat((z_all_train[shuffle_index], batch_all_train), 1)
        t_all_train = MINE_network(sample1_all_train)
        et_all_train = torch.exp(MINE_network(sample2_all_train))
        MINE_estimator_train = torch.mean(t_all_train) - torch.log(torch.mean(et_all_train))

        z_all_test = torch.empty(0, 10)
        batch_all_test = torch.empty(0, 2)

        for tensors_list in trainer_vae.test_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

            sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)
            z_all_test = torch.cat((z_all_test, z), 0)
            batch_all_test = torch.cat((batch_all_test, batch_dummy), 0)

        sample1_all_test = torch.cat((z_all_test, batch_all_test), 1)  # joint
        shuffle_index = torch.randperm(z_all_test.shape[0])
        sample2_all_test = torch.cat((z_all_test[shuffle_index], batch_all_test), 1)
        t_all_test = MINE_network(sample1_all_test)
        et_all_test = torch.exp(MINE_network(sample2_all_test))
        MINE_estimator_test = torch.mean(t_all_test) - torch.log(torch.mean(et_all_test))

    print('MINE MI train: {}, MINE MI test: {}'.format(MINE_estimator_train.detach().item(), MINE_estimator_test.detach().item()))

    # At the final epoch, NN_estimator for each minibatch fluctuates between 0.3 and 0.5
    # when without adaptive learning rate, MINE MI train: 0.6393658518791199, MINE MI test: 0.6368539929389954
    # when with adaptive learning rate, MINE MI train: 0.30684590339660645, MINE MI test: 0.3084377646446228
    # however, as there is no validation dataset, to automatically adjust the learning rate is inapplicable.

    return MINE_estimator_train.detach().item(), MINE_estimator_test.detach().item()

def NN_train_test(trainer_vae):

    NN_train_list, NN_test_list = [], []
    for tensors_list in trainer_vae.train_set:

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
        sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)

        NN_estimator_minibatch_train = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
        NN_train_list.append(NN_estimator_minibatch_train)
    NN_train = sum(NN_train_list)/len(NN_train_list)

    for tensors_list in trainer_vae.test_set:

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
        sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)

        NN_estimator_minibatch_test = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
        NN_test_list.append(NN_estimator_minibatch_test)
    NN_test = sum(NN_test_list)/len(NN_test_list)

    return NN_train, NN_test

def main( ):

    parser = argparse.ArgumentParser(description='pareto_front_paretoMTL')

    parser.add_argument('--taskid', type=int, default=1000 + randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--dataset_name', type=str, default='muris_tabula',
                        help='the name of the dataset')

    parser.add_argument('--confounder', type=str, default='batch',
                        help='the name of the confounder variable')

    # for scVI
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

    parser.add_argument('--use_batches', action='store_true', default=True,
                        help='whether to use batches or not in scVI')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size for scVI')

    # for MINE
    parser.add_argument('--conf_estimator', type=str, default='MINE_MI',
                        help='the method used to estimate confounding effect')

    parser.add_argument('--adv_n_hidden', type=int, default=128,
                        help='the number of hidden nodes in each hidden layer for MINE')

    parser.add_argument('--adv_n_layers', type=int, default=10,
                        help='the number of hidden layers for MINE')

    parser.add_argument('--activation_fun', type=str, default='ELU',
                        help='the activation function used for MINE')

    parser.add_argument('--unbiased_loss', action='store_true', default=True,
                        help='whether to use unbiased loss or not in MINE')

    # for training
    parser.add_argument('--pre_epochs', type=int, default=100,
                        help='number of epochs to pre-train scVI')

    parser.add_argument('--pre_adv_epochs', type=int, default=100,
                        help='number of epochs to pre-train MINE')

    parser.add_argument('--adv_lr', type=float, default=5e-4,
                        help='learning rate in MINE pre-training and adversarial training')

    parser.add_argument('--n_epochs', type=int, default=51,
                        help='number of epochs to train scVI and MINE')

    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for scVI')

    parser.add_argument('--obj2_scale', type=float, default=0.5,
                        help='the scale to standardize the second objective')

    parser.add_argument('--n_tasks', type=int, default=2,
                        help='number of objectives for the multiple optimization problem')

    parser.add_argument('--npref', type=int, default=10,
                        help='number of subproblems when the multiple optimization problem is decomposed')

    parser.add_argument('--pref_idx', type=int, default=0,
                        help='which subproblem')

    parser.add_argument('--nsamples_z', type=int, default=200,
                        help='number of z sampled from aggregated posterior for nearest neighbor method')

    # general usage
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    if args.activation_fun == 'ELU':
        args.adv_w_initial = 'normal'

    #load dataset
    data_save_path = './data/pareto_front_paretoMTL/%s' % (args.dataset_name)
    if not os.path.exists('./data/pareto_front_paretoMTL/%s' % (args.dataset_name)):
        os.makedirs('./data/pareto_front_paretoMTL/%s' % (args.dataset_name))

    if args.dataset_name == 'muris_tabula':
        dataset1 = TabulaMuris('facs', save_path=data_save_path)
        dataset2 = TabulaMuris('droplet', save_path=data_save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

    #generate a random seed to split training and testing dataset
    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 1), dtype=np.uint32)
    desired_seed = int(desired_seeds[0, 0])

    args.save_path = './result/pareto_front_paretoMTL/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.taskid)
    if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.taskid)):
        os.makedirs('./result/pareto_front_paretoMTL/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.taskid))

    # batch1_ratio = gene_dataset.batch_indices[gene_dataset.batch_indices[:,0]==1].shape[0]/gene_dataset.batch_indices.shape[0]

    # calculate ratio to split the gene_dataset into training and testing dataset
    # to avoid the case when there are very few input data points of the last minibatch of every epoch
    intended_trainset_size = int(gene_dataset._X.shape[0] / args.batch_size / 10) * 10 * 0.8 * 128 + (
            int(gene_dataset._X.shape[0] / args.batch_size) % 10) * 128
    args.train_size = int(intended_trainset_size / gene_dataset._X.shape[0] * 1e6) / 1e6

    # If train vae alone
    # vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * True, n_labels=gene_dataset.n_labels,
    #          n_hidden=128, n_latent=10, n_layers_encoder=2, n_layers_decoder=2, dropout_rate=0.1,
    #          reconstruction_loss='zinb', batch1_ratio=batch1_ratio, nsamples_z=128, adv=False,
    #          std=False, save_path='None')
    # frequency controls how often the statistics in trainer_vae.model are evaluated by compute_metrics() function in trainer.py
    # trainer_vae = UnsupervisedTrainer(vae, gene_dataset, batch_size=128, train_size=train_size, seed=desired_seed,
    #                               use_cuda=False, frequency=10, kl=1)
    # trainer_vae.train(n_epochs=400, lr=0.001)
    # torch.save(trainer_vae.model.state_dict(), save_path) #saved into pickle file

    # during vae training, at early epochs, NN_estimator for a minibatch could be as high as 0.9,
    # at final epochs, NN_estimator for each minibatch fluctuates mainly between 0.3 and 0.5.

    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * args.use_batches,
                    n_labels=gene_dataset.n_labels, n_hidden=args.n_hidden, n_latent=args.n_latent,
                    n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,
                    dropout_rate=args.dropout_rate, reconstruction_loss=args.reconstruction_loss)
    trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size,
                                      seed=desired_seed, use_cuda=args.use_cuda, frequency=10, kl=1,
                                      adv=True, adv_estimator=args.conf_estimator, adv_n_hidden=args.adv_n_hidden,
                                      adv_n_layers=args.adv_n_layers, adv_activation_fun=args.activation_fun,
                                      unbiased_loss=args.unbiased_loss, adv_w_initial=args.adv_w_initial,
                                      aggregated_posterior=False, save_path=args.save_path)
    # TODO: it is better to be controled by self.on_epoch_begin(), it should be modified later
    trainer_vae.kl_weight = 1

    obj1_minibatch_list, _, obj2_minibatch_list = trainer_vae.paretoMTL_train(pre_epochs= args.pre_epochs, pre_adv_epochs = args.pre_adv_epochs, adv_lr = args.adv_lr, n_epochs = args.n_epochs,
                                                                              lr = args.lr, n_tasks = args.n_tasks, npref = args.npref, pref_idx = args.pref_idx, obj2_scale = args.obj2_scale)
    #obj1 for the whole training and testing set
    obj1_train, obj1_test = obj1_train_test(trainer_vae)

    full_MINE_train, full_MINE_test = fully_MINE_after_trainerVae(trainer_vae)

    NN_train, NN_test = NN_train_test(trainer_vae)

    asw_train, nmi_train, ari_train, uca_train = trainer_vae.train_set.clustering_scores()
    be_train = trainer_vae.train_set.entropy_batch_mixing()

    asw_test, nmi_test, ari_test, uca_test = trainer_vae.test_set.clustering_scores()
    be_test = trainer_vae.test_set.entropy_batch_mixing()

    args_dict = vars(args)
    with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(args_dict, f)

    results_dict = {'obj1_minibatch': [obj1_minibatch_list[-1]],
                    'obj2_minibatch': [obj2_minibatch_list[-1]],
                    'obj1_train': [obj1_train],
                    'full_MINE_train': [full_MINE_train],
                    'NN_train': [NN_train],
                    'obj1_test': [obj1_test],
                    'full_MINE_test': [full_MINE_test],
                    'NN_test': [NN_test],
                    'asw_train': [asw_train],
                    'nmi_train': [nmi_train],
                    'ari_train': [ari_train],
                    'uca_train': [uca_train],
                    'be_train': [be_train],
                    'asw_test': [asw_test],
                    'nmi_test': [nmi_test],
                    'ari_test': [ari_test],
                    'uca_test': [uca_test],
                    'be_test': [be_test]
                    }
    with open('{}/results.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(results_dict, f)
    print(results_dict)
# Run the actual program
if __name__ == "__main__":
    main()
