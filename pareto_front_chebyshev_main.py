# Run 100 Monte Carlo Samples for each dataset, as scVI is variable for each run.
# For mouse marrow dataset, the hyperparameter tried n_layers for scvi=2, n_hidden=128, n_latent=10, reconstruction_loss=zinb, dropout_rate=0.1, lr=0.001, n_epochs=250, training_size-0.8
# For Pbmc dataset, the hyperparameter tried n_layers_for_scvi=1, n_latent=256, n_latent=14, dropout_rate=0.5, lr=0.01, n_epochs=170, training_size=0.8.

import os
import argparse
from random import randint
import numpy as np
import pandas as pd
from scvi.dataset import *
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.models.modules import MINE_Net
from scvi.inference import UnsupervisedTrainer
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

def std_obj1_train_test(trainer_vae, args):

    trainer_vae.cal_loss = True
    trainer_vae.cal_adv_loss = False

    trainset_obj1_minibatch_list = []
    for tensors_list in trainer_vae.train_set:
        loss, _, _ = trainer_vae.two_loss(tensors_list)
        trainset_obj1_minibatch_list.append(loss.item())

    obj1_trainset = sum(trainset_obj1_minibatch_list) / len(trainset_obj1_minibatch_list)
    std_obj1_trainset = (obj1_trainset - args.min_obj1) / (args.max_obj1 - args.min_obj1)

    testset_obj1_minibatch_list = []
    for tensors_list in trainer_vae.test_set:
        loss, _, _ = trainer_vae.two_loss(tensors_list)
        testset_obj1_minibatch_list.append(loss.item())

    obj1_testset = sum(testset_obj1_minibatch_list) / len(testset_obj1_minibatch_list)
    std_obj1_testset = (obj1_testset - args.min_obj1) / (args.max_obj1 - args.min_obj1)

    return std_obj1_trainset, std_obj1_testset

def fully_MINE_after_trainerVae(trainer_vae):
    MINE_network = MINE_Net(input_dim=10 + 2, n_hidden=128, n_layers=10,
                            activation_fun='ELU', unbiased_loss=True, initial='normal')

    MINE_optimizer = optim.Adam(MINE_network.parameters(), lr=5e-5)
    #scheduler_MINE = ReduceLROnPlateau(MINE_optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(300):
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

        #with torch.no_grad():  # to save memory, no intermediate activations used for gradient calculation is stored.

        #    MINE_network.eval()

        #    train_loss_minibatch_list = []
        #    for tensors_list in trainer_vae.train_set:
        #        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

        #        sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)
        #        train_t = MINE_network(sample1)
        #        train_et = torch.exp(MINE_network(sample2))

        #        train_loss_minibatch = -(torch.mean(train_t) - (1 / MINE_network.ma_et) * torch.mean(train_et))

        #        train_loss_minibatch_list.append(train_loss_minibatch.item())

        #    train_loss_one = np.average(train_loss_minibatch_list)
        #    scheduler_MINE.step(train_loss_one)

    with torch.no_grad():
        MINE_network.eval()
        train_z_all = torch.empty(0, 10)
        train_batch_all = torch.empty(0, 2)

        for tensors_list in trainer_vae.train_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

            sample1, sample2, z, batch_dummy= sample1_sample2(trainer_vae, sample_batch, batch_index)
            train_z_all = torch.cat((train_z_all, z), 0)
            train_batch_all = torch.cat((train_batch_all, batch_dummy), 0)

        train_sample1_all = torch.cat((train_z_all, train_batch_all), 1)  # joint
        shuffle_index = torch.randperm(train_z_all.shape[0])
        train_sample2_all = torch.cat((train_z_all[shuffle_index], train_batch_all), 1)
        train_t_all = MINE_network(train_sample1_all)
        train_et_all = torch.exp(MINE_network(train_sample2_all))
        train_MINE_estimator = torch.mean(train_t_all) - torch.log(torch.mean(train_et_all))

        test_z_all = torch.empty(0, 10)
        test_batch_all = torch.empty(0, 2)

        for tensors_list in trainer_vae.test_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

            sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index)
            test_z_all = torch.cat((test_z_all, z), 0)
            test_batch_all = torch.cat((test_batch_all, batch_dummy), 0)

        test_sample1_all = torch.cat((test_z_all, test_batch_all), 1)  # joint
        shuffle_index = torch.randperm(test_z_all.shape[0])
        test_sample2_all = torch.cat((test_z_all[shuffle_index], test_batch_all), 1)
        test_t_all = MINE_network(test_sample1_all)
        test_et_all = torch.exp(MINE_network(test_sample2_all))
        test_MINE_estimator = torch.mean(test_t_all) - torch.log(torch.mean(test_et_all))

    print('MINE MI train: {}, MINE MI test: {}'.format(train_MINE_estimator.detach().item(), test_MINE_estimator.detach().item()))

    # At the final epoch, NN_estimator for each minibatch fluctuates between 0.3 and 0.5
    # when without adaptive learning rate, MINE MI train: 0.6393658518791199, MINE MI test: 0.6368539929389954
    # when with adaptive learning rate, MINE MI train: 0.30684590339660645, MINE MI test: 0.3084377646446228

    return train_MINE_estimator.detach().item(), test_MINE_estimator.detach().item()


def main( ):

    parser = argparse.ArgumentParser(description='Pareto_Front')

    parser.add_argument('--taskid', type=int, default=1000 + randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--dataset_name', type=str, default='muris_tabula',
                        help='the name of the dataset')

    parser.add_argument('--confounder', type=str, default='batch',
                        help='the name of the confounder variable')


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

    parser.add_argument('--use_batches', action='store_true', default=True,
                        help='whether to use batches or not in scVI')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch size for scVI')


    #for MINE
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


    #for training
    parser.add_argument('--pre_n_epochs', type=int, default=50,
                        help='number of epochs to pre-train scVI')

    parser.add_argument('--pre_lr', type=float, default=1e-3,
                        help='learning rate to pre-train scVI')

    parser.add_argument('--pre_adv_epochs', type=int, default=100,
                        help='number of epochs to pre-train MINE')

    parser.add_argument('--adv_lr', type=float, default=5e-5,
                        help='learning rate in MINE pre-training and adversarial training')

    parser.add_argument('--n_epochs', type=int, default=600,
                        help='number of epochs to train scVI and MINE')

    parser.add_argument('--main_lr', type=float, default=1e-3,
                        help='learning rate of scVI in adversarial training')

    parser.add_argument('--std', action='store_true', default=True,
                        help='whether to standardize value of objective 1 and objective 2')

    parser.add_argument('--scale', type=float, default=0,
                        help='the scale for the standardized value of objective 2')

    parser.add_argument('--min_obj1', type=float, default=12000,
                        help='the minimum value used to standardize value of objective 1')

    parser.add_argument('--max_obj1', type=float, default=20000,
                        help='the maximum value used to standardize value of objective 1')

    parser.add_argument('--min_obj2', type=float, default=-0.1,
                        help='the minimum value used to standardize value of objective 2')

    parser.add_argument('--max_obj2', type=float, default=0.9,
                        help='the maximum value used to standardize value of objective 2')

    parser.add_argument('--MCs', type=int, default=100,
                        help='the number of repetitions for each scale')

    parser.add_argument('--nsamples_z', type=int, default=200,
                        help='number of z sampled from aggregated posterior for nearest neighbor method')

    #general usage
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before logging training status')

    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    if args.activation_fun == 'ELU':
        args.adv_w_initial = 'normal'

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
    seed_idx =  args.taskid - int(args.taskid/args.MCs)*args.MCs
    desired_seed = int(desired_seeds[0, seed_idx])

    args.save_path = './result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder, args.taskid)
    if not os.path.exists('./result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder, args.taskid)):
        os.makedirs('./result/pareto_front_scVI_MINE/%s/%s/taskid%s' % (args.dataset_name, args.confounder, args.taskid))

    #batch1_ratio = gene_dataset.batch_indices[gene_dataset.batch_indices[:,0]==1].shape[0]/gene_dataset.batch_indices.shape[0]

    #calculate ratio to split the gene_dataset into training and testing dataset
    # to avoid the case when there are very few input data points of the last minibatch of every epoch
    intended_trainset_size=int(gene_dataset._X.shape[0]/args.batch_size/10)*10*0.8*128 + (int(gene_dataset._X.shape[0]/args.batch_size) % 10)*128
    args.train_size = int(intended_trainset_size/gene_dataset._X.shape[0]*1e6)/1e6

    #If train vae alone
    #vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * True, n_labels=gene_dataset.n_labels,
    #          n_hidden=128, n_latent=10, n_layers_encoder=2, n_layers_decoder=2, dropout_rate=0.1,
    #          reconstruction_loss='zinb', batch1_ratio=batch1_ratio, nsamples_z=128, adv=False,
    #          std=False, save_path='None')
    # frequency controls how often the statistics in trainer_vae.model are evaluated by compute_metrics() function in trainer.py
    #trainer_vae = UnsupervisedTrainer(vae, gene_dataset, batch_size=128, train_size=train_size, seed=desired_seed,
    #                               use_cuda=False, frequency=10, kl=1)
    #trainer_vae.train(n_epochs=400, lr=0.001)
    #torch.save(trainer_vae.model.state_dict(), save_path) #saved into pickle file

    # during vae training, at early epochs, NN_estimator for a minibatch could be as high as 0.9,
    # at final epochs, NN_estimator for each minibatch fluctuates mainly between 0.3 and 0.5.


    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * args.use_batches, n_labels=gene_dataset.n_labels,
                    n_hidden=args.n_hidden, n_latent=args.n_latent, n_layers_encoder=args.n_layers_encoder,
                    n_layers_decoder=args.n_layers_decoder, dropout_rate=args.dropout_rate, reconstruction_loss=args.reconstruction_loss)
    trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size, seed=desired_seed,
                                      use_cuda=args.use_cuda, frequency=10, kl=1,
                                      adv=True, adv_estimator=args.conf_estimator, adv_n_hidden=args.adv_n_hidden, adv_n_layers=args.adv_n_layers,
                                      adv_activation_fun=args.activation_fun, unbiased_loss=args.unbiased_loss,
                                      adv_w_initial=args.adv_w_initial, aggregated_posterior=False, save_path=args.save_path)
    # TODO: it is better to be controled by self.on_epoch_begin(), it should be modified later
    trainer_vae.kl_weight = 1

    obj1_minibatch_list, _, obj2_minibatch_list = trainer_vae.adversarial_train(pre_n_epochs=args.pre_n_epochs, pre_lr=args.pre_lr, pre_adv_epochs=args.pre_adv_epochs,
                                            adv_lr=args.adv_lr, n_epochs=args.n_epochs, main_lr=args.main_lr, std=args.std,
                                            min_obj1=args.min_obj1, max_obj1=args.max_obj1, min_obj2=args.min_obj2, max_obj2=args.max_obj2,
                                            scale=args.scale)
    #std_obj1 and std_obj2 of the last minibatch of the last epoch
    std_obj1_minibatch = (obj1_minibatch_list[-1] - args.min_obj1)/(args.max_obj1 - args.min_obj1)
    std_obj2_minibatch = (obj2_minibatch_list[-1] - args.min_obj2)/(args.max_obj2 - args.min_obj2)

    asw_train, nmi_train, ari_train, uca_train = trainer_vae.train_set.clustering_scores()
    be_train = trainer_vae.train_set.entropy_batch_mixing()

    asw_test, nmi_test, ari_test, uca_test = trainer_vae.test_set.clustering_scores()
    be_test = trainer_vae.test_set.entropy_batch_mixing()

    #n_samples_tsne = 1000
    if args.scale == 0 or args.scale == 1:
        trainer_vae.train_set.show_t_sne(1000, color_by='batches', save_name='{}/trainset_tsne'.format(args.save_path))
        trainer_vae.test_set.show_t_sne(1000, color_by='batches', save_name='{}/testset_tsne'.format(args.save_path))
    # latent, batch_indices, labels = trainer_vae.train_set.get_latent(sample=False)

    # calculate std_obj1 (std_neg_ELBO) for the whole train dataset and test dataset
    std_obj1_train, std_obj1_test = std_obj1_train_test(trainer_vae, args)

    # Train a new MINE network after trainer_vae's training, and calculate non_std obj2 for the whole train dataset and test dataset
    full_MINE_train, full_MINE_test = fully_MINE_after_trainerVae(trainer_vae)

    args_dict = vars(args)
    with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(args_dict, f)

    results_dict = {'std_obj1_minibatch': [std_obj1_minibatch],
                    'std_obj2_minibatch': [std_obj2_minibatch],
                    'std_obj1_train': [std_obj1_train],
                    'full_MINE_train': [full_MINE_train],
                    'std_obj1_test': [std_obj1_test],
                    'full_MINE_test': [full_MINE_test],
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

# Run the actual program
if __name__ == "__main__":
    main()

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
