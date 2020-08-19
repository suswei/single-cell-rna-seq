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
from scvi.models.modules import MINE_Net, discrete_continuous_info, hsic
import pickle
import matplotlib.pyplot as plt

def sample1_sample2(trainer_vae, sample_batch, batch_index, type):
    x_ = sample_batch
    if trainer_vae.model.log_variational:
        x_ = torch.log(1 + x_)
    # Sampling
    qz_m, qz_v, z = trainer_vae.model.z_encoder(x_, None)

    batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.numpy().ravel())})
    batch_dummy = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)
    batch_dummy = Variable(batch_dummy, requires_grad=True)

    if type == 'MINE':
        sample1 = torch.cat((z, batch_dummy), 1)  # joint
        shuffle_index = torch.randperm(z.shape[0])
        sample2 = torch.cat((z[shuffle_index], batch_dummy), 1)

        return sample1, sample2, z, batch_dummy
    else:
        return None, None, z, batch_dummy

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

def MINE_after_trainerVae(trainer_vae):
    MINE_network = MINE_Net(input_dim=10 + 2, n_hidden=128, n_layers=10,
                            activation_fun='ELU', unbiased_loss=True, initial='normal')

    MINE_optimizer = optim.Adam(MINE_network.parameters(), lr=5e-5)

    for epoch in range(200):
        MINE_network.train()
        for tensors_list in trainer_vae.data_loaders_loop():
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list[0]

            sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, 'MINE')
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

            sample1, sample2, z, batch_dummy= sample1_sample2(trainer_vae, sample_batch, batch_index, 'MINE')
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

            sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, 'MINE')
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

def HSIC_NN_train_test(trainer_vae, type):

    estimator_train_list, estimator_test_list = [], []
    for tensors_list in trainer_vae.train_set:

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
        _, _, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, type)
        if type == 'HSIC':
            estimator_minibatch_train = hsic(z, batch_dummy)
            estimator_train_list.append(estimator_minibatch_train.item())
        elif type == 'NN':
            estimator_minibatch_train = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
            estimator_train_list.append(estimator_minibatch_train)
    estimator_train = sum(estimator_train_list)/len(estimator_train_list)

    for tensors_list in trainer_vae.test_set:

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
        _, _, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, type)
        if type == 'HSIC':
            estimator_minibatch_test = hsic(z, batch_dummy)
            estimator_test_list.append(estimator_minibatch_test.item())
        elif type == 'NN':
            estimator_minibatch_test = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
            estimator_test_list.append(estimator_minibatch_test)
    estimator_test = sum(estimator_test_list)/len(estimator_test_list)

    return estimator_train, estimator_test

def draw_diagnosis_plot(list1,  list2, name1, name2, title, path):
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(0, len(list1), 1), list1)
    plt.title('{}'.format(title), fontsize=5)
    plt.ylabel('{}'.format(name1))

    plt.subplot(212)
    plt.plot(np.arange(0, len(list2), 1), list2)
    plt.ylabel('{}'.format(name2))
    plt.xlabel('epochs * minibatches')

    plt.savefig("{}.png".format(path))
    plt.close()

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

    parser.add_argument('--train_size', type=float, default=0.8,
                        help='the ratio to split the training and testing data set')

    # for MINE
    parser.add_argument('--adv_estimator', type=str, default='MINE',
                        help='the method used to estimate confounding effect')

    parser.add_argument('--adv_n_hidden', type=int, default=128,
                        help='the number of hidden nodes in each hidden layer for MINE')

    parser.add_argument('--adv_n_layers', type=int, default=10,
                        help='the number of hidden layers for MINE')

    parser.add_argument('--adv_activation_fun', type=str, default='ELU',
                        help='the activation function used for MINE')

    parser.add_argument('--unbiased_loss', action='store_true', default=True,
                        help='whether to use unbiased loss or not in MINE')

    #for pre_train
    parser.add_argument('--pre_train', action='store_true', default=False,
                        help='whether to pre train neural network')

    parser.add_argument('--pre_epochs', type=int, default=250,
                        help='number of epochs to pre-train scVI')

    parser.add_argument('--pre_adv_epochs', type=int, default=100,
                        help='number of epochs to pre-train MINE')

    parser.add_argument('--pre_lr', type=float, default=1e-3,
                        help='learning rate in scVI pre-training')

    parser.add_argument('--adv_lr', type=float, default=5e-5,
                        help='learning rate in MINE pre-training and adversarial training')

    #for gradnorm
    parser.add_argument('--gradnorm_hypertune', action='store_true', default=False,
                        help='whether to tune hyperparameter for gradnorm')

    parser.add_argument('--alpha', type=float, default=2,
                        help='hyperparameter alpha for gradnorm')

    parser.add_argument('--gradnorm_epochs', type=int, default=100,
                        help='epochs for gradnorm training')

    parser.add_argument('--gradnorm_lr', type=float, default=1e-3,
                        help='learning rate for gradnorm training')

    #to get min and max for obj1 and obj2 to standardize
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='whether to get min and max value for obj1 and obj2 to standardize')

    #for paretoMTL

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs to train scVI and MINE')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for paretoMTL')

    parser.add_argument('--n_tasks', type=int, default=2,
                        help='number of objectives for the multiple optimization problem')

    parser.add_argument('--npref', type=int, default=10,
                        help='number of subproblems when the multiple optimization problem is decomposed')

    parser.add_argument('--pref_idx', type=int, default=0,
                        help='which subproblem')

    parser.add_argument('--gradnorm_paretoMTL', action='store_true', default=False,
                        help='whether to use gradnorm during paretoMTL')

    parser.add_argument('--gradnorm_weight_lowlimit', type=float, default=1e-6,
                        help='the low limit for the smaller weight in gradnorm')


    parser.add_argument('--std_paretoMTL', action='store_true', default=False,
                        help='whether to standardize the two objectives')

    parser.add_argument('--obj1_max', type=float, default=18000,
                        help='maximum value for objective 1')

    parser.add_argument('--obj1_min', type=float, default=12000,
                        help='minimum value for objective 1')

    parser.add_argument('--obj2_max', type=float, default=0.4,
                        help='maximum value for objective 2')

    parser.add_argument('--obj2_min', type=float, default=-0.1,
                        help='minimum value for objective 2')

    parser.add_argument('--n_samples_tsne', type=int, default=1500,
                        help='the number of samples for tsne plot')

    parser.add_argument('--MCs', type=int, default=100,
                        help='the number to repeat pareto MTL')

    parser.add_argument('--MC', type=int, default=0,
                        help='which MC')

    # general usage
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--mode", default='client')

    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    if args.adv_activation_fun == 'ELU':
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
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, args.MCs), dtype=np.uint32)
    if args.pre_train == True:
        desired_seed = int(desired_seeds[0, args.taskid])
    else:
        if args.gradnorm_hypertune == True:
            desired_seed = int(desired_seeds[0, 0])
        else:
            desired_seed = int(desired_seeds[0, int(args.taskid/args.npref)])

    if args.pre_train == True:
        args.save_path = './result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid)
        if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid)):
            os.makedirs('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid))
    else:
        if args.gradnorm_hypertune == True:
            args.save_path = './result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, 0)
            if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, 0)):
                print('Error: please pretrain first!')
        else:
            args.save_path = './result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, int(args.taskid/args.npref))
            if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, int(args.taskid/args.npref))):
                print('Error: please pretrain first!')

    # calculate ratio to split the gene_dataset into training and testing dataset
    # to avoid the case when there are very few input data points of the last minibatch in every epoch
    intended_trainset_size = int(gene_dataset._X.shape[0] / args.batch_size / 10) * 10 * args.train_size * args.batch_size + (
            int(gene_dataset._X.shape[0] / args.batch_size) % 10) * 128
    args.train_size = int(intended_trainset_size / gene_dataset._X.shape[0] * 1e6) / 1e6

    # If train vae alone
    # vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * True, n_labels=gene_dataset.n_labels,
    #          n_hidden=128, n_latent=10, n_layers_encoder=2, n_layers_decoder=2, dropout_rate=0.1,
    #          reconstruction_loss='zinb', batch1_ratio=batch1_ratio, nsamples_z=128,
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
    if args.adv_estimator == 'MINE':
        trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size,
                                          seed=desired_seed, use_cuda=args.use_cuda, frequency=10, kl=1, adv_estimator=args.adv_estimator,
                                          adv_n_hidden=args.adv_n_hidden, adv_n_layers=args.adv_n_layers, adv_activation_fun=args.adv_activation_fun,
                                          unbiased_loss=args.unbiased_loss, adv_w_initial=args.adv_w_initial)
    elif args.adv_estimator == 'HSIC':
        trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size,
                                          seed=desired_seed, use_cuda=args.use_cuda, frequency=10, kl=1, adv_estimator=args.adv_estimator)

    # TODO: it is better to be controled by self.on_epoch_begin(), it should be modified later
    trainer_vae.kl_weight = 1

    if args.pre_train == True:
        trainer_vae.pretrain_gradnorm_paretoMTL(pre_train=args.pre_train, pre_epochs=args.pre_epochs, pre_lr=args.pre_lr,
                        pre_adv_epochs=args.pre_adv_epochs, adv_lr=args.adv_lr, path=args.save_path)
    elif args.standardize == True:
        obj1_minibatch_list, obj2_minibatch_list = trainer_vae.pretrain_gradnorm_paretoMTL(
            path=args.save_path, standardize=args.standardize, n_epochs=args.n_epochs, n_tasks=args.n_tasks,
            npref=args.npref, pref_idx=args.pref_idx)
        if args.pref_idx == 0:
            obj2_max = max(obj2_minibatch_list) * (1 + 0.1)  # 1 + 0.1 or 1-0.1 is for variation in different MCs
            obj2_min = min(obj2_minibatch_list) * (1 - 0.1)
            print('obj2_max: {}, obj2_min: {}'.format(obj2_max, obj2_min))
        if args.pref_idx == 9:
            obj1_max = max(obj1_minibatch_list) * (1 + 0.1)
            obj1_min = min(obj1_minibatch_list) * (1 - 0.1)
            print('obj1_max: {}, obj1_min: {}'.format(obj1_max, obj1_min))
    else:
        if args.gradnorm_hypertune == True:
            gradnorm_weights, weightloss1_list, weightloss2_list, obj1_minibatch_list, obj2_minibatch_list = trainer_vae.pretrain_gradnorm_paretoMTL(
                path=args.save_path, gradnorm_hypertune=args.gradnorm_hypertune, lr=args.lr, adv_lr=args.adv_lr, alpha=args.alpha,
                gradnorm_epochs=args.gradnorm_epochs, gradnorm_lr=args.gradnorm_lr)
        else:
            if args.std_paretoMTL == True:
                obj1_minibatch_list, obj2_minibatch_list = trainer_vae.pretrain_gradnorm_paretoMTL(path=args.save_path, lr=args.lr, adv_lr=args.adv_lr,
                    std_paretoMTL=args.std_paretoMTL,obj1_max=args.obj1_max, obj1_min=args.obj1_min, obj2_max=args.obj2_max, obj2_min=args.obj2_min,
                    n_epochs = args.n_epochs, n_tasks = args.n_tasks, npref = args.npref, pref_idx = args.pref_idx, taskid=args.taskid)
            elif args.gradnorm_paretoMTL == True:
                obj1_minibatch_list, obj2_minibatch_list = trainer_vae.pretrain_gradnorm_paretoMTL(path=args.save_path,
                    lr=args.lr, adv_lr=args.adv_lr, gradnorm_paretoMTL=args.gradnorm_paretoMTL,
                    alpha=args.alpha, gradnorm_lr=args.gradnorm_lr, gradnorm_weight_lowlimit=args.gradnorm_weight_lowlimit,
                    obj1_max=args.obj1_max, obj1_min=args.obj1_min, obj2_max=args.obj2_max, obj2_min=args.obj2_min,
                    n_epochs=args.n_epochs, n_tasks=args.n_tasks, npref=args.npref, pref_idx=args.pref_idx, taskid=args.taskid)

        #obj1 for the whole training and testing set
        obj1_train, obj1_test = obj1_train_test(trainer_vae)

        if trainer_vae.adv_estimator == 'MINE':
            obj2_train, obj2_test = MINE_after_trainerVae(trainer_vae)
        elif trainer_vae.adv_estimator == 'HSIC':
            obj2_train, obj2_test = HSIC_NN_train_test(trainer_vae,'HSIC')

        NN_train, NN_test = HSIC_NN_train_test(trainer_vae, 'NN')

        asw_train, nmi_train, ari_train, uca_train = trainer_vae.train_set.clustering_scores()
        be_train = trainer_vae.train_set.entropy_batch_mixing()

        asw_test, nmi_test, ari_test, uca_test = trainer_vae.test_set.clustering_scores()
        be_test = trainer_vae.test_set.entropy_batch_mixing()

        results_dict = {
                        'obj1_train': [obj1_train],
                        'obj2_train': [obj2_train],
                        'NN_train': [NN_train],
                        'obj1_test': [obj1_test],
                        'obj2_test': [obj2_test],
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

        if args.gradnorm_hypertune == True:
            args.save_path = './result/pareto_front_paretoMTL/{}/{}/gradnorm_hypertune/taskid{}'.format(args.dataset_name, args.confounder,args.taskid)
            if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/gradnorm_hypertune/taskid{}'.format(args.dataset_name, args.confounder, args.taskid)):
                os.makedirs('./result/pareto_front_paretoMTL/{}/{}/gradnorm_hypertune/taskid{}'.format(args.dataset_name, args.confounder, args.taskid))

            results_dict.update({'gradnorm_weights': gradnorm_weights,
                                 'gradnorm_weightloss1_list': weightloss1_list,
                                 'gradnorm_weightloss2_list': weightloss2_list,
                                 'obj1_minibatch_list': obj1_minibatch_list,
                                 'obj2_minibatch_list': obj2_minibatch_list
                                 })

            gradnorm_path = os.path.dirname(args.save_path) + '/taskid{}_weightloss_minibatch'.format(args.taskid)
            gradnorm_title = 'alpha: {}, epochs: {}, lr: {}'.format(args.alpha, args.gradnorm_epochs, args.gradnorm_lr)
            draw_diagnosis_plot(weightloss1_list, weightloss2_list, 'weight for obj1', 'weight for obj2', gradnorm_title, gradnorm_path)

            gradnorm_path = os.path.dirname(args.save_path) + '/taskid{}_obj_minibatch'.format(args.taskid)
            gradnorm_title = 'alpha: {}, epochs: {}, lr: {},\nobj1_train: {:.2f}, NN_train: {:.2f},\nasw_train: {:.2f}, nmi_train: {:.2f}, ari_train:{:.2f},\nuca_train: {:.2f}, be_train: {:.2f}'.format(args.alpha,
                            args.gradnorm_epochs, args.gradnorm_lr, obj1_train, obj2_train, asw_train, nmi_train, ari_train, uca_train, be_train)
            draw_diagnosis_plot(obj1_minibatch_list, obj2_minibatch_list, 'obj1 minibatch', 'obj2 minibatch', gradnorm_title, gradnorm_path)
        else:
            if args.std_paretoMTL == True:
                args.save_path = './result/pareto_front_paretoMTL/{}/{}/std_{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid)
                if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/std_{}taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid)):
                    os.makedirs('./result/pareto_front_paretoMTL/{}/{}/std_{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid))
            elif args.gradnorm_paretoMTL == True:
                args.save_path = './result/pareto_front_paretoMTL/{}/{}/gradnorm_{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator,args.taskid)
                if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/gradnorm_{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator,args.taskid)):
                    os.makedirs('./result/pareto_front_paretoMTL/{}/{}/gradnorm_{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator,args.taskid))

            if args.pref_idx == 0 or args.pref_idx==9:
                trainer_vae.train_set.show_t_sne(args.n_samples_tsne, color_by='batches', save_name=args.save_path + '/tsne_batch_train')
                trainer_vae.train_set.show_t_sne(args.n_samples_tsne, color_by='labels',save_name=args.save_path + '/tsne_labels_train')
                trainer_vae.test_set.show_t_sne(args.n_samples_tsne, color_by='batches', save_name=args.save_path + '/tsne_batch_test')
                trainer_vae.test_set.show_t_sne(args.n_samples_tsne, color_by='labels', save_name=args.save_path + '/tsne_labels_test')

        args_dict = vars(args)
        with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
            pickle.dump(args_dict, f)

        with open('{}/results.pkl'.format(args.save_path), 'wb') as f:
            pickle.dump(results_dict, f)
        print(results_dict)

# Run the actual program
if __name__ == "__main__":
    main()
