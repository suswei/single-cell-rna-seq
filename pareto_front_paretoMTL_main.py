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
from scvi.dataset.pbmc_scp256_scp548 import Pbmc_SCP256_SCP548
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
from scvi.models.modules import MINE_Net, Nearest_Neighbor_Estimate, MMD_loss
from scipy import sparse
import pickle
import matplotlib.pyplot as plt

def sample1_sample2(trainer_vae, sample_batch, batch_index, obj2_type, reference_batch: int=0, compare_batch:int=1):
    x_ = sample_batch
    if trainer_vae.model.log_variational:
        x_ = torch.log(1 + x_)
    # Sampling
    qz_m, qz_v, z = trainer_vae.model.z_encoder(x_, None)

    batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.numpy().ravel())})
    batch_dummy = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)
    batch_dummy = Variable(batch_dummy, requires_grad=True)

    if obj2_type == 'MINE':
        sample1 = torch.cat((z, batch_dummy), 1)  # joint
        shuffle_index = torch.randperm(z.shape[0])
        sample2 = torch.cat((z[shuffle_index], batch_dummy), 1)
        return sample1, sample2, z, batch_dummy
    elif obj2_type in ['MMD', 'stdz_MMD']:
        if obj2_type == 'stdz_MMD':
            # standardize each dimension for z
            z_mean = torch.mean(z, 0).unsqueeze(0).expand(int(z.size(0)), int(z.size(1)))
            z_std = torch.std(z, 0).unsqueeze(0).expand(int(z.size(0)), int(z.size(1)))
            z = (z - z_mean) / z_std  # element by element
        z_reference_batch = z[(batch_index[:, 0] == reference_batch).nonzero().squeeze(1)]
        z_compare_batch = z[(batch_index[:, 0] == compare_batch).nonzero().squeeze(1)]
        return z_reference_batch, z_compare_batch, z, batch_dummy
    elif obj2_type == 'NN':
        return None, None, z, None

def sample1_sample2_all(trainer_vae, input_data, obj2_type, reference_batch: int=0, compare_batch:int=1):
    z_all = torch.empty(0, trainer_vae.model.n_latent)
    batch_dummy_all = torch.empty(0, trainer_vae.model.n_batch)
    batch_index_all = torch.empty(0, 1).type(torch.LongTensor)
    z_reference_all = torch.empty(0, trainer_vae.model.n_latent)
    z_compare_all = torch.empty(0, trainer_vae.model.n_latent)

    for tensors_list in input_data:
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list

        sample1, sample2, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, obj2_type, reference_batch, compare_batch)
        if obj2_type == 'MINE':
            z_all = torch.cat((z_all, z), 0)
            batch_dummy_all = torch.cat((batch_dummy_all, batch_dummy), 0)
        elif obj2_type in ['MMD','stdz_MMD']:
            z_reference_all = torch.cat((z_reference_all, sample1),0)
            z_compare_all = torch.cat((z_compare_all, sample2),0)
        elif obj2_type == 'NN':
            z_all = torch.cat((z_all, z), 0)
            batch_index_all = torch.cat((batch_index_all, batch_index), 0)

    if obj2_type == 'MINE':
        return z_all, batch_dummy_all
    elif obj2_type in ['MMD','stdz_MMD']:
        return z_reference_all, z_compare_all
    elif obj2_type == 'NN':
        return z_all, batch_index_all

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

def MINE_eval(trainer_vae, MINE_network, input_data):

    z_all, batch_dummy_all = sample1_sample2_all(trainer_vae, input_data, 'MINE')

    sample1_all = torch.cat((z_all, batch_dummy_all), 1)  # joint
    shuffle_index = torch.randperm(z_all.shape[0])
    sample2_all = torch.cat((z_all[shuffle_index], batch_dummy_all), 1)
    t_all = MINE_network(sample1_all)
    et_all = torch.exp(MINE_network(sample2_all))
    MINE_estimator = torch.mean(t_all) - torch.log(torch.mean(et_all))
    return MINE_estimator.item()

def MINE_after_trainerVae(trainer_vae):
    MINE_network = MINE_Net(input_dim=trainer_vae.model.n_latent + trainer_vae.model.n_batch, n_hidden=128, n_layers=10,
                            activation_fun='ELU', unbiased_loss=True, initial='normal')

    MINE_optimizer = optim.Adam(MINE_network.parameters(), lr=5e-5)

    for epoch in range(400):
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

            loss.backward()
            MINE_optimizer.step()
            MINE_optimizer.zero_grad()
            trainer_vae.optimizer.zero_grad()

    with torch.no_grad():
        MINE_network.eval()
        MINE_estimator_train = MINE_eval(trainer_vae, MINE_network, trainer_vae.train_set)
        MINE_estimator_test = MINE_eval(trainer_vae, MINE_network, trainer_vae.test_set)

    print('MINE MI train: {}, MINE MI test: {}'.format(MINE_estimator_train, MINE_estimator_test))

    return MINE_estimator_train, MINE_estimator_test

def MMD_train_test(z_reference, z_compare, MMD_kernel_mul, MMD_kernel_num):

    MMD_loss_fun = MMD_loss(MMD_kernel_mul, MMD_kernel_num)

    if z_reference.shape[0]<=2000:
        z_reference_subset = z_reference
    else:
        z_reference_subset = z_reference[0:2000,:]

    if z_compare.shape[0]<=2000:
        z_compare_subset = z_compare
    else:
        z_compare_subset = z_compare[0:2000,:]

    estimator = MMD_loss_fun(z_reference_subset, z_compare_subset)
    return estimator.item()

def MMD_NN_train_test(trainer_vae, obj2_type, args):

    if obj2_type in ['MMD','stdz_MMD']:
        reference_batch = 0
        MMD_loss_train, MMD_loss_test = [], []
        for i in range(trainer_vae.model.n_batch-1):
            compare_batch = i + 1
            z_reference_train, z_compare_train = sample1_sample2_all(trainer_vae, trainer_vae.train_set, obj2_type, reference_batch, compare_batch)
            MMD_loss_train += [MMD_train_test(z_reference_train, z_compare_train, args.MMD_kernel_mul, args.MMD_kernel_num)]

            z_reference_test, z_compare_test = sample1_sample2_all(trainer_vae, trainer_vae.test_set, obj2_type, reference_batch, compare_batch)
            MMD_loss_test += [MMD_train_test(z_reference_test, z_compare_test, args.MMD_kernel_mul, args.MMD_kernel_num)]

        estimator_train = max(MMD_loss_train)
        estimator_test = max(MMD_loss_test)
    elif obj2_type == 'NN':
        NN_train_list, NN_test_list = [], []
        for tensors_list in trainer_vae.train_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
            z_batch0, z_batch1, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, obj2_type)
            NN_minibatch_train = Nearest_Neighbor_Estimate(batch_index, z)
            NN_train_list.append(NN_minibatch_train)
        estimator_train = sum(NN_train_list) / len(NN_train_list)
        for tensors_list in trainer_vae.test_set:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list
            z_batch0, z_batch1, z, batch_dummy = sample1_sample2(trainer_vae, sample_batch, batch_index, obj2_type)
            NN_minibatch_test = Nearest_Neighbor_Estimate(batch_index, z)
            NN_test_list.append(NN_minibatch_test)
        estimator_test = sum(NN_test_list)/len(NN_test_list)

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

    parser.add_argument('--change_composition', action='store_true', default=True,
                        help='whether to change the cell type composition in the original dataset')

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

    #for empirical MI
    parser.add_argument('--empirical_MI', action='store_true', default=False,
                        help='whether to calculate empirical MI during training')

    parser.add_argument('--batch_ratio', type=list, default=[],
                        help='the list showing the percent of each batch in the dataset')

    parser.add_argument('--nsamples', type=int, default=1000,
                        help='number of samples from aggregated posterior to get empirical MI')

    #for MMD
    parser.add_argument('--MMD_kernel_mul', type=float, default=2.0,
                        help='the multiplier value to calculate bandwidth')

    parser.add_argument('--MMD_kernel_num', type=int, default=5,
                        help='the number of kernels to get MMD')

    #for pre_train
    parser.add_argument('--pre_train', action='store_true', default=False,
                        help='whether to pre train neural network')

    parser.add_argument('--pre_epochs', type=int, default=200,
                        help='number of epochs to pre-train scVI')

    parser.add_argument('--pre_adv_epochs', type=int, default=400,
                        help='number of epochs to pre-train MINE')

    parser.add_argument('--pre_lr', type=float, default=1e-3,
                        help='learning rate in scVI pre-training')

    parser.add_argument('--pre_adv_lr', type=float, default=5e-5,
                        help='learning rate in MINE pre-training and adversarial training')

    #to get min and max for obj1 and obj2 to standardize
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='whether to get min and max value for obj1 and obj2 to standardize')

    #for paretoMTL

    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train scVI and MINE')

    parser.add_argument('--adv_epochs', type=int, default=1,
                        help='number of epochs to train MINE adversarially')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for paretoMTL')

    parser.add_argument('--adv_lr', type=float, default=5e-5,
                        help='learning rate in MINE pre-training and adversarial training')

    parser.add_argument('--n_tasks', type=int, default=2,
                        help='number of objectives for the multiple optimization problem')

    parser.add_argument('--npref', type=int, default=10,
                        help='number of subproblems when the multiple optimization problem is decomposed')

    parser.add_argument('--pref_idx', type=int, default=0,
                        help='which subproblem')

    parser.add_argument('--std_paretoMTL', action='store_true', default=False,
                        help='whether to standardize the two objectives')

    parser.add_argument('--obj1_max', type=float, default=20000,
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

    #for evaluation
    parser.add_argument('--eval_samplesize', type=int, default=3000,
                        help='sample size to get NN estimator and MMD estimator at evaluation stage')

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
        if args.change_composition == True:
            dataset1_labels = dataset1.__dict__['labels']
            dataset1_labels_df = pd.DataFrame.from_dict({'label': dataset1_labels[:, 0].tolist()})
            dataset1_celltypes = dataset1.__dict__['cell_types']
            dataset1_celltypes_df = pd.DataFrame.from_dict({'cell_type': dataset1_celltypes.tolist()})
            dataset1_celltypes_df['label'] = pd.Series(np.array(list(range(dataset1_celltypes_df.shape[0]))),index=dataset1_celltypes_df.index)

            delete_labels_celltypes = dataset1_celltypes_df[dataset1_celltypes_df.cell_type.isin(['granulocyte', 'nan', 'monocyte', 'hematopoietic precursor cell', 'granulocytopoietic cell'])]
            dataset1.__dict__['_X'] = sparse.csr_matrix(dataset1.__dict__['_X'].toarray()[~dataset1_labels_df.label.isin(delete_labels_celltypes.loc[:, 'label'].values.tolist())])
            for key in ['local_means', 'local_vars', 'batch_indices', 'labels']:
                dataset1.__dict__[key] = dataset1.__dict__[key][~dataset1_labels_df.label.isin(delete_labels_celltypes.loc[:,'label'].values.tolist())]
            dataset1.__dict__['n_labels'] = 18
            dataset1.__dict__['cell_types'] = np.delete(dataset1.__dict__['cell_types'], delete_labels_celltypes.loc[:, 'label'].values.tolist())

            dataset1_celltypes_new = dataset1.__dict__['cell_types']
            dataset1_celltypes_df_new = pd.DataFrame.from_dict({'cell_type': dataset1_celltypes_new.tolist()})
            dataset1_celltypes_df_new['label'] = pd.Series(np.array(list(range(dataset1_celltypes_df_new.shape[0]))),index=dataset1_celltypes_df_new.index)
            label_change = dataset1_celltypes_df.merge(dataset1_celltypes_df_new, how='right', left_on='cell_type', right_on='cell_type').iloc[:, 1:]
            label_change.columns = ['label','new_label']

            dataset1_labels_new = dataset1.__dict__['labels']
            dataset1_labels_df_new = pd.DataFrame.from_dict({'label': dataset1_labels_new[:, 0].tolist()})

            dataset1.__dict__['labels'] = dataset1_labels_df_new.merge(label_change, how='left', left_on='label', right_on='label').loc[:,'new_label'].values.reshape(dataset1.__dict__['labels'].shape[0],1)

        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
    elif args.dataset_name == 'pbmc_scp256_scp548':
        dataset1 = Pbmc_SCP256_SCP548('SCP256', save_path=args.data_save_path)
        dataset2 = Pbmc_SCP256_SCP548('SCP548', save_path=args.data_save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

    #generate a random seed to split training and testing dataset
    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, args.MCs), dtype=np.uint32)
    if args.pre_train == True:
        desired_seed = int(desired_seeds[0, args.taskid])
    else:
        desired_seed = int(desired_seeds[0, int(args.taskid/args.npref)])

    if args.pre_train == True:
        args.save_path = './result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid)
        if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid)):
            os.makedirs('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, args.taskid))
    else:
        args.save_path = './result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, int(args.taskid/args.npref))
        if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/pre_train/MC{}'.format(args.dataset_name, args.confounder, int(args.taskid/args.npref))):
            print('Error: please pretrain first!')

    if args.empirical_MI == True:
        for i in range(gene_dataset.n_batches-1):
            ratio_perbatch = gene_dataset.batch_indices[gene_dataset.batch_indices[:,0]==i].shape[0]/gene_dataset.batch_indices.shape[0]
            args.batch_ratio +=[ratio_perbatch]
        args.batch_ratio += [1-sum(args.batch_ratio)]

    # calculate ratio to split the gene_dataset into training and testing dataset
    # to avoid the case when there are very few input data points of the last minibatch in every epoch
    intended_trainset_size = int(gene_dataset._X.shape[0] / args.batch_size / 10) * 10 * args.train_size * args.batch_size + (int(gene_dataset._X.shape[0] / args.batch_size) % 10) * 128
    args.train_size = int(intended_trainset_size / gene_dataset._X.shape[0] * 1e10) / 1e10

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

    vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * args.use_batches,n_labels=gene_dataset.n_labels,
                    n_hidden=args.n_hidden, n_latent=args.n_latent,n_layers_encoder=args.n_layers_encoder,
                    n_layers_decoder=args.n_layers_decoder,dropout_rate=args.dropout_rate, reconstruction_loss=args.reconstruction_loss)
    if args.adv_estimator == 'MINE':
        trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size,
                                          seed=desired_seed, use_cuda=args.use_cuda, frequency=10, kl=1, adv_estimator=args.adv_estimator,
                                          adv_n_hidden=args.adv_n_hidden, adv_n_layers=args.adv_n_layers, adv_activation_fun=args.adv_activation_fun,
                                          unbiased_loss=args.unbiased_loss, adv_w_initial=args.adv_w_initial, batch_ratio=args.batch_ratio, nsamples=args.nsamples)
    elif args.adv_estimator in ['MMD','stdz_MMD']:
        trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=args.batch_size, train_size=args.train_size,
                                          seed=desired_seed, use_cuda=args.use_cuda, frequency=10, kl=1, adv_estimator=args.adv_estimator,
                                          MMD_kernel_mul=args.MMD_kernel_mul, MMD_kernel_num=args.MMD_kernel_num, batch_ratio=args.batch_ratio, nsamples=args.nsamples)

    # TODO: it is better to be controled by self.on_epoch_begin(), it should be modified later
    trainer_vae.kl_weight = 1

    if args.pre_train == True:
        trainer_vae.pretrain_paretoMTL(pre_train=args.pre_train, pre_epochs=args.pre_epochs, pre_lr=args.pre_lr,
                        pre_adv_epochs=args.pre_adv_epochs, pre_adv_lr=args.pre_adv_lr, path=args.save_path)
    elif args.standardize == True:
        obj1_minibatch_list, obj2_minibatch_list = trainer_vae.pretrain_paretoMTL(
            path=args.save_path, standardize=args.standardize, lr=args.lr, adv_lr=args.adv_lr, epochs=args.epochs,
            adv_epochs=args.adv_epochs, n_tasks=args.n_tasks, npref=args.npref, pref_idx=args.pref_idx)

        obj1_max = max(obj1_minibatch_list)
        obj1_min = min(obj1_minibatch_list)
        print('obj1_max: {}, obj1_min: {}'.format(obj1_max, obj1_min))
        obj2_max = max(obj2_minibatch_list)
        obj2_min = min(obj2_minibatch_list)
        print('obj2_max: {}, obj2_min: {}'.format(obj2_max, obj2_min))

    elif args.std_paretoMTL == True:
        _, _ = trainer_vae.pretrain_paretoMTL(path=args.save_path, lr=args.lr, adv_lr=args.adv_lr, std_paretoMTL=args.std_paretoMTL,
               obj1_max=args.obj1_max, obj1_min=args.obj1_min, obj2_max=args.obj2_max, obj2_min=args.obj2_min, epochs = args.epochs,
               adv_epochs=args.adv_epochs, n_tasks = args.n_tasks, npref = args.npref, pref_idx = args.pref_idx, taskid=args.taskid)

        #obj1 for the whole training and testing set
        obj1_train, obj1_test = obj1_train_test(trainer_vae)

        # obj2 for the whole training and testing set
        if trainer_vae.adv_estimator == 'MINE':
            obj2_train, obj2_test = MINE_after_trainerVae(trainer_vae)
        elif trainer_vae.adv_estimator == 'MMD':
            obj2_train, obj2_test = MMD_NN_train_test(trainer_vae, 'MMD', args)
        elif trainer_vae.adv_estimator == 'stdz_MMD':
            obj2_train, obj2_test = MMD_NN_train_test(trainer_vae, 'stdz_MMD', args)

        NN_train, NN_test = MMD_NN_train_test(trainer_vae, 'NN', args)

        asw_train, nmi_train, ari_train, uca_train = trainer_vae.train_set.clustering_scores()
        be_train = trainer_vae.train_set.entropy_batch_mixing()

        asw_test, nmi_test, ari_test, uca_test = trainer_vae.test_set.clustering_scores()
        be_test = trainer_vae.test_set.entropy_batch_mixing()

        results_dict = {'obj1_train': [obj1_train],
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
                        'be_test': [be_test]}

        args.save_path = './result/pareto_front_paretoMTL/{}/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid)
        if not os.path.exists('./result/pareto_front_paretoMTL/{}/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid)):
            os.makedirs('./result/pareto_front_paretoMTL/{}/{}/{}/taskid{}'.format(args.dataset_name, args.confounder, args.adv_estimator, args.taskid))

        if args.pref_idx == 0 or args.pref_idx==9:
            trainer_vae.train_set.show_t_sne(args.n_samples_tsne, color_by='batches and labels', save_name=args.save_path + '/tsne_batch_label_train')
            trainer_vae.test_set.show_t_sne(args.n_samples_tsne, color_by='batches and labels', save_name=args.save_path + '/tsne_batch_label_test')

        args_dict = vars(args)
        with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
            pickle.dump(args_dict, f)

        with open('{}/results.pkl'.format(args.save_path), 'wb') as f:
            pickle.dump(results_dict, f)
        print(results_dict)

# Run the actual program
if __name__ == "__main__":
    main()
