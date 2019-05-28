#Run 100 Monte Carlo Samples for 20 hyperparameter configurations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scvi.dataset import *
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch
import itertools


def main(taskid):
    if not os.path.exists('data/Tune_Hyperparameter_For_Minenet/2019-05-28'):
        os.makedirs('data/Tune_Hyperparameter_For_Minenet/2019-05-28')
    if not os.path.exists('result/Tune_Hyperparameter_For_Minenet/2019-05-28'):
        os.makedirs('result/Tune_Hyperparameter_For_Minenet/2019-05-28')

    save_path = 'data/Tune_Hyperparameter_For_Minenet/2019-05-28'

    hyperparameter_config = {
        'n_hidden_z': [10,30],
        'n_layers_z': [3,10],
        'MineLoss_Scale': [1000,5000,10000,50000,100000]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    #pbmc_dataset = PbmcDataset(save_path=save_path)
    pbmc_dataset = RetinaDataset(save_path=save_path) #here pbmc_dataset is actually RetinaDataset

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100),dtype=np.uint32)

    taskid = int(taskid[0])
    desired_seed = int(desired_seeds[0,taskid])

    n_epochs_all = None
    show_plot = True
    n_epochs = 400 if n_epochs_all is None else n_epochs_all
    lr = 0.0005
    use_batches = True
    use_cuda = False
    train_size = 0.6

    n_samples_tsne = 1000
    clustering_metric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be'])

    for i in range(len(hyperparameter_experiments)):
        key, value = zip(*hyperparameter_experiments[i].items())
        n_hidden_z = value[0]
        n_layers_z = value[1]
        MineLoss_Scale = value[2]

        vae_mine = VAE_MINE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches, n_hidden_z=n_hidden_z, n_layers_z=n_layers_z, MineLoss_Scale=MineLoss_Scale)
        trainer_vae_mine = UnsupervisedTrainer(vae_mine, pbmc_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda,frequency=5, kl=1)
        vae_mine_file_path = '%s/Retina_Sample%s_Hidden%s_layers%s_MineLossScale%s_VaeMine.pk1'%(save_path, taskid, n_hidden_z, n_layers_z, MineLoss_Scale)

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

        fig1_path = 'result/Tune_Hyperparameter_For_Minenet/2019-05-28/training_testing_error_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}.png'.format('Retina',taskid, n_hidden_z, n_layers_z, MineLoss_Scale)
        fig.savefig(fig1_path)
        plt.close(fig)

        trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels',save_name='result/Tune_Hyperparameter_For_Minenet/2019-05-28/trainset_tsne_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Retina', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
        trainer_vae_mine.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/Tune_Hyperparameter_For_Minenet/2019-05-28/testset_tsne_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Retina', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))

        asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
        be = trainer_vae_mine.train_set.entropy_batch_mixing()
        label = 'sample%s_VaeMine_trainset_Hidden%s_layers%s_MineLossScale%s'%(taskid, n_hidden_z,n_layers_z,MineLoss_Scale)
        intermediate_dataframe1 = pd.DataFrame.from_dict({'Label':[label],'asw':[asw],'nmi':[nmi],'ari':[ari],'uca':[uca],'be':[be]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

        asw, nmi, ari, uca = trainer_vae_mine.test_set.clustering_scores()
        be = trainer_vae_mine.test_set.entropy_batch_mixing()
        label = 'sample%s_VaeMine_testset_Hidden%s_layers%s_MineLossScale%s' % (taskid, n_hidden_z, n_layers_z, MineLoss_Scale)
        intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
        clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)

    vae = VAE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches)
    trainer_vae = UnsupervisedTrainer(vae, pbmc_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5)
    vae_file_path = '%s/Retina_Sample%s_Vae.pk1'%(save_path, taskid)

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

    fig2_path = 'result/Tune_Hyperparameter_For_Minenet/2019-05-28/training_testing_error_SCVI_{}_Sample{}.png'.format('Retina', taskid)
    fig.savefig(fig2_path)
    plt.close(fig)

    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/Tune_Hyperparameter_For_Minenet/2019-05-28/trainset_tsne_SCVI_{}_Sample{}'.format('Retina', taskid))
    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='result/Tune_Hyperparameter_For_Minenet/2019-05-28/testset_tsne_SCVI_{}_Sample{}'.format('Retina', taskid))

    # clustering_scores() -- these metrics measure clustering performance
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    #   entropy_batch_mixing() -- this metric measures batch effect
    #   entropy batch mixing (be, higher is better meaning less batch effect)


    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()
    label = 'sample%s_Vae_trainset' % (taskid)
    intermediate_dataframe1 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe1], axis=0)

    asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
    be = trainer_vae.test_set.entropy_batch_mixing()
    label = 'sample%s_Vae_testset' % (taskid)
    intermediate_dataframe2 = pd.DataFrame.from_dict({'Label': [label], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
    clustering_metric = pd.concat([clustering_metric, intermediate_dataframe2], axis=0)

    clustering_metric.to_csv('result/Tune_Hyperparameter_For_Minenet/2019-05-26/Retina_Sample%s_ClusterMetric.csv'%(taskid), index=None, header=True)


# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1:])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
