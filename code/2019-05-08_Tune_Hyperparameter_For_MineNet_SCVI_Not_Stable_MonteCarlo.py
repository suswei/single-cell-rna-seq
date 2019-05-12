#SCVI or SCVI+MINE runs very slow on the Pbmc dataset, when the train_size=0.6,
#for the monte carlo(B=100), if all the 20 hyperparameter combinations in the
# 2019-05-01_Tune_Hyperparameter_For_MineNet.py file are run, 12 days will be needed
#for 21 cups to finish the whole task. Therefore only one combination (n_latent_z: 30, n_layers_z:3,
# MineLoss_Scale: 1000) is chosen to compare with original SCVI for Monte Carlo to see whether
#SCVI+Mine does work better than SCVI
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scvi.dataset import *
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch


def main(taskid):
    if not os.path.exists('.\\data\\2019-05-08'):
        os.makedirs('.\\data\\2019-05-08')
    if not os.path.exists('.\\result\\2019-05-08'):
        os.makedirs('.\\result\\2019-05-08')

    save_path = '.\\data\\2019-05-08\\'
    n_hidden_z = 30
    n_layers_z = 3
    MineLoss_Scale = 1000

    pbmc_dataset = PbmcDataset(save_path=save_path)

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100),dtype=np.uint32)

    taskid = int(taskid[0])
    desired_seed = desired_seeds[taskid]

    n_epochs_all = None
    show_plot = True
    n_epochs = 400 if n_epochs_all is None else n_epochs_all
    lr = 0.0005
    use_batches = True
    use_cuda = False
    train_size = 0.6

    vae_mine = VAE_MINE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches, n_hidden_z=n_hidden_z, n_layers_z=n_layers_z, MineLoss_Scale=MineLoss_Scale)
    trainer_vae_mine = UnsupervisedTrainer(vae_mine, pbmc_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda,frequency=5, kl=1)
    vae_mine_file_path = '%s\\Pbmc_Sample%s_Hidden%s_layers%s_MineLossScale%s_VaeMine.pk1'%(save_path, taskid, n_hidden_z, n_layers_z, MineLoss_Scale)

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

    fig1_path = '.\\result\\2019-05-08\\2019-05-08_training_testing_error_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}.png'.format('Pbmc',taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    fig.savefig(fig1_path)
    plt.close(fig)

    vae = VAE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches)
    trainer_vae = UnsupervisedTrainer(vae, pbmc_dataset, train_size=train_size, seed=desired_seed, use_cuda=use_cuda, frequency=5)
    vae_file_path = '%s\\Pbmc_Sample%s_Hidden%s_layers%s_MineLossScale%s_Vae.pk1'%(save_path, taskid, n_hidden_z, n_layers_z, MineLoss_Scale)

    if os.path.isfile(vae_file_path):
        trainer_vae.model.load_state_dict(torch.load(vae_file_path))
        trainer_vae.model.eval()
    else:
        trainer_vae.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer_vae.model.state_dict(), vae_file_path)

    ll_train_set = trainer_vae.history["ll_train_set"]
    ll_test_set = trainer_vaee.history["ll_test_set"]
    x = np.linspace(0, 500, (len(ll_train_set)))

    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, ll_train_set)
    plt.plot(x, ll_test_set)
    plt.ylim(1150, 1600)
    plt.title("Blue for training error and orange for testing error")

    fig2_path = '.\\result\\2019-05-08\\2019-05-08_training_testing_error_SCVI_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}.png'.format('Pbmc', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    fig.savefig(fig2_path)
    plt.close(fig)

    n_samples_tsne = 1000

    trainer_vae_mine.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-08\\trainset_tsne_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Pbmc',taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    plt.show()
    trainer_vae.train_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-08\\trainset_tsne_SCVI_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Pbmc', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    plt.show()

    trainer_vae_mine.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-08\\testset_tsne_SCVI+MINE_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Pbmc', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    plt.show()
    trainer_vae.test_set.show_t_sne(n_samples_tsne, color_by='batches and labels', save_name='.\\result\\2019-05-08\\testset_tsne_SCVI_{}_Sample{}_Hidden{}_layers{}_MineLossScale{}'.format('Pbmc', taskid, n_hidden_z, n_layers_z, MineLoss_Scale))
    plt.show()

    # clustering_scores() -- these metrics measure clustering performance
    #   silhouette width (asw, higher is better),
    #   normalised mutual information (nmi, higher is better),
    #   adjusted rand index (ari, higher is better),
    #   unsupervised clustering accuracy (uca)
    #   entropy_batch_mixing() -- this metric measures batch effect
    #   entropy batch mixing (be, higher is better meaning less batch effect)

    clustering_metric = np.empty((0, 5), int)
    asw, nmi, ari, uca = trainer_vae.train_set.clustering_scores()
    be = trainer_vae.train_set.entropy_batch_mixing()
    clustering_metric = np.append(clustering_metric, np.array([[asw, nmi, ari, uca, be]]),axis=0)

    asw, nmi, ari, uca = trainer_vae_mine.train_set.clustering_scores()
    be = trainer_vae_mine.train_set.entropy_batch_mixing()
    clustering_metric = np.append(clustering_metric, np.array([[asw, nmi, ari, uca, be]]),axis=0)

    asw, nmi, ari, uca = trainer_vae.test_set.clustering_scores()
    be = trainer_vae.test_set.entropy_batch_mixing()
    clustering_metric = np.append(clustering_metric, np.array([[asw, nmi, ari, uca, be]]),axis=0)

    asw, nmi, ari, uca = trainer_vae_mine.test_set.clustering_scores()
    be = trainer_vae_mine.test_set.entropy_batch_mixing()
    clustering_metric = np.append(clustering_metric, np.array([[asw, nmi, ari, uca, be]]),axis=0)

    clustering_metric = pd.DataFrame(clustering_metric)
    Label = {'Label':['vae_train', 'vaemine_train', 'vae_test', 'vaemine_test']}
    Label_DataFrame = pd.DataFrame(Label,columns=['Label'])

    clustering_metric2 = pd.concat([Label_DataFrame, clustering_metric.reset_index(drop=True)],axis=1)
    clustering_metric2.to_csv('.\\result\\2019-05-08\\Pbmc_Sample%s_Hidden%s_layers%s_MineLossScale%s_ClusterMetric.csv'%(taskid, n_hidden_z, n_layers_z, MineLoss_Scale), index=None, header=True)


# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1:])

# In terminal type
# python hypertuning.py taskid
# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
