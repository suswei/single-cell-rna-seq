
import numpy as np

from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.muris_tabula import TabulaMuris
from scvi.models import *
from scvi.inference import UnsupervisedTrainer


data_save_path = './data/pareto_front_scVI_MINE/%s' % ('muris_tabula')
dataset1 = TabulaMuris('facs', save_path=data_save_path)
dataset2 = TabulaMuris('droplet', save_path=data_save_path)
dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)


np.random.seed(1011)
desired_seeds = np.random.randint(0, 2 ** 32, size=(1, 100), dtype=np.uint32)
desired_seed = int(desired_seeds[0, 1])

batch1_ratio = gene_dataset.batch_indices[gene_dataset.batch_indices[:,0]==1].shape[0]/gene_dataset.batch_indices.shape[0]
#calculate ratio to split the gene_dataset into training and testing dataset
# to avoid the case when there are very few input data points of the last minibatch of every epoch
intended_trainset_size=int(gene_dataset._X.shape[0]/128/10)*10*0.8*128 + (int(gene_dataset._X.shape[0]/128) % 10)*128
train_size = int(intended_trainset_size/gene_dataset._X.shape[0]*1e6)/1e6

vae_MI = VAE_MI(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * True, n_labels=gene_dataset.n_labels,
              n_hidden=128, n_latent=10, n_layers_encoder=2, n_layers_decoder=2, dropout_rate=0.1,
              reconstruction_loss='zinb')
trainer_vae = UnsupervisedTrainer(vae_MI, gene_dataset, batch_size=128, train_size=train_size, seed=desired_seed, use_cuda=False, frequency=10, kl=1,
                adv = True, adv_estimator = 'MINE_MI', adv_n_hidden = 128, adv_n_layers = 10, adv_activation_fun = 'ELU',
                 unbiased_loss = True, adv_w_initial = 'normal', aggregated_posterior = False,
                 batch1_ratio=batch1_ratio, nsamples_z=128)
#TODO: it is better to be controled by self.on_epoch_begin(), it should be modified later
trainer_vae.kl_weight=1

#The strategy is to train MINE for each minibatch in each epoch in the main training,
#but the MINE adversary training runs only for one minibatch, not the whole training dataset
#obj2_type is 'MINE_estimator', not negative of the unibased loss
loss_minibatch_list, neg_adv_lost_list, MINE_estimator_minibatch_list = trainer_vae.adversarial_train(pre_n_epochs=50, pre_lr=1e-3, pre_adv_epochs=100,
                                            adv_lr=5e-5, n_epochs=201, main_lr=1e-3, std=False, scale=0)
#the min_value, max_value for obj1 (neg_ELBO) is

loss_minibatch_list, neg_adv_lost_list, MINE_estimator_minibatch_list = trainer_vae.adversarial_train(pre_n_epochs=50, pre_lr=1e-3, pre_adv_epochs=100,
                                            adv_lr=5e-5, n_epochs=201, main_lr=1e-3, std=False, scale=1)
#the min_value, max_value for obj2 (MINE) is

loss_minibatch_list, neg_adv_lost_list, MINE_estimator_minibatch_list = trainer_vae.adversarial_train(pre_n_epochs=50, pre_lr=1e-3, pre_adv_epochs=100,
                                            adv_lr=5e-5, n_epochs=201, main_lr=1e-3, std=True,
                                            min_obj1=20000, max_obj1=12000, scale=0)
