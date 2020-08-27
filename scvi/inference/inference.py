import copy
import matplotlib.pyplot as plt
import torch
from . import Trainer
import numpy as np
from torch.autograd import Variable
import pandas as pd
from scvi.models.modules import discrete_continuous_info, hsic

plt.switch_backend('agg')

class UnsupervisedTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.8, test_size=None, kl=None, n_epochs_kl_warmup=400, seed=0, **kwargs):
        super().__init__(model, gene_dataset, **kwargs)
        self.kl = kl
        self.seed = seed
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(model, gene_dataset, train_size, test_size, seed=self.seed)
            self.train_set.to_monitor = ['ll']
            self.test_set.to_monitor = ['ll']

    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        # for when model is vae
        reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index)

        #why self.kl_weight * kl_divergence here? because of kl_annealing in variational inference, check reference.
        #Why + here, not -, because the reconst_loss is -logp().
        neg_ELBO = torch.mean(reconst_loss + self.kl_weight*kl_divergence)

        loss = neg_ELBO
        if self.epoch % 10 == 0:
            print('Epoch: {}, mean_reconst_loss: {}, mean_kl_divergence_z: {}, neg_ELBO: {}'.format(
                self.epoch, torch.mean(reconst_loss), torch.mean(kl_divergence), neg_ELBO))
        return loss

    #If your applications rely on the posterior quality, (i.e.differential expression, batch effect removal), ensure
    #the number of total epochs( or iterations) exceed the number of epochs( or iterations) used for KL warmup
    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / self.n_epochs_kl_warmup)  # self.n_epochs)

    def two_loss(self, tensors):

        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors

        if self.cal_loss == True and self.cal_adv_loss == False:
            #for when model is vae_MI
            reconst_loss, kl_divergence, qz_m, qz_v, z = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
            loss = torch.mean(reconst_loss + self.kl_weight*kl_divergence)

        elif self.cal_loss == False and self.cal_adv_loss == True:
            x_ = sample_batch
            if self.model.log_variational:
                x_ = torch.log(1 + x_)
            qz_m, qz_v, z = self.model.z_encoder(x_, None)

            sample1, sample2, _, _ = self.adv_load_minibatch(z, batch_index)
            adv_loss, obj2_minibatch = self.adv_loss(sample1, sample2)

        elif self.cal_loss == True and self.cal_adv_loss == True:
            reconst_loss, kl_divergence, qz_m, qz_v, z = self.model(sample_batch, local_l_mean, local_l_var,batch_index)
            loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)

            sample1, sample2 = self.adv_load_minibatch(z, batch_index)
            adv_loss, obj2_minibatch = self.adv_loss(sample1, sample2)

            if self.epoch % 10 == 0 and self.cal_loss == True:

                NN_estimator = discrete_continuous_info(torch.transpose(batch_index, 0, 1), torch.transpose(z, 0, 1))
                print('obj2:')
                print(obj2_minibatch)
                print('Epoch: {}, neg_ELBO: {}, {}: {}, NN: {}.'.format(self.epoch, loss, self.adv_estimator, obj2_minibatch, NN_estimator))

        #objective 1 equals loss
        if self.cal_loss == True and self.cal_adv_loss == False:
            return loss, None, None
        elif self.cal_loss == False and self.cal_adv_loss == True:
            return None, adv_loss, obj2_minibatch
        elif self.cal_loss == True and self.cal_adv_loss == True:
            return loss, adv_loss, obj2_minibatch

    def adv_load_minibatch(self, z, batch_index):

        batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.numpy().ravel())})
        batch_dummy = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)

        #check if z.requires_grad == True
        if self.use_cuda == True:
            batch_dummy = batch_dummy.cuda()
        batch_dummy = Variable(batch_dummy, requires_grad=True)

        if self.adv_estimator == 'MINE':
            z_batch = torch.cat((z, batch_dummy), 1)  # joint
            shuffle_index = torch.randperm(z.shape[0])
            shuffle_z_batch = torch.cat((z[shuffle_index], batch_dummy), 1)  # marginal
            return z_batch, shuffle_z_batch
        elif self.adv_estimator == 'HSIC':
            if self.use_cuda == True:
                batch_index = batch_index.type(torch.FloatTensor).cuda()
            batch_index = Variable(batch_index.type(torch.FloatTensor), requires_grad=True)
            return z, batch_index
        elif self.adv_estimator == 'MMD':
            if self.use_cuda == True:
                batch_index = batch_index.type(torch.FloatTensor).cuda()
            batch_index = Variable(batch_index.type(torch.FloatTensor), requires_grad=True)
            z_batch0 = z[(batch_index[:, 0] == 0).nonzero().squeeze(1)]
            z_batch1 = z[(batch_index[:, 0] == 1).nonzero().squeeze(1)]
            return z_batch0, z_batch1

    def adv_loss(self, sample1, sample2):
        if self.adv_estimator == 'MINE':
            t = self.adv_model(sample1)
            et = torch.exp(self.adv_model(sample2))

            if self.adv_model.unbiased_loss:
                if self.adv_model.ma_et is None:
                    self.adv_model.ma_et = torch.mean(et).detach().item()  # detach means will not calculate gradient for ma_et, ma_et is just a number
                self.adv_model.ma_et = (1 - self.adv_model.ma_rate) * self.adv_model.ma_et + self.adv_model.ma_rate * torch.mean(et).detach().item()

                # Pay attention, this unbiased loss is not our MINE estimator,
                # The MINE estimator is still torch.mean(t) - torch.log(torch.mean(et)) after training
                # The unbiased_loss is only for getting unbiased gradient.
                loss = -(torch.mean(t) - (1 / self.adv_model.ma_et) * torch.mean(et))
            else:
                loss = -(torch.mean(t) - torch.log(torch.mean(et)))

            MINE_estimator_minibatch = torch.mean(t) - torch.log(torch.mean(et))

            return loss, MINE_estimator_minibatch
        elif self.adv_estimator == 'HSIC':
            hsic_minibatch = hsic(sample1, sample2)
            loss = hsic_minibatch
            return loss, hsic_minibatch
        elif self.adv_estimator == 'MMD':
            mmd_minibatch = self.MMD_loss(sample1, sample2)
            loss = mmd_minibatch
            return loss, mmd_minibatch

class AdapterTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, posterior_test, frequency=5):
        super().__init__(model, gene_dataset, frequency=frequency)
        self.test_set = posterior_test
        self.test_set.to_monitor = ['ll']
        self.params = list(self.model.z_encoder.parameters()) + list(self.model.l_encoder.parameters())
        self.z_encoder_state = copy.deepcopy(model.z_encoder.state_dict())
        self.l_encoder_state = copy.deepcopy(model.l_encoder.state_dict())

    @property
    def posteriors_loop(self):
        return ['test_set']

    def train(self, n_path=10, n_epochs=50, **kwargs):
        for i in range(n_path):
            # Re-initialize to create new path
            self.model.z_encoder.load_state_dict(self.z_encoder_state)
            self.model.l_encoder.load_state_dict(self.l_encoder_state)
            super().train(n_epochs, params=self.params, **kwargs)

        return min(self.history["ll_test_set"])
