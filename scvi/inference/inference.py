import copy
import matplotlib.pyplot as plt
import torch
from . import Trainer
import numpy as np
from torch.autograd import Variable
import pandas as pd
from scvi.models.modules import Nearest_Neighbor_Estimate, EmpiricalMI_From_Aggregated_Posterior

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
    #the number of total epochs(or iterations) exceed the number of epochs( or iterations) used for KL warmup
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
            if torch.cuda.device_count() > 1:
                log_variational = self.model.module.log_variational
            else:
                log_variational = self.model.log_variational
            if log_variational == True:
                x_ = torch.log(1 + x_)

            if torch.cuda.device_count() > 1:
                qz_m, qz_v, z = self.model.module.z_encoder(x_, None)
            else:
                qz_m, qz_v, z = self.model.z_encoder(x_, None)

            sample1, sample2= self.adv_load_minibatch(z, batch_index)
            adv_loss, obj2_minibatch = self.adv_loss(sample1, sample2)

        elif self.cal_loss == True and self.cal_adv_loss == True:
            reconst_loss, kl_divergence, qz_m, qz_v, z = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
            loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)

            if self.adv_estimator == 'MINE':
                sample1, sample2 = self.adv_load_minibatch(z, batch_index)
                adv_loss, obj2_minibatch = self.adv_loss(sample1, sample2)
            elif self.adv_estimator in ['MMD','stdz_MMD']:
                reference_batch = 0
                for i in range(self.gene_dataset.n_batches -1):
                    compare_batch = i + 1
                    sample1, sample2 = self.adv_load_minibatch(z, batch_index, reference_batch, compare_batch)
                    adv_loss_one, obj2_minibatch_one = self.adv_loss(sample1, sample2)
                    if i == 0:
                        adv_loss_tensor = adv_loss_one.reshape(1,1)
                    else:
                        adv_loss_tensor = torch.cat([adv_loss_tensor, adv_loss_one.reshape(1,1)],dim=1)
                adv_loss = torch.max(adv_loss_tensor)
                obj2_minibatch = adv_loss

            if self.epoch >= 0:
                NN_estimator = Nearest_Neighbor_Estimate(batch_index, z)
                if len(self.batch_ratio)>0:
                    empirical_MI = EmpiricalMI_From_Aggregated_Posterior(qz_m, qz_v, batch_index, self.batch_ratio.to(self.device), self.nsamples)
                    print('Epoch: {}, neg_ELBO: {}, {}: {}, empirical_MI: {}, NN: {}.'.format(self.epoch, loss, self.adv_estimator, obj2_minibatch, empirical_MI, NN_estimator))
                elif self.cross_validation==False:
                    print('Epoch: {}, neg_ELBO: {}, {}: {}, NN: {}.'.format(self.epoch, loss, self.adv_estimator, obj2_minibatch, NN_estimator))
                else:
                    print('Epoch: {}, neg_ELBO: {}, {}: {}, regularized_loss: {}.'.format(self.epoch, loss, self.adv_estimator,
                                                                            obj2_minibatch, loss + self.regularize_weight*adv_loss))
        #objective 1 equals loss
        if self.cal_loss == True and self.cal_adv_loss == False:
            return loss, None, None
        elif self.cal_loss == False and self.cal_adv_loss == True:
            return None, adv_loss, obj2_minibatch
        elif self.cal_loss == True and self.cal_adv_loss == True and self.cross_validation==False:
            return loss, adv_loss, obj2_minibatch
        elif self.cal_loss == True and self.cal_adv_loss == True and self.cross_validation==True:
            return loss + self.regularize_weight*adv_loss

    def adv_load_minibatch(self, z, batch_index, reference_batch:float=0, compare_batch: float=1):

        if torch.cuda.is_available():
            batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.cpu().numpy().ravel())})
        else:
            batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_index.numpy().ravel())})
        batch_dummy = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)

        #check if z.requires_grad == True
        batch_dummy = Variable(batch_dummy.to(self.device), requires_grad=True)

        if self.adv_estimator == 'MINE':
            z_batch = torch.cat((z, batch_dummy), 1)  # joint
            shuffle_index = torch.randperm(z.shape[0])
            shuffle_z_batch = torch.cat((z[shuffle_index], batch_dummy), 1)  # marginal
            return z_batch, shuffle_z_batch
        elif self.adv_estimator in ['MMD','stdz_MMD']:
            batch_index_copy = batch_index.type(torch.FloatTensor).to(self.device)
            batch_index_copy = Variable(batch_index_copy.type(torch.FloatTensor), requires_grad=True)
            if self.adv_estimator == 'stdz_MMD':
                #standardize each dimension for z
                z_mean = torch.mean(z,0).unsqueeze(0).expand(int(z.size(0)), int(z.size(1)))
                z_std = torch.std(z,0).unsqueeze(0).expand(int(z.size(0)), int(z.size(1)))
                z = (z - z_mean)/z_std #element by element
            z_reference_batch = z[(batch_index_copy[:, 0] == reference_batch).nonzero().squeeze(1)]
            z_compare_batch = z[(batch_index_copy[:, 0] == compare_batch).nonzero().squeeze(1)]
            return z_reference_batch, z_compare_batch

    def adv_loss(self, sample1, sample2):
        if self.adv_estimator == 'MINE':
            t = self.adv_model(sample1)
            et = torch.exp(self.adv_model(sample2))

            if torch.cuda.device_count() > 1:
                if self.adv_model.module.unbiased_loss:
                    if self.adv_model.module.ma_et is None:
                        self.adv_model.module.ma_et = torch.mean(et).detach().item()  # detach means will not calculate gradient for ma_et, ma_et is just a number
                    self.adv_model.module.ma_et = (1 - self.adv_model.module.ma_rate) * self.adv_model.module.ma_et + self.adv_model.module.ma_rate * torch.mean(et).detach().item()

                    # Pay attention, this unbiased loss is not our MINE estimator,
                    # The MINE estimator is still torch.mean(t) - torch.log(torch.mean(et)) after training
                    # The unbiased_loss is only for getting unbiased gradient.
                    loss = -(torch.mean(t) - (1 / self.adv_model.module.ma_et) * torch.mean(et))
                else:
                    loss = -(torch.mean(t) - torch.log(torch.mean(et)))

            else:
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
        else:
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
