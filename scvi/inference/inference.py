import copy

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from scvi.models.modules import Sample_From_Aggregated_Posterior

from . import Trainer

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

    def __init__(self, model, gene_dataset, train_size=0.8, test_size=None, kl=None, seed=0, adv_model=None, adv_optimizer=None, **kwargs):
        super().__init__(model, gene_dataset, **kwargs)
        self.kl = kl
        self.seed = seed
        self.adv_model = adv_model
        self.adv_optimizer = adv_optimizer
        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(model, gene_dataset, train_size, test_size, seed=self.seed)
            self.train_set.to_monitor = ['ll']
            self.test_set.to_monitor = ['ll']

    @property
    def posteriors_loop(self):
        return ['train_set']

    def loss(self, tensors):
        if self.model.adv == True:
            for tensor_adv in self.data_loaders_loop():
                sample_batch_adv, local_l_mean_adv, local_l_var_adv, batch_index_adv, _ = tensor_adv[0]
                x_ = sample_batch_adv
                if self.model.log_variational:
                    x_ = torch.log(1 + x_)
                # Sampling
                qz_m, qz_v, z = self.model.z_encoder(x_, None)
                z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v,batch_index_adv,self.model.nsamples_z)
                z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
                z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
                pred_xz, pred_x_z = self.adv_model(x=z_batch0_tensor, y=z_batch1_tensor)
                loss_adv = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
                loss_adv2 = -loss_adv #maximizing loss_adv equals minimizing -loss_adv
                self.adv_model.zero_grad()
                loss_adv2.backward()
                self.adv_optimizer.step()

            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
            reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
            x_ = sample_batch
            if self.model.log_variational:
                x_ = torch.log(1 + x_)
            # Sampling
            qz_m, qz_v, z = self.model.z_encoder(x_, None)
            z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v, batch_index_adv, self.model.nsamples_z)
            z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
            z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
            pred_xz, pred_x_z = self.adv_model(x=z_batch0_tensor, y=z_batch1_tensor)
            MI_loss = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
            scaled_MI_loss = self.model.MIScale*MI_loss
            print('reconst_loss:{}, MI_loss:{}, scaled_MI_loss:{}'.format(reconst_loss.mean(), MI_loss, scaled_MI_loss))
            loss = torch.mean(reconst_loss + kl_divergence+scaled_MI_loss) #why self.kl_weight * kl_divergence here? Why + here, not -, because the reconst_loss is -logp(), for vae_mine, although reconst_loss's size is 128, kl_divergence's size is 1, they can be added together.
            return loss
        else:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
            reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
            print('penalty:{}'.format(kl_divergence))
            loss = torch.mean(reconst_loss + kl_divergence)  # why + here, not -, because the reconst_loss is -logp(), for vae_mine, although reconst_loss's size is 128, kl_divergence's size is 1, they can be added together.
            return loss

    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / 400)  # self.n_epochs)


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
