import sys
import time
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch
from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scvi.models.modules import Sample_From_Aggregated_Posterior

from tqdm import tqdm, trange

from scvi.inference.posterior import Posterior
import random

class Trainer:
    r"""The abstract Trainer class for training a PyTorch model and monitoring its statistics. It should be
    inherited at least with a .loss() function to be optimized in the training loop.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :use_cuda: Default: ``True``.
        :metrics_to_monitor: A list of the metrics to monitor. If not specified, will use the
            ``default_metrics_to_monitor`` as specified in each . Default: ``None``.
        :benchmark: if True, prevents statistics computation in the training. Default: ``False``.
        :verbose: If statistics should be displayed along training. Default: ``None``.
        :frequency: The frequency at which to keep track of statistics. Default: ``None``.
        :early_stopping_metric: The statistics on which to perform early stopping. Default: ``None``.
        :save_best_state_metric:  The statistics on which we keep the network weights achieving the best store, and
            restore them at the end of training. Default: ``None``.
        :on: The data_loader name reference for the ``early_stopping_metric`` and ``save_best_state_metric``, that
            should be specified if any of them is. Default: ``None``.
    """
    default_metrics_to_monitor = []

    def __init__(self, model, gene_dataset, use_cuda=True, metrics_to_monitor=None, benchmark=False,
                 verbose=False, frequency=None, weight_decay=1e-6, early_stopping_kwargs=dict(),
                 data_loader_kwargs=dict(), batch_size=128, adv_model=None, adv_optimizer=None, adv_epochs=1, change_adv_epochs_index=0, change_adv_epochs=50):

        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()
        self.adv_model = adv_model
        self.adv_optimizer = adv_optimizer
        self.adv_epochs = adv_epochs
        self.change_adv_epochs_index = change_adv_epochs_index
        self.change_adv_epochs = change_adv_epochs

        self.data_loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": use_cuda
        }

        self.data_loader_kwargs.update(data_loader_kwargs)

        self.weight_decay = weight_decay
        self.benchmark = benchmark
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0

        if metrics_to_monitor is not None:
            self.metrics_to_monitor = metrics_to_monitor
        else:
            self.metrics_to_monitor = self.default_metrics_to_monitor

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.frequency = frequency if not benchmark else None
        self.verbose = verbose

        self.history = defaultdict(lambda: [])

    @torch.no_grad()
    def compute_metrics(self):
        begin = time.time()
        epoch = self.epoch + 1
        if self.frequency and (epoch == 0 or epoch == self.n_epochs or (epoch % self.frequency == 0)):
            with torch.set_grad_enabled(False):
                self.model.eval()
                if self.verbose:
                    print("\nEPOCH [%d/%d]: " % (epoch, self.n_epochs))

                for name, posterior in self._posteriors.items():
                    print_name = ' '.join([s.capitalize() for s in name.split('_')[-2:]])
                    if hasattr(posterior, 'to_monitor'):
                        for metric in posterior.to_monitor:
                            if self.verbose:
                                print(print_name, end=' : ')
                            result = getattr(posterior, metric)(verbose=self.verbose)
                            self.history[metric + '_' + name] += [result]
                self.model.train()
        self.compute_metrics_time += time.time() - begin

    def adv_train(self):
        for adv_epoch in tqdm(range(self.adv_epochs)):
            minibatch_index = 0
            for tensor_adv in self.adv_model.data_loader:
                sample_batch_adv, local_l_mean_adv, local_l_var_adv, batch_index_adv, _ = tensor_adv[0]
                x_ = sample_batch_adv
                if self.model.log_variational:
                    x_ = torch.log(1 + x_)
                # Sampling
                qz_m, qz_v, z = self.model.z_encoder(x_, None)
                ql_m, ql_v, library = self.model.l_encoder(x_)
                # z_batch0, z_batch1 = Sample_From_Aggregated_Posterior(qz_m, qz_v,batch_index_adv,self.model.nsamples_z)
                # z_batch0_tensor = Variable(torch.from_numpy(z_batch0).type(torch.FloatTensor), requires_grad=True)
                # z_batch1_tensor = Variable(torch.from_numpy(z_batch1).type(torch.FloatTensor), requires_grad=True)
                batch_index_adv_list = np.ndarray.tolist(batch_index_adv.detach().numpy())
                z_batch0_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
                z_batch1_tensor = z[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
                l_batch0_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [0]], :]
                l_batch1_tensor = library[[i for i in range(len(batch_index_adv_list)) if batch_index_adv_list[i] == [1]], :]
                l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
                l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)

                if (l_z_batch0_tensor.shape[0] == 0) or (l_z_batch1_tensor.shape[0] == 0):
                    continue

                pred_xz = self.adv_model(input=l_z_batch0_tensor)
                pred_x_z = self.adv_model(input=l_z_batch1_tensor)

                if self.adv_model.unbiased_loss:
                    t = pred_xz
                    et = torch.exp(pred_x_z)
                    if self.adv_model.ma_et is None:
                        self.adv_model.ma_et = torch.mean(et).detach().item()
                    self.adv_model.ma_et = (1 - self.adv_model.ma_rate) * self.adv_model.ma_et + self.adv_model.ma_rate * torch.mean(et)
                    # unbiasing use moving average
                    loss_adv2 = -(torch.mean(t) - (1 / self.adv_model.ma_et.mean()).detach() * torch.mean(et))
                else:
                    loss_adv = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
                    loss_adv2 = -loss_adv  # maximizing loss_adv equals minimizing -loss_adv

                self.model.adv_minibatch_MI = -loss_adv2
                print('adv_minibatch_MI: %s' % (-loss_adv2))
                self.adv_optimizer.zero_grad()
                loss_adv2.backward()
                self.adv_optimizer.step()

                if (self.adv_model.save_path != 'None') and (l_z_batch0_tensor.shape[0] != 0) and (l_z_batch1_tensor.shape[0] != 0) and (self.epoch % 10 == 0) and (minibatch_index == len(list(self.adv_model.data_loader)) - 2):
                    activation = {}

                    def get_activation(name):
                        def hook(model, input, output):
                            activation[name] = output.detach()

                        return hook

                    intermediate_adv_model = self.adv_model()

                    intermediate_adv_model.fc2.register_forward_hook(get_activation('fc2'))
                    output0 = intermediate_adv_model(input=l_z_batch0_tensor)
                    fig = plt.figure(figsize=(14, 7))
                    plt.plot([i + 1 for i in range(len(activation['fc2'].squeeze().tolist()))],activation['fc2'].squeeze().tolist())
                    plt.title("Distribution of activations of nodes in hidden layer2 for the second last minibatch in epoch%s" % (self.epoch))
                    fig.savefig(self.adv_model.save_path + 'Dist_of_activations_layer2_epoch%s.png' % (self.epoch))
                    plt.close(fig)

                    for k in range(self.adv_model.n_hidden_layers / 10):
                        intermediate_adv_model.layers[(k + 1) * 10].register_forward_hook(get_activation('fc%s' % ((k + 1) * 10)))
                        output0 = intermediate_adv_model(input=l_z_batch0_tensor)
                        print(activation['fc2'])
                        fig = plt.figure(figsize=(14, 7))
                        plt.plot([i + 1 for i in range(len(activation['fc%s' % ((k + 1) * 10)].squeeze().tolist()))],activation['fc%s' % ((k + 1) * 10)].squeeze().tolist())
                        plt.title("Distribution of activations of nodes in hidden layer%s for the second last minibatch in epoch%s" % ((k + 1) * 10, self.epoch))
                        fig.savefig(self.adv_model.save_path + 'Dist_of_activations_layer%s_epoch%s.png' % ((k + 1) * 10, self.epoch))
                        plt.close(fig)
                minibatch_index += 1

            self.change_adv_epochs_index = 1
            self.adv_epochs = self.change_adv_epochs

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        # if hasattr(self, 'optimizer'):
        #     optimizer = self.optimizer
        # else:
        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)  # weight_decay=self.weight_decay,

        self.n_epochs = n_epochs

        reconst_loss_list = list()
        MI_loss_list = list()

        for self.epoch in range(self.n_epochs):
            #make the minibatches the same for every input datasets
            #torch.manual_seed(0)
            #torch.cuda.manual_seed(0)
            #torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
            #np.random.seed(0)  # Numpy module.
            #random.seed(0)  # Python random module.
            #torch.manual_seed(0)
            #torch.backends.cudnn.benchmark = False
            #torch.backends.cudnn.deterministic = True

            self.adv_train()
            for tensors_list in self.data_loaders_loop():
                if self.model.adv :
                   loss, reconst_loss, MI_loss = self.loss(*tensors_list)
                   reconst_loss_list.append(reconst_loss.detach().cpu().numpy())
                   MI_loss_list.append(MI_loss.detach().cpu().numpy())
                else:
                   loss = self.loss(*tensors_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return reconst_loss_list, MI_loss_list
    '''
    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        self.model.train()
        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        # if hasattr(self, 'optimizer'):
        #     optimizer = self.optimizer
        # else:
        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)  # weight_decay=self.weight_decay,

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        reconst_loss_list = list()
        MI_loss_list = list()

        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                #make the minibatches the same for every input datasets
                #torch.manual_seed(0)
                #torch.cuda.manual_seed(0)
                #torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
                #np.random.seed(0)  # Numpy module.
                #random.seed(0)  # Python random module.
                #torch.manual_seed(0)
                #torch.backends.cudnn.benchmark = False
                #torch.backends.cudnn.deterministic = True

                self.adv_train()
                for tensors_list in self.data_loaders_loop():
                    if self.model.adv :
                       loss, reconst_loss, MI_loss = self.loss(*tensors_list)
                       reconst_loss_list.append(reconst_loss.detach().cpu().numpy())
                       MI_loss_list.append(MI_loss.detach().cpu().numpy())
                    else:
                       loss = self.loss(*tensors_list)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if not self.on_epoch_end():
                    break

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print("\nTraining time:  %i s. / %i epochs" % (int(self.training_time), self.n_epochs))

        return reconst_loss_list, MI_loss_list
    '''
    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.compute_metrics()
        on = self.early_stopping.on
        early_stopping_metric = self.early_stopping.early_stopping_metric
        save_best_state_metric = self.early_stopping.save_best_state_metric
        if save_best_state_metric is not None and on is not None:
            if self.early_stopping.update_state(self.history[save_best_state_metric + '_' + on][-1]):
                self.best_state_dict = self.model.state_dict()
                self.best_epoch = self.epoch

        continue_training = True
        if early_stopping_metric is not None and on is not None:
            continue_training = self.early_stopping.update(
                self.history[early_stopping_metric + '_' + on][-1]
            )
        return continue_training

    @property
    @abstractmethod
    def posteriors_loop(self):
        pass

    def data_loaders_loop(self):  # returns an zipped iterable corresponding to loss signature
        data_loaders_loop = [self._posteriors[name] for name in self.posteriors_loop] #posteriors_loop is a list:['train_set']
        return zip(data_loaders_loop[0], *[cycle(data_loader) for data_loader in data_loaders_loop[1:]])

    def register_posterior(self, name, value):
        name = name.strip('_')
        self._posteriors[name] = value

    def corrupt_posteriors(self, rate=0.1, corruption="uniform", update_corruption=True):
        if not hasattr(self.gene_dataset, 'corrupted') and update_corruption:
            self.gene_dataset.corrupt(rate=rate, corruption=corruption)
        for name, posterior in self._posteriors.items():
            self.register_posterior(name, posterior.corrupted())

    def uncorrupt_posteriors(self):
        for name_, posterior in self._posteriors.items():
            self.register_posterior(name_, posterior.uncorrupted())

    def __getattr__(self, name):
        if '_posteriors' in self.__dict__:
            _posteriors = self.__dict__['_posteriors']
            if name.strip('_') in _posteriors:
                return _posteriors[name.strip('_')]
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        if name.strip('_') in self._posteriors:
            del self._posteriors[name.strip('_')]
        else:
            object.__delattr__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, Posterior):
            name = name.strip('_')
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train_test(self, model=None, gene_dataset=None, train_size=0.1, test_size=None, seed=0, type_class=Posterior, adv=False):
        """
        :param train_size: float, int, or None (default is 0.1)
        :param test_size: float, int, or None (default is None)
        """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset
        n = len(gene_dataset)
        n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        np.random.seed(seed=seed)
        permutation = np.random.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test:(n_test + n_train)]

        if adv:
            return (
                self.create_posterior(model, gene_dataset, indices=indices_train, type_class=type_class),
                self.create_posterior(model, gene_dataset, indices=indices_test, type_class=type_class)
            )
        else:
            return (
                self.create_posterior(model, gene_dataset, indices=indices_train, type_class=type_class),
                self.create_posterior(model, gene_dataset, indices=indices_test, type_class=type_class)
            )

    def create_posterior(self, model=None, gene_dataset=None, shuffle=False, indices=None, type_class=Posterior, adv=False):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset
        if adv:
            return type_class(model, gene_dataset, shuffle=shuffle, indices=indices, use_cuda=self.use_cuda,
                              data_loader_kwargs=self.adv_data_loader_kwargs)
        else:
            return type_class(model, gene_dataset, shuffle=shuffle, indices=indices, use_cuda=self.use_cuda,
                              data_loader_kwargs=self.data_loader_kwargs)


class SequentialSubsetSampler(SubsetRandomSampler):
    def __init__(self, indices):
        self.indices = np.sort(indices)

    def __iter__(self):
        return iter(self.indices)


class EarlyStopping:
    def __init__(self, early_stopping_metric=None, save_best_state_metric=None, on='test_set',
                 patience=15, threshold=3, benchmark=False):
        self.benchmark = benchmark
        self.patience = patience
        self.threshold = threshold
        self.epoch = 0
        self.wait = 0
        self.mode = getattr(Posterior, early_stopping_metric).mode if early_stopping_metric is not None else None
        # We set the best to + inf because we're dealing with a loss we want to minimize
        self.current_performance = np.inf
        self.best_performance = np.inf
        self.best_performance_state = np.inf
        # If we want to maximize, we start at - inf
        if self.mode == "max":
            self.best_performance *= -1
            self.current_performance *= -1
        self.mode_save_state = getattr(Posterior,
                                       save_best_state_metric).mode if save_best_state_metric is not None else None
        if self.mode_save_state == "max":
            self.best_performance_state *= -1

        self.early_stopping_metric = early_stopping_metric
        self.save_best_state_metric = save_best_state_metric
        self.on = on

    def update(self, scalar):
        self.epoch += 1
        if self.benchmark or self.epoch < self.patience:
            continue_training = True
        elif self.wait >= self.patience:
            continue_training = False
        else:
            # Shift
            self.current_performance = scalar

            # Compute improvement
            if self.mode == "max":
                improvement = self.current_performance - self.best_performance
            elif self.mode == "min":
                improvement = self.best_performance - self.current_performance

            # updating best performance
            if improvement > 0:
                self.best_performance = self.current_performance

            if improvement < self.threshold:
                self.wait += 1
            else:
                self.wait = 0

            continue_training = True
        if not continue_training:
            print("\nStopping early: no improvement of more than " + str(self.threshold) +
                  " nats in " + str(self.patience) + " epochs")
            print("If the early stopping criterion is too strong, "
                  "please instantiate it with different parameters in the train method.")
        return continue_training

    def update_state(self, scalar):
        improved = ((self.mode_save_state == "max" and scalar - self.best_performance_state > 0) or
                    (self.mode_save_state == "min" and self.best_performance_state - scalar > 0))
        if improved:
            self.best_performance_state = scalar
        return improved
