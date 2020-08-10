import os
import sys
import time
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from scvi.models.modules import MINE_Net3
from tqdm import trange

from scvi.inference.posterior import Posterior
import plotly.graph_objects as go
import pickle

from paretoMTL_helper import circle_points, get_d_paretomtl_init, get_d_paretomtl

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
                 data_loader_kwargs=dict(), save_path='None', batch_size=128, adv_estimator='None',
                 adv_n_hidden=128, adv_n_layers=10, adv_activation_fun='ELU', unbiased_loss=True, adv_w_initial='Normal'):

        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()

        self.adv_estimator = adv_estimator
        if self.adv_estimator == 'MINE':
            adv_input_dim = self.model.n_latent + self.model.n_batch
            self.adv_model = MINE_Net3(input_dim=adv_input_dim, n_hidden=adv_n_hidden, n_layers=adv_n_layers,
                             activation_fun=adv_activation_fun, unbiased_loss=unbiased_loss, initial=adv_w_initial)

        self.data_loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": use_cuda
        }
        self.save_path = save_path
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
        self.compute_metrics() #the model will be evaluated one epoch before training.

        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                for tensors_list in self.data_loaders_loop():
                    loss, ELBO = self.loss(*tensors_list)
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

    def train_test(self, model=None, gene_dataset=None, train_size=0.1, test_size=None, seed=0, type_class=Posterior):
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

        return (
            self.create_posterior(model, gene_dataset, indices=indices_train, type_class=type_class),
            self.create_posterior(model, gene_dataset, indices=indices_test, type_class=type_class)
        )

    def create_posterior(self, model=None, gene_dataset=None, shuffle=False, indices=None, type_class=Posterior):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = self.gene_dataset if gene_dataset is None and hasattr(self, "model") else gene_dataset

        return type_class(model, gene_dataset, shuffle=shuffle, indices=indices, use_cuda=self.use_cuda,
                          data_loader_kwargs=self.data_loader_kwargs)

    def adversarial_train(self, pre_n_epochs, pre_lr, pre_adv_epochs, adv_lr, n_epochs, lr,
                          eps: float = 0.01, std: bool=False, min_obj1: float = 10000, max_obj1: float = 20000,
                          min_obj2: float = 0, max_obj2: float = 0.9, scale: float = 0):

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=pre_lr, eps=eps)
        self.adv_optimizer = torch.optim.Adam(self.adv_model.parameters(), lr=adv_lr)

        #pretrain vae_MI, here loss is not standardized
        self.cal_loss = True
        self.cal_adv_loss = False
        for pre_n_epoch in range(pre_n_epochs):
            for tensors_list in self.data_loaders_loop():
                loss, _, _ = self.two_loss(*tensors_list)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # pretrain adv_model, here adv_loss is not standardized
        self.cal_loss = False
        self.cal_adv_loss = True
        for pre_adv_epoch in range(pre_adv_epochs):
            for adv_tensors_list in self.data_loaders_loop():
                _, adv_loss, obj2_minibatch = self.two_loss(*adv_tensors_list)
                adv_loss.backward()
                self.adv_optimizer.step()
                self.adv_optimizer.zero_grad()
                self.optimizer.zero_grad()

        #change the learning rate for vae_MI after pre-training if necessary
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        loss_minibatch_list, obj2_minibatch_list = [], []
        for self.epoch in range(n_epochs):

            self.model.train()
            self.adv_model.train()

            for tensors_list in self.data_loaders_loop():

                # here adv_loss is also not standardized
                self.cal_loss = False
                _, adv_loss, obj2_minibatch = self.two_loss(*tensors_list)
                adv_loss.backward()
                self.adv_optimizer.step()
                self.adv_optimizer.zero_grad()
                self.optimizer.zero_grad()
                self.cal_loss = True

                loss, adv_loss, obj2_minibatch = self.two_loss(*tensors_list)

                #pay attention, although when train MINE, the adv_loss is the unbiased loss
                #however, in torch.max(obj1, obj2), the obj2 should be MINE estimator, not negative of unbiased loss
                #Obj1 is just loss.
                if std == True:
                    std_obj1 = (loss - min_obj1)/(max_obj1 - min_obj1)
                    # pay attention: here adv_min_value, adv_max_value of MINE_estimator_minibatch
                    std_obj2 = (obj2_minibatch - min_obj2) / (max_obj2 - min_obj2)
                    combined_loss = torch.max((1 - scale)*std_obj1, scale*std_obj2)
                    if combined_loss.item() == (1 - scale) * std_obj1.item():
                        combined_loss = loss
                    elif combined_loss.item() == scale * std_obj2.item():
                        combined_loss = obj2_minibatch
                else: #to get min and max value to standardize obj1 and obj2
                    combined_loss = torch.max((1 - scale) * loss, scale * obj2_minibatch)

                combined_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.adv_optimizer.zero_grad()
                self.optimizer.zero_grad()

                loss_minibatch_list.append(loss.item())
                obj2_minibatch_list.append(obj2_minibatch.item())

        return loss_minibatch_list, obj2_minibatch_list

    def pretrain_gradnorm_paretoMTL(self, pre_train: bool=False, pre_epochs: int=250, pre_lr: float=1e-3, eps: float = 0.01,
                        pre_adv_epochs: int=100, adv_lr: float=5e-5, path: str='None', gradnorm_hypertune: bool=False,
                        alpha: float=3, gradnorm_epochs: int=100, gradnorm_lr: float=1e-3, shared_layer: str='last', gradnorm_weights_idx: int=0,
                        mid_epochs: int=50, n_epochs: int=50, lr: float=1e-4, n_tasks: int=2, npref: int=10, pref_idx: int=0, gradnorm_paretoMTL: bool=False):

        if pre_train == True:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=pre_lr, eps=eps)

            # pretrain vae_MI, here loss is not standardized
            self.cal_loss = True
            self.cal_adv_loss = False
            for pre_n_epoch in range(pre_epochs):
                for tensors_list in self.data_loaders_loop():
                    loss, _, _ = self.two_loss(*tensors_list)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if pre_train == True:
                torch.save(self.model.state_dict(), path + '/vae.pkl')

            if self.adv_estimator == 'MINE':
                self.adv_optimizer = torch.optim.Adam(self.adv_model.parameters(), lr=adv_lr)
                # pretrain adv_model to make MINE works
                self.cal_loss = False
                self.cal_adv_loss = True
                for pre_adv_epoch in range(pre_adv_epochs):
                    for adv_tensors_list in self.data_loaders_loop():
                        _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                        self.adv_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        adv_loss.backward()
                        self.adv_optimizer.step()
                if pre_train == True:
                    torch.save(self.adv_model.state_dict(), path + '/MINE.pkl')

        else:
            self.model.load_state_dict(torch.load(path + '/vae.pkl'))
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30], gamma=0.5)

            if self.adv_estimator == 'MINE':
                self.adv_model.load_state_dict(torch.load(path + '/MINE.pkl'))
                self.adv_optimizer = torch.optim.Adam(self.adv_model.parameters(), lr=adv_lr)
                # self.adv_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.adv_optimizer, milestones=[30], gamma=0.5)

            if gradnorm_hypertune == True:

                # Pretrain gradnorm
                weightloss1 = [1]
                weightloss2 = [1]
                gradnorm_weights = [weightloss1, weightloss2]

                gradnorm_weights, weightloss1_list, weightloss2_list = self.gradnorm(alpha, gradnorm_epochs, gradnorm_lr, gradnorm_weights, shared_layer)

                return gradnorm_weights, weightloss1_list, weightloss2_list

            else:
                # train vae_MI further in addition to pre_train and loss is not standardized
                self.cal_loss = True
                self.cal_adv_loss = False
                for mid_n_epoch in range(mid_epochs):
                    for tensors_list in self.data_loaders_loop():
                        loss, _, _ = self.two_loss(*tensors_list)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                gradnorm_weights_path = os.path.dirname(os.path.dirname(path)) + '/gradnorm_hypertune/taskid{}/results.pkl'.format(gradnorm_weights_idx)
                gradnorm_weights_dict = pickle.load(open(gradnorm_weights_path, "rb"))
                gradnorm_weights = gradnorm_weights_dict['gradnorm_weights']

                if gradnorm_paretoMTL == True:
                    # gradnorm + paretoMTL
                    obj1_minibatch_list, obj2_minibatch_list = self.paretoMTL(n_epochs=n_epochs, n_tasks=n_tasks,
                        npref=npref, pref_idx=pref_idx, gradnorm_weights=gradnorm_weights, gradnorm_paretoMTL=True,
                        gradnorm_lr=gradnorm_lr, shared_layer=shared_layer, alpha=alpha)
                else:
                    # gradnorm_weights remain constant during paretoMTL
                    obj1_minibatch_list, obj2_minibatch_list = self.paretoMTL(n_epochs=n_epochs, n_tasks=n_tasks,
                        npref=npref, pref_idx=pref_idx, gradnorm_weights=gradnorm_weights, gradnorm_paretoMTL=False)

                return obj1_minibatch_list, obj2_minibatch_list

    def gradnorm(self, alpha, gradnorm_epochs, gradnorm_lr, params, shared_layer):

        print('begin GradNorm standardization')
        # weightloss1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        # weightloss2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        weightloss1 = torch.FloatTensor(params[0]).clone().detach().requires_grad_(True)
        weightloss2 = torch.FloatTensor(params[1]).clone().detach().requires_grad_(True)
        params = [weightloss1, weightloss2]

        gradnorm_opt = torch.optim.Adam(params, lr=gradnorm_lr)
        gradnorm_scheduler = torch.optim.lr_scheduler.MultiStepLR(gradnorm_opt, milestones=[100, 200], gamma=0.5)
        Gradloss = torch.nn.L1Loss()

        weightloss1_list = [weightloss1.data.tolist()[0]]
        weightloss2_list = [weightloss2.data.tolist()[0]]

        obj1_minibatch_list, obj2_minibatch_list = [], []

        for epoch in range(gradnorm_epochs):  # loop over the dataset multiple times

            gradnorm_scheduler.step()

            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    for adv_tensors_list in self.data_loaders_loop():
                        _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                        self.adv_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        adv_loss.backward()
                        self.adv_optimizer.step()

                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1_minibatch_list.append(obj1_minibatch.data)
                obj2_minibatch_list.append(obj2_minibatch.data)

                l1 = params[0] * obj1_minibatch
                l2 = params[1] * obj2_minibatch

                # for the first epoch with no l0
                if epoch == 0:
                    l01 = l1.data
                    l02 = l2.data

                # compute the weighted loss w_i(t) * L_i(t)
                weighted_task_loss = torch.div(torch.add(l1, l2), 2)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()
                weighted_task_loss.backward(retain_graph=True)

                gradnorm_opt.zero_grad()
                # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
                G1, G2, C1, C2 = self.gradnorm_loss_param(l1, l2, l01, l02, shared_layer, alpha)
                Lgrad = torch.add(Gradloss(G1 ** (1 / 2), C1), Gradloss(G2 ** (1 / 2), C2))
                Lgrad.backward()  # Lgrad is differentiated only with respect to the params

                # Updating loss weights
                gradnorm_opt.step()

                # Updating the model weights
                self.optimizer.step()

                # Renormalizing the losses weights
                coef = 2 / torch.add(weightloss1, weightloss2)
                params = [coef * weightloss1, coef * weightloss2]

                weightloss1_list.append(params[0].data.tolist()[0])
                weightloss2_list.append(params[1].data.tolist()[0])

            print("obj1 multiplier: {}, obj2: multiplier {}".format(params[0].data, params[1].data))
        print('finish GradNorm standardization')
        return params, weightloss1_list, weightloss2_list, obj1_minibatch_list, obj2_minibatch_list

    def gradnorm_loss_param(self, l1, l2, l01, l02, shared_layer: str='last', alpha: float=3):

        # For each task, getting the L2 norm of the gradient of the weighted single-task loss
        # only the parameters for z_encoder are shared.
        temp = list(self.model.z_encoder.parameters())
        G1 = 0.0
        G2 = 0.0
        if shared_layer == 'all':
            for i in range(temp.__len__()):
                G1R = torch.autograd.grad(l1, temp[i], retain_graph=True, create_graph=True)
                G1 += torch.norm(G1R[0], 2) ** 2
                G2R = torch.autograd.grad(l2, temp[i], retain_graph=True, create_graph=True)
                G2 += torch.norm(G2R[0], 2) ** 2
        elif shared_layer == 'last':
            G1R = torch.autograd.grad(l1, temp[-1], retain_graph=True, create_graph=True)
            G1 += torch.norm(G1R[0], 2) ** 2
            G2R = torch.autograd.grad(l2, temp[-1], retain_graph=True, create_graph=True)
            G2 += torch.norm(G2R[0], 2) ** 2

        G_avg = torch.div(torch.add(G1 ** (1 / 2), G2 ** (1 / 2)), 2)

        # Calculating relative losses
        lhat1 = torch.div(l1, l01)
        lhat2 = torch.div(l2, l02)
        lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

        # Calculating relative inverse training rates for tasks
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)

        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg * (inv_rate1) ** alpha
        C2 = G_avg * (inv_rate2) ** alpha
        C1 = C1.detach()
        C2 = C2.detach()

        G1 = torch.reshape(G1,(-1,))
        G2 = torch.reshape(G2,(-1,))

        return G1, G2, C1, C2

    def paretoMTL(self, n_epochs: int=50, n_tasks: int=2, npref: int=10, pref_idx: int=0, gradnorm_weights: list=[1,1],
                  gradnorm_paretoMTL: bool=False, gradnorm_lr: float=1e-3, shared_layer: str='last', alpha: float=3):

        ref_vec = torch.FloatTensor(circle_points([1], [npref])[0])

        if gradnorm_paretoMTL==True:
            weightloss1 = torch.FloatTensor([gradnorm_weights[0].data.tolist()[0]]).clone().detach().requires_grad_(True)
            weightloss2 = torch.FloatTensor([gradnorm_weights[1].data.tolist()[0]]).clone().detach().requires_grad_(True)
            gradnorm_weights = [weightloss1, weightloss2]

            gradnorm_opt = torch.optim.Adam(gradnorm_weights, lr=gradnorm_lr)
            Gradloss = torch.nn.L1Loss()

        obj1_minibatch_list, obj2_minibatch_list = [], []
        # run at most 2 epochs to find the initial solution
        # stop early once a feasible solution is found
        # usually can be found with a few steps
        for t in range(2):

            self.model.train()
            if self.adv_estimator == 'MINE':
                self.adv_model.train()
            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    # minibatch_number = 0
                    for adv_tensors_list in self.data_loaders_loop():
                        # minibatch_number += 1
                        _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                        self.adv_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        adv_loss.backward()
                        self.adv_optimizer.step()
                        # if minibatch_number == 20:
                        #    break

                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                #refer to gradnorm, gradnorm_weights are renormalized by number of objectives
                obj1 = torch.div(obj1_minibatch * gradnorm_weights[0].data.tolist()[0], 2)
                obj2 = torch.div(obj2_minibatch * gradnorm_weights[1].data.tolist()[0], 2)

                # calculate the weights
                grads, losses_vec = self.paretoMTL_param(n_tasks, obj1, obj2)
                flag, weight_vec = get_d_paretomtl_init(grads, losses_vec, ref_vec, pref_idx)

                # early stop once a feasible solution is obtained
                if flag == True:
                    print("feasible solution is obtained.")
                    break

                # optimization step
                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1_minibatch_list.append(obj1_minibatch.data)
                obj2_minibatch_list.append(obj2_minibatch.data)

                if gradnorm_paretoMTL == True:
                    l1 = gradnorm_weights[0] * obj1_minibatch
                    l2 = gradnorm_weights[1] * obj2_minibatch

                    # for the first epoch with no l0
                    if t == 0:
                        l01 = l1.data
                        l02 = l2.data

                obj1 = torch.div(obj1_minibatch * gradnorm_weights[0], 2)
                obj2 = torch.div(obj2_minibatch * gradnorm_weights[1], 2)

                for i in range(n_tasks):
                    if i == 0:
                        loss_total = weight_vec[i] * obj1
                    else:
                        loss_total = loss_total + weight_vec[i] * obj2

                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()

                if gradnorm_paretoMTL == True:
                    loss_total.backward(retain_graph=True)

                    gradnorm_opt.zero_grad()
                    # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
                    G1, G2, C1, C2 = self.gradnorm_loss_param(l1, l2, l01, l02, shared_layer, alpha)
                    Lgrad = torch.add(Gradloss(G1 ** (1 / 2), C1), Gradloss(G2 ** (1 / 2), C2))
                    Lgrad.backward()

                    # Updating loss weights
                    gradnorm_opt.step()

                    self.optimizer.step()

                    # Renormalizing the losses weights
                    coef = 2 / torch.add(weightloss1, weightloss2)
                    gradnorm_weights = [coef * weightloss1, coef * weightloss2]

                else:
                    loss_total.backward()
                    self.optimizer.step()

            else:
                # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

        # run n_epochs of ParetoMTL
        for self.epoch in range(n_epochs):

            # self.scheduler.step()
            self.model.train()
            if self.adv_estimator == 'MINE':
                # self.adv_scheduler.step()
                self.adv_model.train()

            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    # minibatch_number = 0
                    for adv_tensors_list in self.data_loaders_loop():
                        # minibatch_number += 1
                        _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                        self.adv_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        adv_loss.backward()
                        self.adv_optimizer.step()
                        # if minibatch_number == 20:
                        #    break

                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1 = torch.div(obj1_minibatch * gradnorm_weights[0].data.tolist()[0], 2)
                obj2 = torch.div(obj2_minibatch * gradnorm_weights[1].data.tolist()[0], 2)

                # calculate the weights
                grads, losses_vec = self.paretoMTL_param(n_tasks, obj1, obj2)
                weight_vec = get_d_paretomtl(grads, losses_vec, ref_vec, pref_idx)

                normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
                weight_vec = weight_vec * normalize_coeff

                # optimization step
                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1_minibatch_list.append(obj1_minibatch.data)
                obj2_minibatch_list.append(obj2_minibatch.data)

                if gradnorm_paretoMTL==True:
                    l1 = gradnorm_weights[0] * obj1_minibatch
                    l2 = gradnorm_weights[1] * obj2_minibatch

                    # for the first epoch with no l0
                    if self.epoch == 0:
                        l01 = l1.data
                        l02 = l2.data

                obj1 = torch.div(obj1_minibatch * gradnorm_weights[0], 2)
                obj2 = torch.div(obj2_minibatch * gradnorm_weights[1], 2)

                for i in range(n_tasks):
                    if i == 0:
                        loss_total = weight_vec[i] * obj1
                    else:
                        loss_total = loss_total + weight_vec[i] * obj2

                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()

                if gradnorm_paretoMTL == True:
                    loss_total.backward(retain_graph=True)

                    gradnorm_opt.zero_grad()
                    # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
                    G1, G2, C1, C2 = self.gradnorm_loss_param(l1, l2, l01, l02, shared_layer, alpha)
                    Lgrad = torch.add(Gradloss(G1 ** (1 / 2), C1), Gradloss(G2 ** (1 / 2), C2))
                    Lgrad.backward()

                    # Updating loss weights
                    gradnorm_opt.step()

                    self.optimizer.step()

                    # Renormalizing the losses weights
                    coef = 2 / torch.add(weightloss1, weightloss2)
                    gradnorm_weights = [coef * weightloss1, coef * weightloss2]
                else:
                    loss_total.backward()
                    self.optimizer.step()

        return obj1_minibatch_list, obj2_minibatch_list

    def paretoMTL_param(self, n_tasks, obj1, obj2):

        # obtain and store the gradient
        grads = {}
        losses_vec = []

        for i in range(n_tasks):

            self.optimizer.zero_grad()
            if self.adv_estimator == 'MINE':
                self.adv_optimizer.zero_grad()

            if i == 0:
                losses_vec.append(obj1.data)
                obj1.backward(retain_graph=True)
            elif i == 1:
                losses_vec.append(obj2.data)
                obj2.backward()

            # can use scalable method proposed in the MOO-MTL paper for large scale problem
            # but we keep use the gradient of all parameters in this experiment
            grads[i] = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

        grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
        grads = torch.stack(grads_list)
        losses_vec = torch.stack(losses_vec)

        return grads, losses_vec

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
