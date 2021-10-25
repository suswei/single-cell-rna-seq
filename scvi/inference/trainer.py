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
from scvi.models.modules import MINE_Net, MMD_loss
from tqdm import trange

from scvi.inference.posterior import Posterior
import matplotlib.pyplot as plt

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

    def __init__(self, model, gene_dataset, num_workers=1, metrics_to_monitor=None,
                 benchmark=False, verbose=False, frequency=None, weight_decay=1e-6, early_stopping_kwargs=dict(),
                 data_loader_kwargs=dict(), save_path='None', batch_size=128, adv_estimator='None',
                 adv_n_hidden=128, adv_n_layers=10, adv_activation_fun='ELU', unbiased_loss=True, adv_w_initial='Normal',
                 MMD_kernel_mul: float=2, MMD_kernel_num: int=5, batch_ratio: list=[], nsamples: int=1e5):

        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()

        # use all available GPUs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.adv_estimator = adv_estimator
        if self.adv_estimator == 'MINE':
            adv_input_dim = self.model.n_latent + self.model.n_batch
            self.adv_model = MINE_Net(input_dim=adv_input_dim, n_hidden=adv_n_hidden, n_layers=adv_n_layers,
                             activation_fun=adv_activation_fun, unbiased_loss=unbiased_loss, initial=adv_w_initial)
        elif self.adv_estimator in ['MMD','stdMMD']:
            self.MMD_loss = MMD_loss(kernel_mul = MMD_kernel_mul, kernel_num = MMD_kernel_num)

        self.batch_ratio = batch_ratio
        self.nsamples = nsamples

        self.data_loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.use_cuda
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

        #self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        #self.compute_metrics() #the model will be evaluated one epoch before training.

        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                for tensors_list in self.data_loaders_loop():
                    loss = self.loss(*tensors_list)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #if not self.on_epoch_end():
                #    break

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

    def obj1_obj2_eval(self, type='obj1'):

        with torch.no_grad():
            self.model.eval()
            if self.adv_estimator=='MINE':
                self.adv_model.eval()

            if type == 'obj1':
                self.cal_loss = True
                self.cal_adv_loss = False
            elif type == 'obj2':
                self.cal_loss = False
                self.cal_adv_loss = True

            minibatch_train_eval, minibatch_test_eval = [], []
            for tensors_list in self.train_set:
                if type == 'obj1':
                    minibatch, _, _ = self.two_loss(tensors_list)
                elif type == 'obj2':
                    _, _, minibatch = self.two_loss(tensors_list)
                minibatch_train_eval.append(minibatch.data)
            for tensors_list in self.test_set:
                if type == 'obj1':
                    minibatch, _, _ = self.two_loss(tensors_list)
                elif type == 'obj2':
                    _, _, minibatch = self.two_loss(tensors_list)
                minibatch_test_eval.append(minibatch.data)
        train_eval = sum(minibatch_train_eval) / len(minibatch_train_eval)
        test_eval = sum(minibatch_test_eval) / len(minibatch_test_eval)
        return train_eval, test_eval

    def diagnosis_plot(self, list1, list2, type, path):
        plt.figure()
        plt.plot(np.arange(0, len(list1), 1), list1, label='train')
        plt.plot(np.arange(0, len(list2), 1), list2, label='test')
        plt.legend()
        plt.xticks(np.arange(0, len(list1), 1))
        plt.title('train error vs test error', fontsize=10)
        if type == 'obj1':
            plt.ylabel('error (negative ELBO)')
        elif type == 'obj2':
            if self.adv_estimator=='MINE':
                plt.ylabel('MINE')
            elif self.adv_estimator=='MMD':
                plt.ylabel('MMD')
            elif self.adv_estimator == 'stdMMD':
                plt.ylabel('stdMMD')

        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig("{}/{}_train_test_error.png".format(path,type))
        plt.close()

    def load_dict(self, path, lr, eps, adv_lr):
        self.model.load_state_dict(torch.load(path + '/vae.pkl', map_location='cpu'))

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30], gamma=0.5)

        if self.adv_estimator == 'MINE':
            self.adv_model.load_state_dict(torch.load(path + '/MINE.pkl', map_location='cpu'))

            if torch.cuda.device_count() > 1:
                self.adv_model = torch.nn.DataParallel(self.adv_model)
            self.adv_model.to(self.device)

            self.adv_optimizer = torch.optim.Adam(self.adv_model.parameters(), lr=adv_lr)
            # self.adv_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.adv_optimizer, milestones=[30], gamma=0.5)

    def regularize_fun(self, epochs, adv_epochs, weight, obj1_max, obj1_min, obj2_max, obj2_min):

        for self.epoch in range(epochs):
            self.model.train()
            if self.adv_estimator == 'MINE':
                self.adv_model.train()

            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    for adv_epoch in range(adv_epochs):
                        for adv_tensors_list in self.data_loaders_loop():
                            _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                            self.adv_optimizer.zero_grad()
                            self.optimizer.zero_grad()
                            adv_loss.backward()
                            self.adv_optimizer.step()

                if weight == 0:
                    self.cal_loss = False
                    self.cal_adv_loss = True
                elif weight == 1:
                    self.cal_loss = True
                    self.cal_adv_loss = False
                else:
                    self.cal_loss = True
                    self.cal_adv_loss = True

                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)
                if weight == 0:
                    regularize_loss = (1 - weight) * (obj2_minibatch - obj2_min) / (obj2_max - obj2_min)
                elif weight == 1:
                    regularize_loss = weight * (obj1_minibatch - obj1_min) / (obj1_max - obj1_min)
                else:
                    regularize_loss = weight*(obj1_minibatch - obj1_min)/(obj1_max - obj1_min) + (1- weight)*(obj2_minibatch - obj2_min)/(obj2_max - obj2_min)

                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()

                regularize_loss.backward()
                self.optimizer.step()

                print('z_encoder, the first layer weight:')
                print(self.model.z_encoder.encoder.fc_layers[0][0].weight)


    def pretrain_paretoMTL(self, pre_train: bool=False, pre_epochs: int=200, pre_lr: float=1e-3, eps: float = 0.01,
        pre_adv_epochs: int=400, pre_adv_lr: float=5e-5, path: str='None', lr: float=1e-3, adv_lr: float=5e-5, standardize: bool=False,
        std_paretoMTL: bool = False, n_tasks: int=2, npref: int=10, pref_idx: int=0, taskid: int=0, epochs: int=150, adv_epochs: int=1,
        obj1_max: float = 20000, obj1_min: float = 12000, obj2_max: float = 0.6, obj2_min: float = 0, regularize: bool=False, weight: float=1/11):

        if pre_train == True:

            obj1_train_list, obj1_test_list, obj2_train_list, obj2_test_list = [], [], [], []

            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)

            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=pre_lr, eps=eps)

            # pretrain vae_MI, here loss is not standardized
            self.cal_loss = True
            self.cal_adv_loss = False
            for pre_epoch in range(pre_epochs):
                self.model.train()
                for tensors_list in self.data_loaders_loop():
                    loss, _, _ = self.two_loss(*tensors_list)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if pre_epoch % 10 == 0:
                    train_eval, test_eval = self.obj1_obj2_eval(type='obj1')
                    obj1_train_list.append(train_eval)
                    obj1_test_list.append(test_eval)

            self.diagnosis_plot(obj1_train_list[1:], obj1_test_list[1:], 'obj1', path)

            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), path + '/vae.pkl')
            else:
                torch.save(self.model.state_dict(), path + '/vae.pkl')

            if self.adv_estimator == 'MINE':

                if torch.cuda.device_count() > 1:
                    self.adv_model = torch.nn.DataParallel(self.adv_model)
                self.adv_model.to(self.device)

                self.adv_optimizer = torch.optim.Adam(self.adv_model.parameters(), lr=pre_adv_lr)
                # pretrain adv_model to make MINE works
                self.cal_loss = False
                self.cal_adv_loss = True
                for pre_adv_epoch in range(pre_adv_epochs):
                    self.adv_model.train()
                    for adv_tensors_list in self.data_loaders_loop():
                        _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                        self.adv_optimizer.zero_grad()
                        self.optimizer.zero_grad()
                        adv_loss.backward()
                        self.adv_optimizer.step()
                    if pre_adv_epoch % 10 == 0:
                        train_eval, test_eval = self.obj1_obj2_eval(type='obj2')
                        obj2_train_list.append(train_eval)
                        obj2_test_list.append(test_eval)

                self.diagnosis_plot(obj2_train_list[1:], obj2_test_list[1:], 'obj2', path)
                if torch.cuda.device_count() > 1:
                    torch.save(self.adv_model.module.state_dict(), path + '/MINE.pkl')
                else:
                    torch.save(self.adv_model.state_dict(), path + '/MINE.pkl')
        else:

            self.load_dict(path=path, lr=lr, eps=eps, adv_lr=adv_lr)

            if standardize == True or std_paretoMTL == True:

                obj1_minibatch_list, obj2_minibatch_list = self.paretoMTL(std_paretoMTL=std_paretoMTL, obj1_max=obj1_max,
                    obj1_min=obj1_min, obj2_max=obj2_max, obj2_min=obj2_min, epochs=epochs, adv_epochs=adv_epochs,
                    n_tasks=n_tasks, npref=npref, pref_idx=pref_idx, path=path, taskid=taskid)

                return obj1_minibatch_list, obj2_minibatch_list

            elif regularize == True:

                self.regularize_fun(epochs=epochs, adv_epochs=adv_epochs, weight=weight, obj1_max=obj1_max, obj1_min=obj1_min, obj2_max=obj2_max, obj2_min=obj2_min)

    def paretoMTL(self, std_paretoMTL: bool=False, obj1_max: float=20000, obj1_min: float=12000, obj2_max: float=0.6, obj2_min: float=0,
                  epochs: int=50, adv_epochs: int=1, n_tasks: int=2, npref: int=10, pref_idx: int=0, path: str='.', taskid: int=0):

        ref_vec = torch.FloatTensor(circle_points([1], [npref])[0]).to(self.device)

        obj1_minibatch_list, obj2_minibatch_list, obj1_train_list, obj1_test_list, obj2_train_list, obj2_test_list = [], [], [], [],[],[]
        # run until the initial solution is found when flag==True
        # usually can be found with a few steps

        flag = False
        t = 0
        while flag==False:
            t +=1
            self.model.train()
            if self.adv_estimator == 'MINE':
                self.adv_model.train()
            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    for adv_epoch in range(adv_epochs):
                        for adv_tensors_list in self.data_loaders_loop():
                            _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                            self.adv_optimizer.zero_grad()
                            self.optimizer.zero_grad()
                            adv_loss.backward()
                            self.adv_optimizer.step()

                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                if std_paretoMTL == True:
                    obj1 = (obj1_minibatch - obj1_min)/(obj1_max - obj1_min)
                    obj2 = (obj2_minibatch - obj2_min)/(obj2_max - obj2_min)
                else:
                    obj1 = obj1_minibatch
                    obj2 = obj2_minibatch

                # calculate the weights
                grads, losses_vec = self.paretoMTL_param(n_tasks, obj1, obj2)
                flag, weight_vec = get_d_paretomtl_init(grads, losses_vec, ref_vec, pref_idx)

                weight_vec = weight_vec.to(self.device)

                # stop once a feasible solution is obtained
                if flag == True:
                    print("feasible solution is obtained.")
                    print('t:{}'.format(t))
                    break

                # optimization step
                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1_minibatch_list.append(obj1_minibatch.data)
                obj2_minibatch_list.append(obj2_minibatch.data)

                if std_paretoMTL == True:
                    obj1 = (obj1_minibatch - obj1_min) / (obj1_max - obj1_min)
                    obj2 = (obj2_minibatch - obj2_min) / (obj2_max - obj2_min)
                else:
                    obj1 = obj1_minibatch
                    obj2 = obj2_minibatch

                for i in range(n_tasks):
                    if i == 0:
                        loss_total = weight_vec[i] * obj1
                    else:
                        loss_total = loss_total + weight_vec[i] * obj2

                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()

                loss_total.backward()
                self.optimizer.step()

            else:
            # continue if no feasible solution is found
                continue
            # break the loop once a feasible solutions is found
            break

        # run n_epochs of ParetoMTL
        for self.epoch in range(epochs):

            # self.scheduler.step()
            self.model.train()
            if self.adv_estimator == 'MINE':
                # self.adv_scheduler.step()
                self.adv_model.train()

            for tensors_list in self.data_loaders_loop():

                if self.adv_estimator == 'MINE':
                    self.cal_loss = False
                    self.cal_adv_loss = True
                    for adv_epoch in range(adv_epochs):
                        for adv_tensors_list in self.data_loaders_loop():
                            _, adv_loss, _ = self.two_loss(*adv_tensors_list)
                            self.adv_optimizer.zero_grad()
                            self.optimizer.zero_grad()
                            adv_loss.backward()
                            self.adv_optimizer.step()

                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                if std_paretoMTL == True:
                    obj1 = (obj1_minibatch - obj1_min)/(obj1_max - obj1_min)
                    obj2 = (obj2_minibatch - obj2_min)/(obj2_max - obj2_min)
                else:
                    obj1 = obj1_minibatch
                    obj2 = obj2_minibatch

                # calculate the weights
                grads, losses_vec = self.paretoMTL_param(n_tasks, obj1, obj2)
                weight_vec = get_d_paretomtl(grads, losses_vec, ref_vec, pref_idx)

                normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
                weight_vec = weight_vec * normalize_coeff
                weight_vec = weight_vec.to(self.device)

                # optimization step
                self.cal_loss = True
                self.cal_adv_loss = True
                obj1_minibatch, _, obj2_minibatch = self.two_loss(*tensors_list)

                obj1_minibatch_list.append(obj1_minibatch.data)
                obj2_minibatch_list.append(obj2_minibatch.data)

                if std_paretoMTL == True:
                    obj1 = (obj1_minibatch - obj1_min)/(obj1_max - obj1_min)
                    obj2 = (obj2_minibatch - obj2_min)/(obj2_max - obj2_min)
                else:
                    obj1 = obj1_minibatch
                    obj2 = obj2_minibatch

                for i in range(n_tasks):
                    if i == 0:
                        loss_total = weight_vec[i] * obj1
                    else:
                        loss_total = loss_total + weight_vec[i] * obj2

                self.optimizer.zero_grad()
                if self.adv_estimator == 'MINE':
                    self.adv_optimizer.zero_grad()

                loss_total.backward()
                self.optimizer.step()
            # diagnosis
            if std_paretoMTL == True and self.epoch % 10 == 0:
                obj1_train_eval, obj1_test_eval = self.obj1_obj2_eval(type='obj1')
                obj1_train_list.append(obj1_train_eval)
                obj1_test_list.append(obj1_test_eval)
                obj2_train_eval, obj2_test_eval = self.obj1_obj2_eval(type='obj2')
                obj2_train_list.append(obj2_train_eval)
                obj2_test_list.append(obj2_test_eval)

        if std_paretoMTL==True:
            if self.adv_estimator == 'MINE':
                string = 'paretoMINE'
            else:
                string = 'paretoMMD'
            path = os.path.dirname(os.path.dirname(path)) + '/{}/taskid{}'.format(string, taskid)
            self.diagnosis_plot(obj1_train_list, obj1_test_list, 'obj1', path)
            self.diagnosis_plot(obj2_train_list, obj2_test_list, 'obj2', path)

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
