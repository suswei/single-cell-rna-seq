for trainer()

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

        ELBO_list = list()
        std_ELBO_list = list()

        if self.model.adv == True:
            penalty_list = list()
            std_penalty_list = list()
            nrows = int(self.adv_model.n_hidden_layers / 10) + 1
            activation_mean = np.empty((nrows, 0), dtype=float)
            activation_var = np.empty((nrows, 0), dtype=float)
        clustermetrics_trainingprocess = pd.DataFrame(
            columns=['nth_epochs', 'Label', 'asw', 'nmi', 'ari', 'uca', 'be', 'ELBO'])
        minibatch_number = len(list(self.data_loaders_loop()))
        self.model.minibatch_number = minibatch_number
        with trange(n_epochs, desc="training", file=sys.stdout, disable=self.verbose) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                # make the minibatches the same for every input datasets
                # torch.manual_seed(0)
                # torch.cuda.manual_seed(0)
                # torch.cuda.manual_seed_all(0)  # if you are using multi-GPU.
                # np.random.seed(0)  # Numpy module.
                # random.seed(0)  # Python random module.
                # torch.manual_seed(0)
                # torch.backends.cudnn.benchmark = False
                # torch.backends.cudnn.deterministic = True

                if self.model.adv == True:
                    for adv_epoch in tqdm(range(self.adv_epochs)):
                        self.adv_model.train()
                        # activation_mean_oneepoch = list()
                        # activation_var_oneepoch = list()
                        # minibatch_index = 0

                        for tensor_adv in self.adv_model.data_loader.data_loaders_loop():
                            # activation_hidden = False
                            # print(minibatch_index)
                            sample_batch_adv, local_l_mean_adv, local_l_var_adv, batch_index_adv, _ = tensor_adv[0]
                            x_ = sample_batch_adv
                            if self.model.log_variational:
                                x_ = torch.log(1 + x_)
                            # Sampling
                            qz_m, qz_v, z = self.model.z_encoder(x_, None)
                            # z = z.detach()
                            ql_m, ql_v, library = self.model.l_encoder(x_)

                            if self.adv_model.name == 'MI':
                                '''
                                z_batch0_tensor = z[(Variable(torch.LongTensor([1])) - batch_index_adv).squeeze(1).byte()]
                                z_batch1_tensor = z[batch_index_adv.squeeze(1).byte()]
                                l_batch0_tensor = library[(Variable(torch.LongTensor([1])) - batch_index_adv).squeeze(1).byte()]
                                l_batch1_tensor = library[batch_index_adv.squeeze(1).byte()]
                                l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
                                l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)

                                if (l_z_batch0_tensor.shape[0] == 0) or (l_z_batch1_tensor.shape[0] == 0):
                                    continue

                                pred_xz = self.adv_model(input=l_z_batch0_tensor)
                                pred_x_z = self.adv_model(input=l_z_batch1_tensor)
                                #clip pred_x_z, but not pred_xz
                                pred_x_z = torch.min(pred_x_z, Variable(torch.FloatTensor([1])))
                                pred_x_z = torch.max(pred_x_z, Variable(torch.FloatTensor([-1])))
                                '''
                                l_z_joint = torch.cat((library, z), dim=1)
                                z_shuffle = np.random.permutation(z.detach().numpy())
                                z_shuffle = Variable(torch.from_numpy(z_shuffle).type(torch.FloatTensor),
                                                     requires_grad=True)
                                l_z_indept = torch.cat((library, z_shuffle), dim=1)
                                pred_xz = self.adv_model(input=l_z_joint)
                                pred_x_z = self.adv_model(input=l_z_indept)

                                if self.adv_model.unbiased_loss:
                                    t = pred_xz
                                    et = torch.exp(pred_x_z)
                                    if self.adv_model.ma_et is None:
                                        self.adv_model.ma_et = torch.mean(et).detach().item()
                                    self.adv_model.ma_et += self.adv_model.ma_rate * (
                                            torch.mean(et).detach().item() - self.adv_model.ma_et)
                                    # unbiasing use moving average
                                    loss_adv2 = -(torch.mean(t) - (
                                            torch.log(torch.mean(et)) * torch.mean(et).detach() / self.adv_model.ma_et))
                                else:
                                    loss_adv = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
                                    loss_adv2 = -loss_adv  # maximizing loss_adv equals minimizing -loss_adv

                                self.model.adv_minibatch_loss = -loss_adv2
                                print('adv_minibatch_MI: %s' % (-loss_adv2))
                                self.adv_optimizer.zero_grad()
                                loss_adv2.backward()
                                self.adv_optimizer.step()
                            elif self.adv_model.name == 'Classifier':
                                z_l = torch.cat((library, z), dim=1)
                                batch_index_adv = Variable(
                                    torch.from_numpy(batch_index_adv.detach().numpy()).type(torch.FloatTensor),
                                    requires_grad=False)
                                logit = self.adv_model(z_l)
                                loss_adv2 = self.adv_criterion(logit, batch_index_adv)
                                self.model.adv_minibatch_loss = loss_adv2
                                print('adv_minibatch_CrossEntropy: %s' % (loss_adv2))
                                self.adv_optimizer.zero_grad()
                                loss_adv2.backward()
                                self.adv_optimizer.step()
                            '''
                            if self.adv_model.name=='MI':
                                if (self.adv_model.save_path != 'None') and (adv_epoch == self.adv_epochs-1) and (l_z_batch0_tensor.shape[0] != 0) and (l_z_batch1_tensor.shape[0] != 0) and (self.epoch % 10 == 0) and (minibatch_index == len(list(self.adv_model.data_loader.data_loaders_loop())) - 2):
                                    activation_hidden = True
                            elif self.adv_model.name=='Classifier':
                                if (self.adv_model.save_path != 'None') and (adv_epoch == self.adv_epochs-1) and (self.epoch % 10 == 0) and (minibatch_index == len(list(self.adv_model.data_loader.data_loaders_loop())) - 2):
                                    activation_hidden = True

                            if activation_hidden == True:
                                activation = {}

                                def get_activation(name):
                                    def hook(model, input, output):
                                        activation[name] = output.detach()

                                    return hook

                                self.adv_model.layers[2].register_forward_hook(get_activation('layer2'))
                                if self.adv_model.name == 'MI':
                                    output0 = self.adv_model(input=l_z_batch0_tensor)
                                elif self.adv_model.name == 'Classifier':
                                    output0 = self.adv_model(z_l)

                                fig = plt.figure(figsize=(14, 7))
                                plt.hist(torch.mean(activation['layer2'],dim=0).squeeze(), density=True, facecolor='g')
                                plt.title("Distribution of activations of nodes in hidden layer2 for the second last minibatch in epoch%s" % (self.epoch))
                                fig.savefig(self.adv_model.save_path + 'Dist_of_activations_layer2_epoch%s.png' % (self.epoch))
                                plt.close(fig)

                                activation_mean_oneepoch = activation_mean_oneepoch + [statistics.mean(torch.mean(activation['layer2'],dim=0).squeeze().tolist())]
                                activation_var_oneepoch = activation_var_oneepoch + [statistics.stdev(torch.mean(activation['layer2'],dim=0).squeeze().tolist())]

                                for k in range(int(self.adv_model.n_hidden_layers / 10)):
                                    self.adv_model.layers[(k + 1) * 10-1].register_forward_hook(get_activation('layer%s' % ((k + 1) * 10-1)))
                                    if self.adv_model.name == 'MI':
                                        output0 = self.adv_model(input=l_z_batch0_tensor)
                                    elif self.adv_model.name == 'Classifier':
                                        output0 = self.adv_model(z_l)

                                    fig = plt.figure(figsize=(14, 7))
                                    plt.hist(torch.mean(activation['layer%s' % ((k + 1) * 10-1)],dim=0).squeeze(), density=True, facecolor='g')
                                    plt.title("Distribution of activations of nodes in hidden layer%s for the second last minibatch in epoch%s" % ((k + 1) * 10-1, self.epoch))
                                    fig.savefig(self.adv_model.save_path + 'Dist_of_activations_layer%s_epoch%s.png' % ((k + 1) * 10-1, self.epoch))
                                    plt.close(fig)

                                    activation_mean_oneepoch = activation_mean_oneepoch + [statistics.mean(torch.mean(activation['layer%s' % ((k + 1) * 10-1)],dim=0).squeeze().tolist())]
                                    activation_var_oneepoch = activation_var_oneepoch + [statistics.mean(torch.mean(activation['layer%s' % ((k + 1) * 10-1)],dim=0).squeeze().tolist())]
                                    print(activation_mean_oneepoch)
                            minibatch_index += 1
                            '''
                    self.adv_model.eval()
                    self.adv_epochs = self.change_adv_epochs
                    '''
                    if len(activation_mean_oneepoch) > 0:
                        activation_mean = np.append(activation_mean, np.array([activation_mean_oneepoch]).transpose(), axis=1)
                        print(activation_mean)
                        print(activation_mean.shape)
                        activation_var = np.append(activation_var, np.array([activation_var_oneepoch]).transpose(), axis=1)
                    '''

                self.model.train()

                if self.model.adv == True and self.model.std == True:
                    for tensors_list in self.data_loaders_loop():
                        loss, ELBO, std_ELBO, penalty, std_penalty = self.loss(*tensors_list)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print(torch.sum(self.model.z_encoder.encoder.fc_layers.Layer1[0].weight.data))

                        ELBO_list.append(ELBO.detach().cpu().numpy())
                        std_ELBO_list.append(std_ELBO.detach().cpu().numpy())
                        penalty_list.append(penalty.detach().cpu().numpy())
                        std_penalty_list.append(std_penalty.detach().cpu().numpy())
                elif self.model.adv == False:
                    for tensors_list in self.data_loaders_loop():
                        if self.model.std == True:
                            loss, ELBO, std_ELBO = self.loss(*tensors_list)
                            ELBO_list.append(ELBO.detach().cpu().numpy())
                            std_ELBO_list.append(std_ELBO.detach().cpu().numpy())
                        else:
                            loss, ELBO = self.loss(*tensors_list)
                            ELBO_list.append(ELBO.detach().cpu().numpy())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print(torch.sum(self.model.z_encoder.encoder.fc_layers.Layer1[0].weight.data))
                if (self.model.save_path != 'None') and (self.epoch % 10 == 0):
                    self.model.eval()
                    with torch.no_grad():
                        asw_train, nmi_train, ari_train, uca_train = self.train_set.clustering_scores()
                        be_train = self.train_set.entropy_batch_mixing()

                        ELBO_train_oneepoch = []
                        number_samples_train = 0
                        for tensors in self.train_set:
                            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                            reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var,
                                                                     batch_index)
                            ELBO = torch.mean(reconst_loss + kl_divergence).detach().cpu().numpy()
                            ELBO_train_oneepoch += [ELBO * sample_batch.shape[0]]
                            number_samples_train += sample_batch.shape[0]
                            ELBO_train = sum(ELBO_train_oneepoch) / number_samples_train

                        asw_test, nmi_test, ari_test, uca_test = self.test_set.clustering_scores()
                        be_test = self.test_set.entropy_batch_mixing()

                        ELBO_test_oneepoch = []
                        number_samples_test = 0
                        for tensors in self.test_set:
                            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
                            reconst_loss, kl_divergence = self.model(sample_batch, local_l_mean, local_l_var,
                                                                     batch_index)
                            ELBO = torch.mean(reconst_loss + kl_divergence).detach().cpu().numpy()
                            ELBO_test_oneepoch += [ELBO * sample_batch.shape[0]]
                            number_samples_test += sample_batch.shape[0]
                            ELBO_test = sum(ELBO_test_oneepoch) / number_samples_test
                        clustermetrics_dataframe_oneepoch = pd.DataFrame.from_dict(
                            {'nth_epochs': [self.epoch, self.epoch], 'Label': ['trainset', 'testset'],
                             'asw': [asw_train, asw_test], 'nmi': [nmi_train, nmi_test], 'ari': [ari_train, ari_test],
                             'uca': [uca_train, uca_test], 'be': [be_train, be_test], 'ELBO': [ELBO_train, ELBO_test]})
                        clustermetrics_trainingprocess = pd.concat(
                            [clustermetrics_trainingprocess, clustermetrics_dataframe_oneepoch], axis=0)
                        clustermetrics_trainingprocess.to_csv(
                            self.model.save_path + 'MIScale%s_clustermetrics_duringtraining.csv' % (
                                int(self.model.MIScale_index)), index=None, header=True)

                if not self.on_epoch_end():
                    break

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print("\nTraining time:  %i s. / %i epochs" % (int(self.training_time), self.n_epochs))

        if self.model.adv == True and self.model.std == True:
            return ELBO_list, std_ELBO_list, penalty_list, std_penalty_list
        if self.model.adv == False and self.model.std == True:
            return ELBO_list, std_ELBO_list
        if self.model.adv == False and self.model.std == False:
            return ELBO_list
