import os

if not os.path.isdir('data/compare_MINE_with_trueMI'):
   os.makedirs('data/compare_MINE_with_trueMI')
if not os.path.isdir('result/compare_MINE_with_trueMI'):
   os.makedirs('result/compare_MINE_with_trueMI')



hyperparameter_config = {
        'method': ['MINE','NN'],
        'gaussian_dimension': [2],
        'repos': [100]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

seed = 0
lr = 0.0005
n_epochs = 400
batch_size = 128
n_hidden_z = 10
n_layers_z = 10


for taskid in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[taskid].items())
    method = value[0]
    gaussian_dimension = value[1]
    repos = value[2]

    for iteration in range(p_array.shape[0]):

        if method in ['Mine_Net','Mine_Net4']:
            sample_size = 14388
            train_size = 0.5
            x_array = np.empty((1, sample_size), int)
            y_array = np.empty((gaussian_dimension, sample_size), float)

            for n in range(sample_size):
                index = np.ndarray.tolist(np.random.uniform(size=1) < cum_p).index(True)
                x_array[0, n] = index + 1
                y_array[:, n] = np.random.multivariate_normal([mu_list[index]] * gaussian_dimension,
                                                              ((sigma_list[index]) ** 2) * np.identity(
                                                                  gaussian_dimension), 1)
            x_dataframe = pd.DataFrame.from_dict({'x': np.ndarray.tolist(x_array.ravel())})
            x_dummy = pd.get_dummies(x_dataframe['x']).values ##change categorical variable into dummy variable
            y_array2 = np.transpose(y_array)

            x_dim = x_dummy.shape[1]
            y_dim = y_array2.shape[1]

            dataset2 = np.append(x_dummy, y_array2, axis=1)

            n = len(dataset2)
            n_train, n_test = _validate_shuffle_split(n_samples=n, test_size=None, train_size=train_size)
            np.random.seed(seed=seed)
            permutation = np.random.permutation(n)
            indices_test = permutation[:n_test]
            indices_train = permutation[n_test:(n_test + n_train)]

            training_tensor = Variable(torch.from_numpy(dataset2[indices_train, :]).type(torch.FloatTensor))
            testing_tensor = Variable(torch.from_numpy(dataset2[indices_test, :]).type(torch.FloatTensor))
            if method=='Mine_Net':
                MInet = MINE_Net(n_input_nuisance=x_dim, n_input_z=y_dim, n_hidden_z=n_hidden_z, n_layers_z=n_hidden_z)
            elif method=='Mine_Net4':
                MInet = MINE_Net4(xy_dim=x_dim+y_dim, n_latents=[64,32,16,8,4])

            params = filter(lambda p: p.requires_grad, MInet.parameters())
            optimizer = torch.optim.Adam(params, lr=lr, eps=0.01)
            plot_loss = []
            for epoch in tqdm(range(n_epochs)):

                # X is a torch Variable
                permutation = torch.randperm(training_tensor.size()[0])

                for j in range(0, training_tensor.size()[0], batch_size):
                    j_end = min(j + batch_size, training_tensor.size()[0])
                    if j_end == training_tensor.size()[0]:
                        indices = permutation[j:]
                    else:
                        indices = permutation[j:j_end]

                    batch_x, batch_y = training_tensor[indices, 0:x_dim], training_tensor[indices, x_dim:]

                    batch_x = Variable(batch_x.type(torch.FloatTensor), requires_grad=True)
                    batch_y_shuffle = np.random.permutation(batch_y.detach().numpy())
                    batch_y_shuffle = Variable(torch.from_numpy(batch_y_shuffle).type(torch.FloatTensor),requires_grad=True)

                    if method=='Mine_Net':
                        pred_xy = MInet(batch_x, batch_y)
                        pred_x_y = MInet(batch_x, batch_y_shuffle)
                    elif method=='Mine_Net4':
                        y_x = torch.cat([batch_y,batch_x],dim=1)
                        pred_xy, pred_x_y = MInet(xy=y_x, x_shuffle=batch_y_shuffle, x_n_dim=y_dim) #keep consistent with MI_Net in which the continuous variable is shuffled
                    MI_loss = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
                    loss = -1 * MI_loss
                    plot_loss.append(loss.data.numpy())
                    optimizer.zero_grad()  # clear previous gradients
                    loss.backward()  # compute gradients of all variables wrt loss
                    optimizer.step()  # perform updates using calculated gradients

            MInet.eval()
            final_loss = -np.array(plot_loss).reshape(-1, )[-1]
            training_testing_dict = {'training': training_tensor, 'testing': testing_tensor}
            for type in ['training', 'testing']:
                dataset_tensor = training_testing_dict[type]
                data_x, data_y = dataset_tensor[:, 0:x_dim], dataset_tensor[:, x_dim:]
                data_x = Variable(data_x.type(torch.FloatTensor), requires_grad=True)
                data_y_shuffle = np.random.permutation(data_y.detach().numpy())
                data_y_shuffle = Variable(torch.from_numpy(data_y_shuffle).type(torch.FloatTensor), requires_grad=True)
                if method=='Mine_Net':
                    pred_xy = MInet(data_x, data_y)
                    pred_x_y = MInet(data_x, data_y_shuffle)
                elif method=='Mine_Net4':
                    data_y_x = torch.cat([data_y, data_x], dim=1)
                    pred_xz, pred_x_z = MInet(xy=data_y_x, x_shuffle=data_y_shuffle, x_n_dim=y_dim)
                estimated_MI = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
                estimated_MI = torch.Tensor.cpu(estimated_MI).detach().numpy().item()
                dict = {'method': [method], 'distribution': ['categorical'], 'distribution_dimension': [1],
                        'gaussian_dimension': [gaussian_dimension], 'sample_size': [sample_size],
                        'train_size': [train_size],'rho': [None], 'true_MI': [true_MI2[iteration]], 'training_or_testing': [type],
                        'estimated_MI': [estimated_MI],'final_loss': [None], 'standard_deviation': [None]}
                intermediate_dataframe = pd.DataFrame.from_dict(dict)
                final_dataframe = pd.concat([final_dataframe, intermediate_dataframe])

        elif method == 'nearest_neighbor':
            estimatedMI_NN_list = [] #nn means nearest neighbor method
            sample_size = 128
            for repo in range(repos):
                x_array = np.empty((1, sample_size), int)
                y_array = np.empty((gaussian_dimension, sample_size), float)

                for n in range(sample_size):
                    index = np.ndarray.tolist(np.random.uniform(size=1)<cum_p).index(True)
                    x_array[0,n] = index + 1
                    y_array[:,n] = np.random.multivariate_normal([mu_list[index]]*gaussian_dimension,((sigma_list[index]) ** 2) * np.identity(gaussian_dimension),1)

                one_dc_info_dim = discrete_continuous_info(x_array, y_array, k=3, base=2)

                estimatedMI_NN_list += [one_dc_info_dim]
            mean_estimatedMI_NN = statistics.mean(estimatedMI_NN_list)
            std_estimatedMI_NN = statistics.stdev(estimatedMI_NN_list)
            if gaussian_dimension==2:
                true_MI_selected = true_MI2[iteration]
            dict = {'method': [method], 'distribution': ['categorical'], 'distribution_dimension': [1],
                'gaussian_dimension': [gaussian_dimension], 'sample_size': [sample_size], 'train_size': [None],
                'rho': [None], 'true_MI': [true_MI_selected], 'training_or_testing': [None], 'estimated_MI': [mean_estimatedMI_NN],
                'final_loss': [None],'standard_deviation':[std_estimatedMI_NN]}
            intermediate_dataframe = pd.DataFrame.from_dict(dict)
            final_dataframe = pd.concat([final_dataframe, intermediate_dataframe])

final_dataframe.to_csv('result/compare_estimatedMI_with_trueMI/gaussian_categorical/estimatedMI_with_trueMI.csv', index=None, header=True)
