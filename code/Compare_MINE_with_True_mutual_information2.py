import os
import sys
import numpy as np
import pandas as pd
import torch
from scvi.models.modules import MINE_Net, MINE_Net2, MINE_Net3, MINE_Net4
import itertools
from scipy.stats import multivariate_normal
from sklearn.model_selection._split import _validate_shuffle_split
from torch.autograd import Variable
from tqdm import tqdm

hyperparameter_config = {
        'Net_Name': ['Mine_Net2', 'Mine_Net3', 'Mine_Net4'],
        'Gaussian_Dimension': [2, 20],
        'sample_size': [14388],
        'rho': [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

for taskid in range(13):
    if not os.path.exists('data/Tune_Hyperparameter_For_Minenet/2019-06-04'):
        os.makedirs('data/Tune_Hyperparameter_For_Minenet/2019-06-04')
    if not os.path.exists('result/Tune_Hyperparameter_For_Minenet/2019-06-04'):
        os.makedirs('result/Tune_Hyperparameter_For_Minenet/2019-06-04')

    n_epochs_all = None
    n_epochs = 400 if n_epochs_all is None else n_epochs_all
    lr = 0.0005
    train_size = 0.5
    batch_size = 128
    seed = 0

    final_dataframe = pd.DataFrame(columns=['variable_dimension', 'sample_size', 'rho', 'true_MI', 'estimated_training_MI', 'estimated_testing_MI','final_loss'])
    for i in [taskid*2, taskid*2+1]:
        key, value = zip(*hyperparameter_experiments[i].items())
        Net_Name = value[0]
        Gaussian_Dimension = value[1]
        sample_size = value[2]
        rho = value[3]


        entropy1 = multivariate_normal.entropy(Gaussian_Dimension*[0], np.identity(Gaussian_Dimension))
        entropy2 = multivariate_normal.entropy(Gaussian_Dimension*[0], np.identity(Gaussian_Dimension))

        cov_mat1 = np.concatenate((np.identity(Gaussian_Dimension), np.identity(Gaussian_Dimension)*rho), axis=1)
        cov_mat2 = np.concatenate((np.identity(Gaussian_Dimension)*rho, np.identity(Gaussian_Dimension)), axis=1)

        joint_entropy = multivariate_normal.entropy(2*Gaussian_Dimension*[0], np.concatenate((cov_mat1,cov_mat2),axis=0))

        true_MI = entropy1 + entropy2 - joint_entropy

        np.random.seed(seed)
        dataset1 = np.random.multivariate_normal(2*Gaussian_Dimension*[0], np.concatenate((cov_mat1,cov_mat2),axis=0),(sample_size,1))
        dataset2 = np.reshape(np.ravel(dataset1), (sample_size, 2*Gaussian_Dimension))

        n = len(dataset2)
        n_train, n_test = _validate_shuffle_split(n_samples=n, test_size= None, train_size = train_size)
        np.random.seed(seed=seed)
        permutation = np.random.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test:(n_test + n_train)]

        training_tensor = Variable(torch.from_numpy(dataset2[indices_train,:]).type(torch.FloatTensor))
        testing_tensor = Variable(torch.from_numpy(dataset2[indices_test, :]).type(torch.FloatTensor))
        if Net_Name == 'Mine_Net2':
            minenet = MINE_Net2(n_input_nuisance=Gaussian_Dimension, n_input_z=Gaussian_Dimension, n_hidden_z=10, n_layers_z=10)
        elif Net_Name == 'Mine_Net3':
            minenet = MINE_Net3(n_input_nuisance= Gaussian_Dimension, n_input_z= Gaussian_Dimension, H=10)
        elif Net_Name == 'Mine_Net4':
            layers = [32, 16]
            minenet = MINE_Net4(training_tensor.shape[-1], layers)

        optimizer = torch.optim.Adam(minenet.parameters(), lr=lr)
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

                batch_x, batch_y = training_tensor[indices, 0:Gaussian_Dimension], training_tensor[indices, Gaussian_Dimension:]

                batch_x_shuffle = np.random.permutation(batch_x.detach().numpy())
                batch_x_shuffle = Variable(torch.from_numpy(batch_x_shuffle).type(torch.FloatTensor), requires_grad=True)
                batch_y = Variable(batch_y.type(torch.FloatTensor), requires_grad=True)

                if Net_Name in ['Mine_Net2', 'Mine_Net3']:
                    pred_xy = minenet(batch_y, batch_x)
                    pred_x_y = minenet(batch_y, batch_x_shuffle)
                elif Net_Name == 'Mine_Net4':
                    pred_xy, pred_x_y = minenet(xy=training_tensor[indices, :], x=batch_x_shuffle, n_dim=Gaussian_Dimension)
                mine_loss = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
                loss = -1 * mine_loss
                plot_loss.append(loss.data.numpy())
                minenet.zero_grad()  # clear previous gradients
                loss.backward() # compute gradients of all variables wrt loss
                optimizer.step() # perform updates using calculated gradients

        minenet.eval()
        train_x, train_y = training_tensor[:, 0:Gaussian_Dimension], training_tensor[:, Gaussian_Dimension:]

        train_x_shuffle = np.random.permutation(train_x.detach().numpy())
        train_x_shuffle = Variable(torch.from_numpy(train_x_shuffle).type(torch.FloatTensor), requires_grad=True)
        train_y = Variable(train_y.type(torch.FloatTensor), requires_grad=True)

        if Net_Name in ['Mine_Net2', 'Mine_Net3']:
            train_pred_xy = minenet(train_y, train_x)
            train_pred_x_y = minenet(train_y, train_x_shuffle)
        elif Net_Name == 'Mine_Net4':
            train_pred_xy, train_pred_x_y = minenet(xy=training_tensor[:, :], x=train_x_shuffle, n_dim=Gaussian_Dimension)

        estimated_training_MI = torch.mean(train_pred_xy) - torch.log(torch.mean(torch.exp(train_pred_x_y)))
        estimated_training_MI = torch.Tensor.cpu(estimated_training_MI).detach().numpy().item()

        test_x, test_y = testing_tensor[:, 0:Gaussian_Dimension], training_tensor[:, Gaussian_Dimension:]

        test_x_shuffle = np.random.permutation(test_x.detach().numpy())
        test_x_shuffle = Variable(torch.from_numpy(test_x_shuffle).type(torch.FloatTensor), requires_grad=True)
        test_y = Variable(test_y.type(torch.FloatTensor), requires_grad=True)

        if Net_Name in ['Mine_Net2', 'Mine_Net3']:
            test_pred_xy = minenet(test_y, test_x)
            test_pred_x_y = minenet(test_y, test_x_shuffle)
        elif Net_Name == 'Mine_Net4':
            test_pred_xy, test_pred_x_y = minenet(xy=testing_tensor[:, :], x=test_x_shuffle, n_dim=Gaussian_Dimension)

        estimated_testing_MI = torch.mean(test_pred_xy) - torch.log(torch.mean(torch.exp(test_pred_x_y)))
        estimated_testing_MI = torch.Tensor.cpu(estimated_testing_MI).detach().numpy().item()

        final_loss = -np.array(plot_loss).reshape(-1, )[-1]

        dict = {'variable_dimension': [Gaussian_Dimension],'sample_size':[sample_size], 'rho':[rho], 'true_MI': [true_MI], 'estimated_training_MI': [estimated_training_MI], 'estimated_testing_MI': [estimated_testing_MI], 'final_loss':[final_loss]}
        intermediate_dataframe = pd.DataFrame.from_dict(dict)
        final_dataframe = pd.concat([final_dataframe,intermediate_dataframe])
    if Net_Name == 'Mine_Net2':
        final_dataframe.to_csv('result/Tune_Hyperparameter_For_Minenet/2019-06-04/Taskid%s_Compare2_mine_estimator_with_true_MI.csv'%(taskid), index=None, header=True)
    elif Net_Name == 'Mine_Net3':
        final_dataframe.to_csv('result/Tune_Hyperparameter_For_Minenet/2019-06-04/Taskid%s_Compare3_mine_estimator_with_true_MI.csv' % (taskid), index=None, header=True)
    elif Net_Name == 'Mine_Net4':
        final_dataframe.to_csv('result/Tune_Hyperparameter_For_Minenet/2019-06-04/Taskid%s_Compare4_mine_estimator_with_true_MI.csv' % ( taskid), index=None, header=True)


