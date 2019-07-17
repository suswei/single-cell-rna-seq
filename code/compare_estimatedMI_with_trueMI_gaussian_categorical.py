import os

if not os.path.isdir('data/compare_estimatedMI_with_trueMI/gaussian_categorical'):
   os.makedirs('data/compare_estimatedMI_with_trueMI/gaussian_categorical')
if not os.path.isdir('result/compare_estimatedMI_with_trueMI/gaussian_categorical'):
   os.makedirs('result/compare_estimatedMI_with_trueMI/gaussian_categorical')

import numpy as np
import pandas as pd
import torch
from scvi.models.modules import MI_Net, discrete_continuous_info, MI_Net4
import itertools
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
import math
import statistics
from sklearn.model_selection._split import _validate_shuffle_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

hyperparameter_config = {
        'method': ['MI_Net','nearest_neighbor','MI_Net4'],
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


def y_pdf_entropy_2(y1, y2):
    gaussian_dim = 2
    y_pdf = 0
    for i in range(len(p_list)):
        y_center = np.array([[y1, y2]]) - np.array([[mu_list[i]] * gaussian_dim])
        cov_mat = ((sigma_list[i]) ** 2) * np.identity(gaussian_dim)
        y_pdf += p_list[i] * ((1 / (math.sqrt((2 * math.pi) ** gaussian_dim) * math.sqrt(np.linalg.det(cov_mat))) * math.exp(
            np.matmul(np.matmul(y_center, np.linalg.inv(cov_mat)), np.transpose(y_center)) * (-0.5))))
    return -y_pdf * math.log(y_pdf)

final_dataframe = pd.DataFrame(columns=['method', 'distribution', 'distribution_dimension', 'gaussian_dimension', 'sample_size','train_size', 'rho', 'true_MI', 'training_or_testing', 'estimated_MI', 'final_loss', 'standard_deviation'])

# distribution of categorical variables
p_array = np.array(
    [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.2, 0.2, 0.3, 0.22]])

# center of the gaussian disribution
mu_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                     [0, 0, 100, 100, 200, 200, 0, 0, 0, 0],
                     [0, 1, 2, 2, 2, 3, 3, 3, 3, 4], [0, 2, 4, 0, 0, 2, 0, 0, 0, 0],
                     [0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                     [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]])

# the cov matrix of the gaussian distribution

sigma_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 20, 20, 40, 40, 1, 1, 1, 1],
                        [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 2, 3, 4, 5, 5, 8, 8, 10, 10], [1, 2, 3, 4, 5, 5, 8, 8, 10, 10]])
true_MI2 = []
for iteration in range(p_array.shape[0]):
    for gaussian_dimension in [2]:
        p_list = p_array[iteration, :]
        cum_p = np.cumsum(p_list)
        mu_list = mu_array[iteration, :]
        sigma_list = sigma_array[iteration, :]

        y_min = min((np.array([mu_list]) - 10 * np.array([sigma_list])).ravel())
        y_max = max((np.array([mu_list]) + 10 * np.array([sigma_list])).ravel())

        if gaussian_dimension == 2:
            y_entropy, y_entropy_err = nquad(y_pdf_entropy_2, [[y_min, y_max], [y_min, y_max]])

        y_given_x_entropy = 0
        for i in range(len(p_list)):
            cov_mat = ((sigma_list[i]) ** 2) * np.identity(gaussian_dimension)
            y_given_x_entropy += p_list[i]*multivariate_normal.entropy(gaussian_dimension * [mu_list[i]], cov_mat)

        true_MI = y_entropy - y_given_x_entropy
        if gaussian_dimension==2:
            true_MI2 += [true_MI]

for taskid in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[taskid].items())
    method = value[0]
    gaussian_dimension = value[1]
    repos = value[2]

    for iteration in range(p_array.shape[0]):
        p_list = p_array[iteration, :]
        cum_p = np.cumsum(p_list)
        mu_list = mu_array[iteration, :]
        sigma_list = sigma_array[iteration, :]

        if method in ['MI_Net','MI_Net4']:
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
            x_dummy = pd.get_dummies(x_dataframe['x']).values
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
            if method=='MI_Net':
                MInet = MI_Net(n_input_nuisance=x_dim, n_input_z=y_dim, n_hidden_z=n_hidden_z, n_layers_z=n_hidden_z)
            elif method=='MI_Net4':
                MInet = MI_Net4(xy_dim=x_dim+y_dim, n_latents=[64,32,16,8,4])

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

                    if method=='MI_Net':
                        pred_xy = MInet(batch_x, batch_y)
                        pred_x_y = MInet(batch_x, batch_y_shuffle)
                    elif method=='MI_Net4':
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
                if method=='MI_Net':
                    pred_xy = MInet(data_x, data_y)
                    pred_x_y = MInet(data_x, data_y_shuffle)
                elif method=='MI_Net4':
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
