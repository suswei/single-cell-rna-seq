#the multivariate integration to calculate true_MI when gaussian_dimension=20 for mutual information between
# lognormal and gaussian works very slow. Think of another way.

import os

if not os.path.isdir('data/compare_estimatedMI_with_trueMI/continuous_gaussian'):
   os.makedirs('data/compare_estimatedMI_with_trueMI/continuous_gaussian')
if not os.path.isdir('result/compare_estimatedMI_with_trueMI/continuous_gaussian'):
   os.makedirs('result/compare_estimatedMI_with_trueMI/continuous_gaussian')

import numpy as np
import pandas as pd
import torch
from scvi.models.modules import MINE_Net4
import itertools
from scipy.stats import multivariate_normal,lognorm
from scipy.integrate import nquad
import math
from sklearn.model_selection._split import _validate_shuffle_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def jointpdf_entropy_2(x, y1, y2):
    gaussian_dim = 2
    y_np_center = np.array([[math.log(x)] + [y1, y2]]) - np.array([(gaussian_dim + 1) * [0]])
    cov_mat1 = np.concatenate((np.identity(1), np.array([[1] + (gaussian_dim - 1) * [0]]) * rho), axis=1)
    cov_mat2 = np.concatenate(
        ((np.array([[1] + (gaussian_dim - 1) * [0]]) * rho).transpose(), np.identity(gaussian_dim)), axis=1)
    total_cov = np.concatenate((cov_mat1, cov_mat2), axis=0)
    joint_pdf = (1 / x) * ((
            1 / (math.sqrt((2 * math.pi) ** (gaussian_dim + 1)) * math.sqrt(np.linalg.det(total_cov))) * math.exp(
            np.matmul(np.matmul(y_np_center, np.linalg.inv(total_cov)), np.transpose(y_np_center)) * (-0.5))))
    return -joint_pdf * math.log(joint_pdf)
def dim2_range_x(y1, y2):
    return [math.exp(rho * y1 - 10 * math.sqrt(1 - rho ** 2)), math.exp(rho * y1 + 10 * math.sqrt(1 - rho ** 2))]

def jointpdf_entropy_4(x, y1, y2, y3, y4):
    gaussian_dim = 4
    y_np_center = np.array([[math.log(x)] + [y1, y2, y3, y4]]) - np.array([(gaussian_dim + 1) * [0]])
    cov_mat1 = np.concatenate((np.identity(1), np.array([[1] + (gaussian_dim - 1) * [0]]) * rho), axis=1)
    cov_mat2 = np.concatenate(((np.array([[1] + (gaussian_dim - 1) * [0]]) * rho).transpose(), np.identity(gaussian_dim)), axis=1)
    total_cov = np.concatenate((cov_mat1, cov_mat2), axis=0)
    joint_pdf = (1 / x) * ((1 / (math.sqrt((2 * math.pi) ** (gaussian_dim + 1)) * math.sqrt(np.linalg.det(total_cov))) * math.exp(np.matmul(np.matmul(y_np_center, np.linalg.inv(total_cov)), np.transpose(y_np_center)) * (-0.5))))
    return -joint_pdf * math.log(joint_pdf)
def dim4_range_x(y1, y2, y3, y4):
    return [math.exp(rho * y1 - 10 * math.sqrt(1 - rho ** 2)), math.exp(rho * y1 + 10 * math.sqrt(1 - rho ** 2))]

final_dataframe = pd.DataFrame(columns=['method', 'distribution', 'distribution_dimension', 'gaussian_dimension', 'sample_size','train_size', 'rho', 'true_MI', 'training_or_testing', 'estimated_MI', 'final_loss'])

hyperparameter_config = {
        'distribution': ['lognormal', 'gaussian'],
        'net_name': ['Mine_Net4'],
        'gaussian_dimension': [2, 20],
        'sample_size': [14388],
        'rho': [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

n_epochs = 400
train_size = 0.5
batch_size = 128
seed = 0
layers = [32, 16]

for taskid in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[taskid].items())
    distribution = value[0]
    net_name = value[1]
    gaussian_dimension = value[2]
    sample_size = value[3]
    rho = value[4]

    if distribution == 'lognormal' and gaussian_dimension == 20 :
       continue

    if distribution == 'lognormal':
        continuous_dimension = 1
    elif distribution == 'gaussian':
        continuous_dimension = gaussian_dimension

    lognorm_sigma = 1
    lognorm_mu = 0
    lognorm_rv = lognorm(s=lognorm_sigma, loc=0, scale=math.exp(lognorm_mu))
    lognorm_true_entropy = lognorm.entropy(s=lognorm_sigma, loc=0, scale=math.exp(lognorm_mu))

    gaussian_entropy = multivariate_normal.entropy(gaussian_dimension*[0], np.identity(gaussian_dimension))

    if distribution == 'lognormal':
        if gaussian_dimension == 2:
            if rho in [-0.99,-0.9, 0.9, 0.99]:
                joint_entropy, joint_entropy_err = nquad(jointpdf_entropy_2, [dim2_range_x, [-10, 10], [-10, 10]]) #when rho is large, if x take very small or very large value, jointpdf(x,y1,y2) will be very small, and the logrithm will produce error?
            else:
                joint_entropy, joint_entropy_err = nquad(jointpdf_entropy_2, [[0, math.inf], [-10, 10], [-10, 10]])

        true_MI = lognorm_true_entropy + gaussian_entropy - joint_entropy

        cov_mat1 = np.concatenate((np.identity(continuous_dimension), np.array([[1] + (gaussian_dimension - 1) * [0]]) * rho), axis=1)
        cov_mat2 = np.concatenate(((np.array([[1] + (gaussian_dimension - 1) * [0]]) * rho).transpose(), np.identity(gaussian_dimension)), axis=1)
        total_cov = np.concatenate((cov_mat1, cov_mat2), axis=0)

        np.random.seed(seed)
        dataset1 = np.random.multivariate_normal((gaussian_dimension+continuous_dimension) * [0], total_cov, (sample_size, 1))
        dataset2 = np.reshape(np.ravel(dataset1), (sample_size, gaussian_dimension+continuous_dimension))
        dataset2[:,0] = np.exp(dataset2[:,0])

    elif distribution == 'gaussian':
        cov_mat1 = np.concatenate((np.identity(gaussian_dimension), np.identity(gaussian_dimension)*rho), axis=1)
        cov_mat2 = np.concatenate((np.identity(gaussian_dimension)*rho, np.identity(gaussian_dimension)), axis=1)

        joint_entropy = multivariate_normal.entropy(2*gaussian_dimension*[0], np.concatenate((cov_mat1,cov_mat2),axis=0))

        true_MI = 2*gaussian_entropy - joint_entropy

        np.random.seed(seed)
        dataset1 = np.random.multivariate_normal(2*gaussian_dimension*[0], np.concatenate((cov_mat1,cov_mat2),axis=0),(sample_size,1))
        dataset2 = np.reshape(np.ravel(dataset1), (sample_size, 2*gaussian_dimension))

    n = len(dataset2)
    n_train, n_test = _validate_shuffle_split(n_samples=n, test_size= None, train_size = train_size)
    np.random.seed(seed=seed)
    permutation = np.random.permutation(n)
    indices_test = permutation[:n_test]
    indices_train = permutation[n_test:(n_test + n_train)]

    training_tensor = Variable(torch.from_numpy(dataset2[indices_train,:]).type(torch.FloatTensor))
    testing_tensor = Variable(torch.from_numpy(dataset2[indices_test, :]).type(torch.FloatTensor))

    if net_name == 'Mine_Net4':
        MInet = MINE_Net4(training_tensor.shape[-1], layers)

    if distribution=='gaussian' and gaussian_dimension == 20 and rho in [-0.99, -0.9, 0.9, 0.99]:
        lr = 0.00005
    else:
        lr = 0.0005
    optimizer = torch.optim.Adam(MInet.parameters(), lr=lr)
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
            if distribution == 'lognormal':
               batch_x, batch_y = training_tensor[indices, 0], training_tensor[indices, 1:]
               batch_x_shuffle = np.array([np.random.permutation(batch_x.detach().numpy())]).transpose()

            elif distribution == 'gaussian':
               batch_x, batch_y = training_tensor[indices, 0:gaussian_dimension], training_tensor[indices,gaussian_dimension:]
               batch_x_shuffle = np.random.permutation(batch_x.detach().numpy())

            batch_x_shuffle = Variable(torch.from_numpy(batch_x_shuffle).type(torch.FloatTensor), requires_grad=True)
            batch_y = Variable(batch_y.type(torch.FloatTensor), requires_grad=True)

            if net_name == 'Mine_Net4':
                if distribution == 'lognormal':
                    pred_xy, pred_x_y = MInet(xy=training_tensor[indices, :], x_shuffle=batch_x_shuffle, x_n_dim=1)
                elif distribution == 'gaussian':
                    pred_xy, pred_x_y = MInet(xy=training_tensor[indices, :], x_shuffle=batch_x_shuffle, x_n_dim=gaussian_dimension)

            MI_loss = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            loss = -1 * MI_loss
            plot_loss.append(loss.data.numpy())
            optimizer.zero_grad()  # clear previous gradients
            loss.backward() # compute gradients of all variables wrt loss
            optimizer.step() # perform updates using calculated gradients


    MInet.eval()
    final_loss = -np.array(plot_loss).reshape(-1, )[-1]
    training_testing_dict = {'training': training_tensor, 'testing': testing_tensor}
    for type in ['training','testing']:
        dataset_tensor = training_testing_dict[type]
        if distribution == 'lognormal':
           data_x, data_y = dataset_tensor[:, 0], dataset_tensor[:, 1:]
           data_x_shuffle = np.array([np.random.permutation(data_x.detach().numpy())]).transpose()

        elif distribution == 'gaussian':
           data_x, data_y = dataset_tensor[:, 0:gaussian_dimension], dataset_tensor[:, gaussian_dimension:]
           data_x_shuffle = np.random.permutation(data_x.detach().numpy())

        data_x_shuffle = Variable(torch.from_numpy(data_x_shuffle).type(torch.FloatTensor), requires_grad=True)
        data_y = Variable(data_y.type(torch.FloatTensor), requires_grad=True)

        if net_name == 'Mine_Net4':
            if distribution == 'lognormal':
               data_pred_xy, data_pred_x_y = MInet(xy=dataset_tensor[:, :], x_shuffle=data_x_shuffle, x_n_dim=1)
            elif distribution == 'gaussian':
               data_pred_xy, data_pred_x_y = MInet(xy=dataset_tensor[:, :], x_shuffle=data_x_shuffle, x_n_dim=gaussian_dimension)

        estimated_MI = torch.mean(data_pred_xy) - torch.log(torch.mean(torch.exp(data_pred_x_y)))
        estimated_MI = torch.Tensor.cpu(estimated_MI).detach().numpy().item()

        dict = {'method': [net_name], 'distribution': [distribution], 'distribution_dimension':[continuous_dimension], 'gaussian_dimension':[gaussian_dimension], 'sample_size': [sample_size], 'train_size': [train_size], 'rho': [rho], 'true_MI': [true_MI], 'training_or_testing': [type], 'estimated_MI': [estimated_MI], 'final_loss':[final_loss]}
        intermediate_dataframe = pd.DataFrame.from_dict(dict)
        final_dataframe = pd.concat([final_dataframe,intermediate_dataframe])

final_dataframe.to_csv('result/compare_estimatedMI_with_trueMI/continuous_gaussian/estimatedMI_with_trueMI.csv', index=None, header=True)
