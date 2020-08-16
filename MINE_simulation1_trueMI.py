import os
import numpy as np
import math
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
import pickle

if not os.path.isdir('result/MINE_simulation1'):
   os.makedirs('result/MINE_simulation1')

def y_pdf_entropy(y1, y2):
    gaussian_dim = 2
    y_pdf = 0
    for i in range(len(p_list)):
        y_center = np.array([[y1, y2]]) - np.array([[mu_list[i]] * gaussian_dim])
        cov_mat = ((sigma_list[i]) ** 2) * np.identity(gaussian_dim)
        y_pdf += p_list[i] * ((1 / (math.sqrt((2 * math.pi) ** gaussian_dim) * math.sqrt(np.linalg.det(cov_mat))) * math.exp(
            np.matmul(np.matmul(y_center, np.linalg.inv(cov_mat)), np.transpose(y_center)) * (-0.5))))
    return -y_pdf * math.log(y_pdf)

#The parameters for the 7 cases
# weight of each category for the categorical variable
p_array = np.array(
    [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
     [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.2, 0.2, 0.3, 0.22]])

# mean of the gaussian disribution
mu_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                     [0, 0, 100, 100, 200, 200, 0, 0, 0, 0],
                     [0, 1, 2, 2, 2, 3, 3, 3, 3, 4], [0, 2, 4, 0, 0, 2, 0, 0, 0, 0],
                     [0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                     [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]])

# the sd matrix of the gaussian distribution
sigma_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 20, 20, 40, 40, 1, 1, 1, 1],
                        [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 2, 3, 4, 5, 5, 8, 8, 10, 10], [1, 2, 3, 4, 5, 5, 8, 8, 10, 10]])
trueMI = []
for iteration in range(p_array.shape[0]):
    for gaussian_dimension in [2]:
        p_list = p_array[iteration, :]
        cum_p = np.cumsum(p_list)
        mu_list = mu_array[iteration, :]
        sigma_list = sigma_array[iteration, :]

        y_min = min((np.array([mu_list]) - 10 * np.array([sigma_list])).ravel())
        y_max = max((np.array([mu_list]) + 10 * np.array([sigma_list])).ravel())

        if gaussian_dimension == 2:
            y_entropy, y_entropy_err = nquad(y_pdf_entropy, [[y_min, y_max], [y_min, y_max]])

        y_given_x_entropy = 0
        for i in range(len(p_list)):
            cov_mat = ((sigma_list[i]) ** 2) * np.identity(gaussian_dimension)
            y_given_x_entropy += p_list[i]*multivariate_normal.entropy(gaussian_dimension * [mu_list[i]], cov_mat)

        true_MI = y_entropy - y_given_x_entropy
        if gaussian_dimension==2:
            trueMI += [true_MI]

trueMI_dict = {'trueMI': trueMI}
with open('result/MINE_simulation1/trueMI.pkl', 'wb') as f:
    pickle.dump(trueMI_dict, f)
