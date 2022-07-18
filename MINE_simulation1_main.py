import os
import argparse
from random import randint
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scvi.models.modules import MINE_Net, Nearest_Neighbor_Estimate
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset
from torch.autograd import Variable
#import plotly.graph_objects as go

def density_ratio(args, x, p_list, mu_list, sigma_list):

    y = MultivariateNormal(torch.FloatTensor([mu_list[x].item()] * args.gaussian_dim),
                           ((sigma_list[x]) ** 2) * torch.eye(args.gaussian_dim)).sample()
    log_prob_y_x = MultivariateNormal(torch.FloatTensor([mu_list[x].item()] * args.gaussian_dim),
                           ((sigma_list[x]) ** 2) * torch.eye(args.gaussian_dim)).log_prob(y).item()
    componentwise_log_prob = []
    for i in range(args.category_num):
        log_prob_gaussian = MultivariateNormal(torch.FloatTensor([mu_list[i].item()] * args.gaussian_dim),
                           ((sigma_list[x]) ** 2) * torch.eye(args.gaussian_dim)).log_prob(y).item()
        log_prob_category = math.log(p_list[i].item())
        componentwise_log_prob += [log_prob_gaussian + log_prob_category]

    max_log_prob = max(componentwise_log_prob)
    log_prob_y = max_log_prob + math.log(sum([math.exp(ele - max_log_prob) for ele in componentwise_log_prob]))
    log_density_ratio = log_prob_y_x - log_prob_y
    return y, log_density_ratio

def generate_data_MINE_simulation1(args):
    # The parameters for the 7 cases
    # weight of each category for the categorical variable
    p_tensor = torch.from_numpy(np.array(
        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
         [0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.2, 0.2, 0.3, 0.22]])).type(torch.FloatTensor)

    # mean of the gaussian disribution
    mu_tensor = torch.from_numpy(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                                           [0, 0, 100, 100, 200, 200, 0, 0, 0, 0], [0, 1, 2, 2, 2, 3, 3, 3, 3, 4],
                                           [0, 2, 4, 0, 0, 2, 0, 0, 0, 0], [0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                                           [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]])).type(torch.FloatTensor)

    # the sd matrix of the gaussian distribution
    sigma_tensor = torch.from_numpy(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                              [1, 1, 20, 20, 40, 40, 1, 1, 1, 1], [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 5, 8, 8, 10, 10],
                                              [1, 2, 3, 4, 5, 5, 8, 8, 10, 10]])).type(torch.FloatTensor)

    p_list = p_tensor[args.case_idx, :]
    mu_list = mu_tensor[args.case_idx, :]
    sigma_list = sigma_tensor[args.case_idx, :]

    x_tensor = torch.empty(0, 1)
    y_tensor = torch.empty(0, args.gaussian_dim)

    categorical = Categorical(p_list)

    log_density_ratio_list = []
    for i in range(2 * args.samplesize):
        x = categorical.sample()

        y, log_density_ratio = density_ratio(args, x.item(),p_list, mu_list, sigma_list)
        log_density_ratio_list.append(log_density_ratio)

        x_tensor = torch.cat((x_tensor, x.reshape(1, 1).type(torch.FloatTensor)), 0)
        y_tensor = torch.cat((y_tensor, y.reshape(1, args.gaussian_dim)), 0)

    empirical_MI = sum(log_density_ratio_list)/len(log_density_ratio_list)
    #NN_estimator = discrete_continuous_info(torch.transpose(x_tensor[0:1000, :], 0, 1),
    #                                        torch.transpose(y_tensor[0:1000, :], 0, 1))
    NN_estimator = Nearest_Neighbor_Estimate(x_tensor[0:2000,:], y_tensor[0:2000,:])

    return empirical_MI, NN_estimator, x_tensor, y_tensor

def train_valid_test_loader(x_tensor, y_tensor, args, KL_type, kwargs):

    if KL_type == 'MI':
        x_dataframe = pd.DataFrame.from_dict({'category': np.ndarray.tolist(x_tensor.numpy().ravel())})
        x_tensor = torch.from_numpy(pd.get_dummies(x_dataframe['category']).values).type(torch.FloatTensor)

    train_size = args.samplesize
    valid_size = int(args.samplesize * 0.5)
    test_size = 2 * args.samplesize - train_size - valid_size
    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(y_tensor, x_tensor),
                                                                               [train_size, valid_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader

def sample1_sample2_from_minibatch(args, KL_type, x, y):

    if args.cuda:
        x, y =  x.cuda(), y.cuda()
    else:
        x, y =  Variable(x, requires_grad=True), Variable(y, requires_grad=True),

    if KL_type == 'MI':
        y_x = torch.cat((y, x), 1) #joint
        shuffle_index = torch.randperm(y.shape[0])
        shuffle_y_x = torch.cat((y[shuffle_index], x), 1) #marginal
        return y_x, shuffle_y_x
    else:
        #make the training sample size equals batchsize*some_integer
        #such that y_x0 and y_x1 will not be empty.
        y_x0 = y[(x[:, 0] == 0).nonzero().squeeze(1)]
        y_x1 = y[(x[:, 0] == 1).nonzero().squeeze(1)]

        if KL_type == 'CD_KL_0_1':
            return y_x0, y_x1
        elif KL_type == 'CD_KL_1_0':
            return y_x1, y_x0

def MINE_unbiased_loss(ma_et, t, et):

    loss = -(torch.mean(t) - (1 / ma_et) * torch.mean(et))
    # loss can also be calculated in the following way, it is the same as above
    #loss = -(torch.mean(t) - torch.log(torch.mean(et)) * torch.mean(et).detach() / ma_et)

    return loss

def MINE_train(train_loader, valid_loader, test_loader, KL_type, args):

    #MINE_MI uses MINE to estimate mutual information between y and x, where x changes into dummy variable
    #MINE_CD_KL uses MINE to estimate D(p(y|x=0)||p(y|x=1)) in MINE_simulation2, both networks use unbiased_loss=True

    if KL_type == 'MI':
        input_dim = args.gaussian_dim + args.category_num
    else:
        input_dim = args.gaussian_dim
    MINE = MINE_Net(input_dim=input_dim, n_hidden=args.n_hidden_node, n_layers=args.n_hidden_layer,
                    activation_fun=args.activation_fun, unbiased_loss=args.unbiased_loss, initial=args.w_initial)

    opt_MINE = optim.Adam(MINE.parameters(), lr=args.lr)
    #scheduler_MINE_MI = torch.optim.lr_scheduler.MultiStepLR(opt_MINE, milestones=[200], gamma=0.5)

    MINE_estimator_minibatch_list, negative_loss_minibatch_list, valid_loss_epoch, train_loss_epoch = [], [], [], []
    for epoch in range(args.epochs):

        MINE.train()

        for batch_idx, (y, x) in enumerate(train_loader):

            sample1, sample2 = sample1_sample2_from_minibatch(args, KL_type, x, y)
            t = MINE(sample1)
            et = torch.exp(MINE(sample2))

            if args.unbiased_loss:
                if MINE.ma_et is None:
                    MINE.ma_et = torch.mean(et).detach().item() #detach means will not calculate gradient for ma_et, ma_et is just a number
                MINE.ma_et = (1 - MINE.ma_rate) * MINE.ma_et + MINE.ma_rate * torch.mean(et).detach().item()

                #Pay attention, even if use unbiased_loss, this unbiased loss is not our MINE estimator,
                #The MINE estimator is still torch.mean(t) - torch.log(torch.mean(et)) after training
                #the unbiased_loss is only for getting unbiased gradient.
                loss = MINE_unbiased_loss(MINE.ma_et, t, et)
            else:
                loss = -(torch.mean(t) - torch.log(torch.mean(et)))

            MINE_estimator_minibatch = torch.mean(t) - torch.log(torch.mean(et))

            opt_MINE.zero_grad()
            loss.backward()
            opt_MINE.step()

            #diagnosis
            MINE_estimator_minibatch_list.append(MINE_estimator_minibatch.item())
            # draw the loss together with MINE estimator, therefore change loss into negative of loss
            #if not using unbiased loss, then negative of loss equals MINE estimator
            negative_loss_minibatch_list.append(-loss.item())

        # scheduler_MINE_MI.step()
        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tMINE_estimator_minibatch: {:.6f}\tnegative_loss_minibatch: {:.6f}'.format(epoch, MINE_estimator_minibatch.data.item(), -loss.data.item()))
        '''
        # diagnoisis of overfitting
        with torch.no_grad():  # to save memory, no intermediate activations used for gradient calculation is stored.

            MINE.eval()

            valid_loss_minibatch_list = []
            for valid_batch_idx, (valid_y, valid_x) in enumerate(valid_loader):

                valid_sample1, valid_sample2 = sample1_sample2_from_minibatch(args, KL_type, valid_x, valid_y)
                valid_t = MINE(valid_sample1)
                valid_et = torch.exp(MINE(valid_sample2))

                if args.unbiased_loss == True:
                    valid_loss_minibatch = MINE_unbiased_loss(MINE.ma_et, valid_t, valid_et)
                else:
                    valid_loss_minibatch = -(torch.mean(valid_t) - torch.log(torch.mean(valid_et)))

                valid_loss_minibatch_list.append(valid_loss_minibatch.item())

            valid_loss_one = np.average(valid_loss_minibatch_list)
            scheduler_MINE_MI.step(valid_loss_one)
            valid_loss_epoch.append(valid_loss_one)

            train_loss_minibatch_list = []
            for train_batch_idx, (train_y, train_x) in enumerate(train_loader):

                train_sample1, train_sample2 = sample1_sample2_from_minibatch(args, KL_type, train_x, train_y)
                train_t = MINE(train_sample1)
                train_et = torch.exp(MINE(train_sample2))

                if args.unbiased_loss == True:
                    train_loss_minibatch = MINE_unbiased_loss(MINE.ma_et, train_t, train_et)
                else:
                    train_loss_minibatch = -(torch.mean(train_t) - torch.log(torch.mean(train_et)))

                train_loss_minibatch_list.append(train_loss_minibatch.item())

            train_loss_one = np.average(train_loss_minibatch_list)
            train_loss_epoch.append(train_loss_one)
        '''
    with torch.no_grad():
        MINE.eval()
        train_y_all = torch.empty(0, args.gaussian_dim)
        if KL_type == 'MI':
            train_x_all = torch.empty(0, args.category_num)
        else:
            train_x_all = torch.empty(0, 1)

        for train_batch_idx, (train_y, train_x) in enumerate(train_loader):
            train_y_all = torch.cat((train_y_all, train_y), 0)
            train_x_all = torch.cat((train_x_all, train_x), 0)

        train_sample1_all, train_sample2_all = sample1_sample2_from_minibatch(args, KL_type, train_x_all, train_y_all)
        train_t_all = MINE(train_sample1_all)
        train_et_all = torch.exp(MINE(train_sample2_all))
        train_MINE_estimator = torch.mean(train_t_all) - torch.log(torch.mean(train_et_all))

        test_y_all = torch.empty(0, args.gaussian_dim)
        if KL_type == 'MI':
            test_x_all = torch.empty(0, args.category_num)
        else:
            test_x_all = torch.empty(0, 1)
        for test_batch_idx, (test_y, test_x) in enumerate(test_loader):
            test_y_all = torch.cat((test_y_all, test_y), 0)
            test_x_all = torch.cat((test_x_all, test_x), 0)

        test_sample1_all, test_sample2_all = sample1_sample2_from_minibatch(args, KL_type, test_x_all, test_y_all)
        test_t_all = MINE(test_sample1_all)
        test_et_all = torch.exp(MINE(test_sample2_all))
        test_MINE_estimator = torch.mean(test_t_all) - torch.log(torch.mean(test_et_all))

    return train_MINE_estimator.detach().item(), test_MINE_estimator.detach().item()

def main():
    parser = argparse.ArgumentParser(description='MINE Simulation Study')

    parser.add_argument('--taskid', type=int, default=1000 + randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--case_idx', type=int, default=0,
                        help='which case in the designed 7 cases')

    parser.add_argument('--category_num', type=int, default=10,
                        help='the number of category for the categorical variable')

    parser.add_argument('--gaussian_dim', type=int, default=2,
                        help='the dimension of gaussian component conditioning on each category')

    parser.add_argument('--samplesize', type=int, default=25600,
                        help='training sample size')

    parser.add_argument('--n_hidden_node', type=int, default=128,
                        help='number of hidden nodes per hidden layer in MINE')

    parser.add_argument('--n_hidden_layer', type=int, default=10,
                        help='number of hidden layers in MINE')

    parser.add_argument('--activation_fun', type=str, default='ELU',
                        help='activation function in MINE')

    parser.add_argument('--unbiased_loss', action='store_true', default=False,
                        help='whether to use unbiased loss or not in MINE')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for MINE training')

    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs in MINE training')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate in MINE training')

    parser.add_argument('--MC', type=int, default=0,
                        help='the index to show which repeation')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before logging training status')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("args.cuda is " + str(args.cuda))

    if args.activation_fun in ['ELU']:
        args.w_initial = 'normal'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    empirical_MI, NN_estimator, x_tensor, y_tensor = generate_data_MINE_simulation1(args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, 'MI', kwargs)
    MI_MINE_train, MI_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'MI', args)

    estimated_results = {'empirical_MI': [empirical_MI],
                         'NN_estimator': [NN_estimator],
                         'MI_MINE_train': [MI_MINE_train],
                         'MI_MINE_test': [MI_MINE_test],
                         }

    if not os.path.isdir('result/MINE_simulation1/taskid{}'.format(args.taskid)):
        os.makedirs('result/MINE_simulation1/taskid{}'.format(args.taskid))
    args.save_path = 'result/MINE_simulation1/taskid{}'.format(args.taskid)

    args_dict = vars(args)
    with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(args_dict, f)

    with open('{}/results.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(estimated_results, f)
    print(estimated_results)
if __name__ == "__main__":
    main()
