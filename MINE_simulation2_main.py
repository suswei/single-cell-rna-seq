import os
import argparse
from random import randint
import pickle

import numpy as np
import math
import pandas as pd
import torch
import torch.optim as optim
from scvi.models.modules import MINE_Net, Nearest_Neighbor_Estimate
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset
from torch.autograd import Variable

def density_ratio(args, x, category_index, component_mean1, component_mean2, KL):
    #sample y and calulate log density ratio of p(y|x) and p(y) to estimate the empirical mutual information
    if x == 0:
        y = MultivariateNormal(component_mean1[category_index, :],
                               torch.eye(args.gaussian_dim)).sample()
    elif x == 1 and args.gaussian_covariance_type == 'all_identity':
        y = MultivariateNormal(component_mean2[category_index, :],
                               torch.eye(args.gaussian_dim)).sample()
    elif x == 1 and args.gaussian_covariance_type == 'partial_identity':
        y = MultivariateNormal(component_mean2[category_index, :],
                               torch.eye(args.gaussian_dim) * torch.tensor((args.mean_diff*0.9)**2)).sample()

    #use log-sum-exp trick to calculate log_density_ratio
    componentwise_log_prob = []
    for comp_index in range(2 * args.mixture_component_num):
        if comp_index < args.mixture_component_num:
            componentwise_log_prob += [MultivariateNormal(component_mean1[comp_index, :],
                                                      torch.eye(args.gaussian_dim)).log_prob(y).item()]
        else:
            if args.gaussian_covariance_type == 'all_identity':
                componentwise_log_prob += [MultivariateNormal(component_mean2[comp_index - args.mixture_component_num, :],
                                                              torch.eye(args.gaussian_dim)).log_prob(y).item()]
            elif args.gaussian_covariance_type == 'partial_identity':
                componentwise_log_prob += [MultivariateNormal(component_mean2[comp_index - args.mixture_component_num, :],
                                       torch.eye(args.gaussian_dim)*torch.tensor((args.mean_diff*0.9)**2)).log_prob(y).item()]

    x0_componentwise_log_prob = componentwise_log_prob[0:args.mixture_component_num]
    x1_componentwise_log_prob = componentwise_log_prob[args.mixture_component_num:]
    max0 = max(x0_componentwise_log_prob)
    max1 = max(x1_componentwise_log_prob)
    max_total = max(componentwise_log_prob)

    if KL == 'joint-marginal':
        log_prob_y = - math.log(2*args.mixture_component_num) + max_total + math.log(sum([math.exp(ele - max_total) for ele in componentwise_log_prob]))
        if x == 0:
            log_density_ratio = - math.log(args.mixture_component_num) + max0 + math.log(
                sum([math.exp(ele - max0) for ele in x0_componentwise_log_prob])) - log_prob_y
        elif x == 1:
            log_density_ratio = - math.log(args.mixture_component_num) + max1 + math.log(
                sum([math.exp(ele - max1) for ele in x1_componentwise_log_prob])) - log_prob_y

        return y, log_density_ratio #for empirical mutual information

    else:
        if KL == 'CD_KL_0_1':
            log_density_ratio = max0 + math.log(
                    sum([math.exp(ele - max0) for ele in x0_componentwise_log_prob]))- max1 - math.log(
                    sum([math.exp(ele - max1) for ele in x1_componentwise_log_prob]))
        elif KL == 'CD_KL_1_0':
            log_density_ratio = max1 + math.log(
                sum([math.exp(ele - max1) for ele in x1_componentwise_log_prob])) - max0 - math.log(
                sum([math.exp(ele - max0) for ele in x0_componentwise_log_prob]))

        return log_density_ratio #for empirical D(p(y|x=0)||p(y|x=1))

def generate_data_MINE_simulation2(args):
    #use random_state to make the p(y|x=0) and p(y|x=1) the same for different MCs in MINE simulation2
    m1 = multivariate_normal(torch.zeros(args.gaussian_dim), torch.eye(args.gaussian_dim))
    component_mean1 = torch.from_numpy(m1.rvs(size=args.mixture_component_num, random_state=8)).type(torch.FloatTensor)

    m2 = multivariate_normal(torch.zeros(args.gaussian_dim) + torch.tensor(args.mean_diff),
                            torch.eye(args.gaussian_dim))
    component_mean2 = torch.from_numpy(m2.rvs(size=args.mixture_component_num, random_state=8)).type(torch.FloatTensor)

    bernoulli = Bernoulli(torch.tensor([0.5])) #assume x=0 and x=1 are with equal probability
    categorical = Categorical(torch.tensor([1/args.mixture_component_num]*args.mixture_component_num))

    log_density_ratio_list = []
    x_tensor = torch.empty(0,1)
    y_tensor = torch.empty(0, args.gaussian_dim)
    for i in range(2*args.samplesize):
        x = bernoulli.sample()
        category_index = categorical.sample()
        KL = 'joint-marginal'
        y, log_density_ratio = density_ratio(args, x.item(), category_index.item(),
                                                        component_mean1, component_mean2, KL)

        x_tensor = torch.cat((x_tensor, x.reshape(1,1)),0)
        y_tensor = torch.cat((y_tensor, y.reshape(1, args.gaussian_dim)),0)
        log_density_ratio_list += [log_density_ratio]

    empirical_mutual_info = sum(log_density_ratio_list) / len(log_density_ratio_list)

    #approximate the empirical D(p(y|x=0) || p(y|x=1)), D means KL-divergence.
    log_density_ratio_list2 = []
    for i in range(2*args.samplesize):
        category_index = categorical.sample()
        KL = 'CD_KL_0_1' #'CD_KL_0_1' means conditional distribution 0 and conditional distribution 1.
        log_density_ratio = density_ratio(args, 0, category_index.item(),
                                                        component_mean1, component_mean2, KL)
        log_density_ratio_list2 += [log_density_ratio]
    empirical_CD_KL_0_1 = sum(log_density_ratio_list2)/len(log_density_ratio_list2)

    log_density_ratio_list3 = []
    for i in range(2 * args.samplesize):
        category_index = categorical.sample()
        KL = 'CD_KL_1_0'
        log_density_ratio = density_ratio(args, 1, category_index.item(),
                                                     component_mean1, component_mean2, KL)
        log_density_ratio_list3 += [log_density_ratio]
    empirical_CD_KL_1_0 = sum(log_density_ratio_list3) / len(log_density_ratio_list3)

    #NN_estimator = discrete_continuous_info(torch.transpose(x_tensor[0:1000, :], 0, 1),
    #                                                     torch.transpose(y_tensor[0:1000, :], 0, 1))
    NN_estimator = Nearest_Neighbor_Estimate(x_tensor[0:2000,:], y_tensor[0:2000,:])

    return empirical_mutual_info, empirical_CD_KL_0_1, empirical_CD_KL_1_0, NN_estimator,  x_tensor, y_tensor

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
            #scheduler_MINE_MI.step(valid_loss_one)
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

    parser.add_argument('--category_num', type=int, default=2,
                        help='the number of batches')

    parser.add_argument('--gaussian_dim', type=int, default=2,
                        help='the dimension of gaussian component for each batch')

    parser.add_argument('--mean_diff', type=int, default=1,
                        help='the mean difference for the two gaussians which generate the mean for the gaussian components')

    parser.add_argument('--mixture_component_num', type=int, default=10,
                        help='the number of gaussian components for the gaussian mixture for each batch')

    parser.add_argument('--gaussian_covariance_type', type=str, default='all_identity',
                        help='the covariance type of each gaussian component in the gaussian mixture for each batch')

    parser.add_argument('--samplesize', type=int, default=6400,
                        help='training sample size')

    parser.add_argument('--n_hidden_node', type=int, default=128,
                        help='number of hidden nodes per hidden layer in MINE')

    parser.add_argument('--n_hidden_layer', type=int, default=10,
                        help='number of hidden layers in MINE')

    parser.add_argument('--activation_fun', type=str, default='Leaky_ReLU',
                        help='activation function in MINE')

    parser.add_argument('--unbiased_loss', action='store_true', default=False,
                        help='whether to use unbiased loss or not in MINE')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for MINE training')

    parser.add_argument('--epochs', type=int, default=200,
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

    if args.activation_fun == 'Leaky_ReLU':
        args.w_initial = 'kaiming_normal'
        args.unbiased_loss_type = 'type1'
    elif args.activation_fun in ['ReLU', 'ELU']:
        args.w_initial = 'normal'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    path = './result/MINE_simulation2/taskid{}'.format(args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    args.path = path

    empirical_mutual_info, empirical_CD_KL_0_1,empirical_CD_KL_1_0, NN_estimator, x_tensor, y_tensor = generate_data_MINE_simulation2(args)
    print('empirical_mutual_info: {}, empirical_CD_KL_0_1: {}, empirical_CD_KL_1_0: {}'.format(
        empirical_mutual_info, empirical_CD_KL_0_1, empirical_CD_KL_1_0))

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, 'MI', kwargs)
    MI_MINE_train, MI_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'MI', args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, 'CD_KL_0_1', kwargs)
    CD_KL_0_1_MINE_train, CD_KL_0_1_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'CD_KL_0_1', args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, 'CD_KL_1_0', kwargs)
    CD_KL_1_0_MINE_train, CD_KL_1_0_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'CD_KL_1_0', args)

    args_dict = vars(args)
    with open('{}/config.pkl'.format(path), 'wb') as f:
        pickle.dump(args_dict, f)

    results = {'empirical_mutual_info': [empirical_mutual_info],
               'empirical_CD_KL_0_1': [empirical_CD_KL_0_1],
               'empirical_CD_KL_1_0': [empirical_CD_KL_1_0],
               'NN_estimator': [NN_estimator],
               'MI_MINE_train': [MI_MINE_train],
               'MI_MINE_test': [MI_MINE_test],
               'CD_KL_0_1_MINE_train': [CD_KL_0_1_MINE_train],
               'CD_KL_0_1_MINE_test': [CD_KL_0_1_MINE_test],
               'CD_KL_1_0_MINE_train': [CD_KL_1_0_MINE_train],
               'CD_KL_1_0_MINE_test': [CD_KL_1_0_MINE_test]
               }
    print(results)
    with open('{}/results.pkl'.format(path), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
