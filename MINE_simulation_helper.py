import torch
from scvi.models.modules import discrete_continuous_info
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
#on spartan, as the available version of plotly is 3.2.1, plotly.graph_objects is plotly.graph_objs

def discrete_z_density_ratio(args, batch, category_index, component_mean1, component_mean2, KL):
    #sample z and calulate log density ratio of p(z|s) and p(z) to estimate the empirical mutual information
    if batch == 0:
        z = MultivariateNormal(component_mean1[category_index, :],
                               torch.eye(args.gaussian_dim)).sample()
    elif batch == 1 and args.gaussian_covariance_type == 'all_identity':
        z = MultivariateNormal(component_mean2[category_index, :],
                               torch.eye(args.gaussian_dim)).sample()
    elif batch == 1 and args.gaussian_covariance_type == 'partial_identity':
        z = MultivariateNormal(component_mean2[category_index, :],
                               torch.eye(args.gaussian_dim) * torch.tensor((args.mean_diff*0.9)**2)).sample()

    #use log-sum-exp trick to calculate log_density_ratio
    componentwise_log_prob = []
    for comp_index in range(2 * args.mixture_component_num):
        if comp_index < args.mixture_component_num:
            componentwise_log_prob += [MultivariateNormal(component_mean1[comp_index, :],
                                                      torch.eye(args.gaussian_dim)).log_prob(z).item()]
        else:
            if args.gaussian_covariance_type == 'all_identity':
                componentwise_log_prob += [MultivariateNormal(component_mean2[comp_index - args.mixture_component_num, :],
                                                              torch.eye(args.gaussian_dim)).log_prob(z).item()]
            elif args.gaussian_covariance_type == 'partial_identity':
                componentwise_log_prob += [MultivariateNormal(component_mean2[comp_index - args.mixture_component_num, :],
                                       torch.eye(args.gaussian_dim)*torch.tensor((args.mean_diff*0.9)**2)).log_prob(z).item()]

    batch0_componentwise_log_prob = componentwise_log_prob[0:args.mixture_component_num]
    batch1_componentwise_log_prob = componentwise_log_prob[args.mixture_component_num:]
    max0 = max(batch0_componentwise_log_prob)
    max1 = max(batch1_componentwise_log_prob)
    max_total = max(componentwise_log_prob)

    if KL == 'joint-marginal':
        log_prob_z = - math.log(2*args.mixture_component_num) + max_total + math.log(sum([math.exp(ele - max_total) for ele in componentwise_log_prob]))
        if batch == 0:
            log_density_ratio = - math.log(args.mixture_component_num) + max0 + math.log(
                sum([math.exp(ele - max0) for ele in batch0_componentwise_log_prob])) - log_prob_z
        elif batch == 1:
            log_density_ratio = - math.log(args.mixture_component_num) + max1 + math.log(
                sum([math.exp(ele - max1) for ele in batch1_componentwise_log_prob])) - log_prob_z

        return z, log_density_ratio #for empirical mutual information

    else:
        if KL == 'CD_KL_0_1':
            log_density_ratio = max0 + math.log(
                    sum([math.exp(ele - max0) for ele in batch0_componentwise_log_prob]))- max1 - math.log(
                    sum([math.exp(ele - max1) for ele in batch1_componentwise_log_prob]))
        elif KL == 'CD_KL_1_0':
            log_density_ratio = max1 + math.log(
                sum([math.exp(ele - max1) for ele in batch1_componentwise_log_prob])) - max0 - math.log(
                sum([math.exp(ele - max0) for ele in batch0_componentwise_log_prob]))

        return log_density_ratio #for empirical D(p(z|s=0)||p(z|s=1))


def generate_data_MINE_simulation(args):

    if args.confounder_type == 'discrete':

        #use random_state to make the p(z|s=0) and p(z|s=1) the same for different MCs
        m1 = multivariate_normal(torch.zeros(args.gaussian_dim), torch.eye(args.gaussian_dim))
        component_mean1 = torch.from_numpy(m1.rvs(size=args.mixture_component_num, random_state=8)).type(torch.FloatTensor)

        m2 = multivariate_normal(torch.zeros(args.gaussian_dim) + torch.tensor(args.mean_diff),
                                torch.eye(args.gaussian_dim))
        component_mean2 = torch.from_numpy(m2.rvs(size=args.mixture_component_num, random_state=8)).type(torch.FloatTensor)

        bernoulli = Bernoulli(torch.tensor([0.5])) #assume two batches are with equal probability
        categorical = Categorical(torch.tensor([1/args.mixture_component_num]*args.mixture_component_num))

        log_density_ratio_list = []
        batch_tensor = torch.empty(0,1)
        z_tensor = torch.empty(0, args.gaussian_dim)
        for i in range(2*args.samplesize):
            batch = bernoulli.sample()
            category_index = categorical.sample()
            KL = 'joint-marginal'
            z, log_density_ratio = discrete_z_density_ratio(args, batch.item(), category_index.item(),
                                                            component_mean1, component_mean2, KL)

            batch_tensor = torch.cat((batch_tensor, batch.reshape(1,1)),0)
            z_tensor = torch.cat((z_tensor, z.reshape(1, args.gaussian_dim)),0)
            log_density_ratio_list += [log_density_ratio]

        empirical_mutual_info = sum(log_density_ratio_list) / len(log_density_ratio_list)

        #approximate the empirical D(p(z|s=0) || p(z|s=1)), D means KL-divergence.
        log_density_ratio_list2 = []
        for i in range(2*args.samplesize):
            category_index = categorical.sample()
            KL = 'CD_KL_0_1' #'CD_KL_0_1' means conditional distribution 0 and conditional distribution 1.
            log_density_ratio = discrete_z_density_ratio(args, 0, category_index.item(),
                                                            component_mean1, component_mean2, KL)
            log_density_ratio_list2 += [log_density_ratio]
        empirical_CD_KL_0_1 = sum(log_density_ratio_list2)/len(log_density_ratio_list2)

        log_density_ratio_list3 = []
        for i in range(2 * args.samplesize):
            category_index = categorical.sample()
            KL = 'CD_KL_1_0'
            log_density_ratio = discrete_z_density_ratio(args, 1, category_index.item(),
                                                         component_mean1, component_mean2, KL)
            log_density_ratio_list3 += [log_density_ratio]
        empirical_CD_KL_1_0 = sum(log_density_ratio_list3) / len(log_density_ratio_list3)

        #The dimension for batch_array is (1, 2*hyperparameters['samplesize'])
        #The dimension for z_array is (hyperparameters['gaussian_dim'], 2*hyperparameters['samplesize'])
        #discrete_continuous_info() will be very slow if sample size is large.
        #Therefore, only 256 datapoints are used for nearest neighbor estimate, 256 could be a minibatch size.

        nearest_neighbor_estimate = discrete_continuous_info(torch.transpose(batch_tensor[0:256, :], 0, 1),
                                                             torch.transpose(z_tensor[0:256, :], 0, 1))


        return empirical_mutual_info, empirical_CD_KL_0_1, empirical_CD_KL_1_0, nearest_neighbor_estimate, z_tensor, batch_tensor

def train_valid_test_loader(z_tensor, batch_tensor, model_type, args, kwargs):

    if model_type == 'MI':
        batch_dataframe = pd.DataFrame.from_dict({'batch': np.ndarray.tolist(batch_tensor.numpy().ravel())})
        batch_tensor = torch.from_numpy(pd.get_dummies(batch_dataframe['batch']).values).type(torch.FloatTensor)

    train_size = args.samplesize
    valid_size = int(args.samplesize * 0.5)
    test_size = 2 * args.samplesize - train_size - valid_size
    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(TensorDataset(z_tensor, batch_tensor),
                                                                               [train_size, valid_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchsize, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader

def sample1_sample2_from_minibatch(args, model_type, z, batch):
    if args.cuda:
        z, batch = z.cuda(), batch.cuda()
    else:
        z, batch = Variable(z, requires_grad=True), Variable(batch, requires_grad=True)

    if model_type == 'MI':
        z_batch = torch.cat((z, batch), 1) #joint
        shuffle_index = torch.randperm(z.shape[0])
        shuffle_z_batch = torch.cat((z[shuffle_index], batch), 1) #marginal
        return z_batch, shuffle_z_batch
    else:
        #make the training sample size equals batchsize*some_integer
        #such that z_batch0 and z_batch1 will not be empty.
        z_batch0 = z[(batch[:, 0] == 0).nonzero().squeeze(1)]
        z_batch1 = z[(batch[:, 0] == 1).nonzero().squeeze(1)]

        if model_type == 'CD_KL_0_1':
            return z_batch0, z_batch1
        elif model_type == 'CD_KL_1_0':
            return z_batch1, z_batch0

def MINE_unbiased_loss(args, ma_et, t, et):
    if args.unbiased_loss_type == 'type1':
        loss = -(torch.mean(t) - torch.log(torch.mean(et)) * torch.mean(et).detach() / ma_et)
    elif args.unbiased_loss_type == 'type2':
        loss = -(torch.mean(t) - (1 / ma_et) * torch.mean(et))

    return loss

def diagnosis_loss_plot(args, model_type, MINE_estimator_minibatch_list, negative_loss_minibatch_list, valid_loss_epoch, train_loss_epoch):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(0, len(MINE_estimator_minibatch_list))),
                             y=MINE_estimator_minibatch_list,
                             mode='lines+markers', name='MINE estimator (minibatch)'))
    fig.add_trace(go.Scatter(x=list(range(0, len(negative_loss_minibatch_list))),
                             y=negative_loss_minibatch_list,
                             mode='lines+markers', name='negative train loss (minibatch)'))

    fig.update_xaxes(title_text='epochs * minibatches')
    fig.update_layout(
        title={'text': 'MINE estimator and negative train loss per minibatch <br> unbiased_loss: {}'.format(args.unbiased_loss),
               'y': 0.9,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
    fig.write_image('{}/{}_MINE_negative_loss_per_minibatch.png'.format(args.path, model_type))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(0, args.epochs)),
                             y=train_loss_epoch,
                             mode='lines+markers', name='train loss (epoch)'))
    fig.add_trace(go.Scatter(x=list(range(0, args.epochs)),
                             y=valid_loss_epoch,
                             mode='lines+markers', name='valid loss (epoch)'))

    fig.update_xaxes(title_text='epochs')
    fig.update_layout(
        title={'text': 'train loss and valid loss per epoch',
               'y': 0.9,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
    fig.write_image('{}/{}_train_valid_loss_per_epoch.png'.format(args.path, model_type))

