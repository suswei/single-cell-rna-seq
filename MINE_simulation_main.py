import os
from MINE_simulation_helper import generate_data_MINE_simulation, train_valid_test_loader, sample1_sample2_from_minibatch, MINE_unbiased_loss, diagnosis_loss_plot
import argparse
from random import randint
import torch
from scvi.models.modules import MINE_Net3
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle

def MINE_train(train_loader, valid_loader, test_loader, model_type, args):

    #MINE_MI uses MINE to estimate mutual information between z and s, where s changes into dummy variable
    #MINE_CD_KL uses MINE to estimate D(p(z|s=0)||p(z|s=1)), both networks use unbiased_loss=True

    if model_type == 'MI':
        input_dim = args.gaussian_dim + 2
    elif model_type != 'MI':
        input_dim = args.gaussian_dim
    MINE = MINE_Net3(input_dim=input_dim, n_latents=[args.n_hidden_node]*args.n_hidden_layer,
                         activation_fun=args.activation_fun, unbiased_loss=args.unbiased_loss, initial=args.w_initial)

    opt_MINE = optim.Adam(MINE.parameters(), lr=args.lr)
    scheduler_MINE_MI = ReduceLROnPlateau(opt_MINE, mode='min', factor=0.1, patience=10, verbose=True)

    MINE_estimator_minibatch_list, negative_loss_minibatch_list, valid_loss_epoch, train_loss_epoch = [], [], [], []
    for epoch in range(args.epochs):

        MINE.train()

        for batch_idx, (z, batch) in enumerate(train_loader):

            sample1, sample2 = sample1_sample2_from_minibatch(args, model_type, z, batch)
            t = MINE(sample1)
            et = torch.exp(MINE(sample2))

            if args.unbiased_loss == True:
                if MINE.ma_et is None:
                    MINE.ma_et = torch.mean(et).detach().item() #detach means will not calculate gradient for ma_et, ma_et is just a number
                MINE.ma_et = (1 - MINE.ma_rate) * MINE.ma_et + MINE.ma_rate * torch.mean(et).detach().item()

                #Pay attention, even if use unbiased_loss, this unbiased loss is not our MINE estimator,
                #The MINE estimator is still torch.mean(t) - torch.log(torch.mean(et)) after training
                #the unbiased_loss is only for getting unbiased gradient.
                loss = MINE_unbiased_loss(args, MINE.ma_et, t, et)
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

        if epoch % args.log_interval == 0:
            print('Train Epoch: {} \tMINE_estimator_minibatch: {:.6f}\tnegative_loss_minibatch: {:.6f}'.format(epoch, MINE_estimator_minibatch.data.item(), -loss.data.item()))

        # diagnoisis of overfitting
        with torch.no_grad():  # to save memory, no intermediate activations used for gradient calculation is stored.

            MINE.eval()

            valid_loss_minibatch_list = []
            for valid_batch_idx, (valid_z, valid_batch) in enumerate(valid_loader):

                valid_sample1, valid_sample2 = sample1_sample2_from_minibatch(args, model_type, valid_z, valid_batch)
                valid_t = MINE(valid_sample1)
                valid_et = torch.exp(MINE(valid_sample2))

                if args.unbiased_loss == True:
                    valid_loss_minibatch = MINE_unbiased_loss(args, MINE.ma_et, valid_t, valid_et)
                else:
                    valid_loss_minibatch = -(torch.mean(valid_t) - torch.log(torch.mean(valid_et)))

                valid_loss_minibatch_list.append(valid_loss_minibatch.item())

            valid_loss_one = np.average(valid_loss_minibatch_list)
            scheduler_MINE_MI.step(valid_loss_one)
            valid_loss_epoch.append(valid_loss_one)

            train_loss_minibatch_list = []
            for train_batch_idx, (train_z, train_batch) in enumerate(train_loader):

                train_sample1, train_sample2 = sample1_sample2_from_minibatch(args, model_type, train_z, train_batch)
                train_t = MINE(train_sample1)
                train_et = torch.exp(MINE(train_sample2))

                if args.unbiased_loss == True:
                    train_loss_minibatch = MINE_unbiased_loss(args, MINE.ma_et, train_t, train_et)
                else:
                    train_loss_minibatch = -(torch.mean(train_t) - torch.log(torch.mean(train_et)))

                train_loss_minibatch_list.append(train_loss_minibatch.item())

            train_loss_one = np.average(train_loss_minibatch_list)
            train_loss_epoch.append(train_loss_one)

    diagnosis_loss_plot(args, model_type, MINE_estimator_minibatch_list, negative_loss_minibatch_list, valid_loss_epoch, train_loss_epoch)

    with torch.no_grad():
        MINE.eval()
        train_z_all = torch.empty(0, args.gaussian_dim)
        if model_type == 'MI':
            train_batch_all = torch.empty(0, 2)
        else:
            train_batch_all = torch.empty(0, 1)
        for train_batch_idx, (train_z, train_batch) in enumerate(train_loader):
            train_z_all = torch.cat((train_z_all, train_z), 0)
            train_batch_all = torch.cat((train_batch_all, train_batch), 0)

        train_sample1_all, train_sample2_all = sample1_sample2_from_minibatch(args, model_type, train_z_all, train_batch_all)
        train_t_all = MINE(train_sample1_all)
        train_et_all = torch.exp(MINE(train_sample2_all))
        train_MINE_estimator = torch.mean(train_t_all) - torch.log(torch.mean(train_et_all))

        test_z_all = torch.empty(0, args.gaussian_dim)
        if model_type == 'MI':
            test_batch_all = torch.empty(0, 2)
        else:
            test_batch_all = torch.empty(0, 1)
        for test_batch_idx, (test_z, test_batch) in enumerate(test_loader):
            test_z_all = torch.cat((test_z_all, test_z), 0)
            test_batch_all = torch.cat((test_batch_all, test_batch), 0)

        test_sample1_all, test_sample2_all = sample1_sample2_from_minibatch(args, model_type, test_z_all, test_batch_all)
        test_t_all = MINE(test_sample1_all)
        test_et_all = torch.exp(MINE(test_sample2_all))
        test_MINE_estimator = torch.mean(test_t_all) - torch.log(torch.mean(test_et_all))

    return train_MINE_estimator.detach().item(), test_MINE_estimator.detach().item()

def main():
    parser = argparse.ArgumentParser(description='MINE Simulation Study')

    parser.add_argument('--taskid', type=int, default=1000 + randint(0, 1000),
                        help='taskid from sbatch')

    parser.add_argument('--confounder_type', type=str, default='discrete',
                        help='the type of the confounder variable')

    parser.add_argument('--category_num', type=int, default=2,
                        help='the number of batches')

    parser.add_argument('--gaussian_dim', type=int, default=2,
                        help='the dimension of gaussian component for each batch')

    parser.add_argument('--mean_diff', type=int, default=1,
                        help='the mean difference for the two gaussians which generate the mean for the gaussian components')

    parser.add_argument('--mixture_component_num', type=int, default=10,
                        help='the number of gaussian components for the gaussian mixture for each batch')

    parser.add_argument('--gaussian_covariance_types', type=str, default='all_identity',
                        help='the covariance type of each gaussian component in the gaussian mixture for each batch')

    parser.add_argument('--samplesize', type=int, default=6400,
                        help='training sample size')

    parser.add_argument('--n_hidden_node', type=int, default=128,
                        help='number of hidden nodes per hidden layer in MINE')

    parser.add_argument('--n_hidden_layer', type=int, default=10,
                        help='number of hidden layers in MINE')

    parser.add_argument('--activation_fun', type=str, default='Leaky_ReLU',
                        help='activation function in MINE')

    parser.add_argument('--unbiased_loss', type=bool, default=True,
                        help='whether to use unbiased loss or not in MINE')

    parser.add_argument('--batchsize', type=int, default=128,
                        help='batch size for MINE training')

    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs in MINE training')

    parser.add_argument('--lr', type=int, default=5e-4,
                        help='learning rate in MINE training')

    parser.add_argument('--unbiased_loss_type', type=str, default='type2',
                        help='which unbiased loss to use')

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
    elif args.activation_fun in ['ReLU', 'ELU']:
        args.w_initial = 'normal'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    path = './MINE_simulation/{}/taskid{}'.format(args.confounder_type, args.taskid)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.confounder_type == 'discrete':

        empirical_mutual_info, empirical_CD_KL_0_1,empirical_CD_KL_1_0, nearest_neighbor_estimate, z_tensor, batch_tensor = generate_data_MINE_simulation(args)
        print('empirical_mutual_info: {}, empirical_CD_KL_0_1: {}, empirical_CD_KL_1_0: {}, nearest_neighbor_estimate: {}'.format(
            empirical_mutual_info, empirical_CD_KL_0_1, empirical_CD_KL_1_0, nearest_neighbor_estimate))

        train_loader, valid_loader, test_loader = train_valid_test_loader(z_tensor, batch_tensor, 'MI', args, kwargs)
        MI_MINE_train, MI_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'MI', args)

        train_loader, valid_loader, test_loader = train_valid_test_loader(z_tensor, batch_tensor, 'CD_KL_0_1', args, kwargs)
        CD_KL_0_1_MINE_train, CD_KL_0_1_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'CD_KL_0_1', args)

        train_loader, valid_loader, test_loader = train_valid_test_loader(z_tensor, batch_tensor, 'CD_KL_1_0', args, kwargs)
        CD_KL_1_0_MINE_train, CD_KL_1_0_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'CD_KL_1_0', args)

    args_dict = vars(args)
    with open('{}/config.pkl'.format(path), 'wb') as f:
        pickle.dump(args_dict, f)

    results = {'empirical_mutual_info': [empirical_mutual_info],
               'empirical_CD_KL_0_1': [empirical_CD_KL_0_1],
               'empirical_CD_KL_1_0': [empirical_CD_KL_1_0],
               'nearest_neighbor_estimate': [nearest_neighbor_estimate],
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
