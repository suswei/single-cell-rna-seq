import os
from MINE_simulation_helper import generate_data_MINE_simulation2, train_valid_test_loader, MINE_train
import argparse
from random import randint
import torch
import pickle


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

    parser.add_argument('--batchsize', type=int, default=128,
                        help='batch size for MINE training')

    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs in MINE training')

    parser.add_argument('--lr', type=float, default=5e-4,
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

    empirical_mutual_info, empirical_CD_KL_0_1,empirical_CD_KL_1_0, nearest_neighbor_estimate, x_tensor, y_tensor = generate_data_MINE_simulation2(args)
    print('empirical_mutual_info: {}, empirical_CD_KL_0_1: {}, empirical_CD_KL_1_0: {}, nearest_neighbor_estimate: {}'.format(
        empirical_mutual_info, empirical_CD_KL_0_1, empirical_CD_KL_1_0, nearest_neighbor_estimate))

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, kwargs)
    MI_MINE_train, MI_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'MI', args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, kwargs)
    CD_KL_0_1_MINE_train, CD_KL_0_1_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'CD_KL_0_1', args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, kwargs)
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
