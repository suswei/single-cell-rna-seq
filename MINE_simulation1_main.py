import os
import torch
import pandas as pd
from MINE_simulation_helper import generate_data_MINE_simulation1, train_valid_test_loader, MINE_train, NN_eval
import argparse
from random import randint
import pickle

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

    parser.add_argument('--samplesize', type=int, default=6400,
                        help='training sample size')

    parser.add_argument('--n_hidden_node', type=int, default=128,
                        help='number of hidden nodes per hidden layer in MINE')

    parser.add_argument('--n_hidden_layer', type=int, default=10,
                        help='number of hidden layers in MINE')

    parser.add_argument('--activation_fun', type=str, default='ELU',
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

    if args.activation_fun in ['ELU']:
        args.w_initial = 'normal'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    x_tensor, y_tensor = generate_data_MINE_simulation1(args)

    train_loader, valid_loader, test_loader = train_valid_test_loader(x_tensor, y_tensor, args, kwargs)
    MI_MINE_train, MI_MINE_test = MINE_train(train_loader, valid_loader, test_loader, 'MI', args)
    NN_train, NN_test = NN_eval(train_loader, test_loader)

    estimated_results = {'MI_MINE_train': [MI_MINE_train],
                         'MI_MINE_test': [MI_MINE_test],
                         'NN_train': {NN_train},
                         'NN_test': [NN_test]}

    if not os.path.isdir('result/MINE_simulation1/taskid{}'.format(args.taskid)):
        os.makedirs('result/MINE_simulation1/taskid{}'.format(args.taskid))
    args.save_path = 'result/MINE_simulation1/taskid{}'.format(args.taskid)

    args_dict = vars(args)
    with open('{}/config.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(args_dict, f)

    with open('{}/results.pkl'.format(args.save_path), 'wb') as f:
        pickle.dump(estimated_results, f)

if __name__ == "__main__":
    main()
