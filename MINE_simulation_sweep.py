import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    hyperparameter_config = {
        'confounder_type': ['discrete'],
        'category_num': [2],
        'gaussian_dim': [2, 10],
        'mean_diff': [1, 5],
        'mixture_component_num': [10, 128],
        'gaussian_covariance_type': ['all_identity', 'partial_identity'],
        'samplesize': [12800],
        'activation_fun': ['Leaky_ReLU', 'ELU'],
        'unbiased_loss': [True, False],
        'MCs': 20*[1]
        # all_identity means the covariance structure of the gaussian components of both categories are identity
        # partial_identity means the covariance structure of the gaussian components of only one category is identity
        # check MINE_simulation_helper.py for data generation prccess.
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 MINE_simulation_main.py --taskid %s --confounder_type %s --category_num %s --gaussian_dim %s "
              "--mean_diff %s --mixture_component_num %s --gaussian_covariance_type %s --samplesize %s "
              "--activation_fun %s --unbiased_loss %s"
              %(taskid, temp['confounder_type'], temp['category_num'],temp['gaussian_dim'], temp['mean_diff'],
                temp['mixture_component_num'], temp['gaussian_covariance_type'], temp['samplesize'],
                temp['activation_fun'], temp['unbiased_loss'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
