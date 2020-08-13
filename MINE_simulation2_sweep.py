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
        'n_hidden_node': [128],
        'n_hidden_layer': [10],
        'activation_fun': ['ELU'],
        'batch_size': [128],
        'epochs': [400],
        'lr': [5e-4],
        'MC': list(range(20))
        # all_identity means the covariance structure of the gaussian components of both categories are identity
        # partial_identity means the covariance structure of the gaussian components of only one category is identity
        # check MINE_simulation_helper.py for data generation prccess.
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 MINE_simulation2_main.py --taskid %s --confounder_type %s --category_num %s --gaussian_dim %s "
              "--mean_diff %s --mixture_component_num %s --gaussian_covariance_type %s --samplesize %s "
              "--n_hidden_node %s --n_hidden_layer %s --activation_fun %s --batch_size %s --epochs %s --lr %s "
              " --MC %s --unbiased_loss"
              %(taskid, temp['confounder_type'], temp['category_num'],temp['gaussian_dim'], temp['mean_diff'],
                temp['mixture_component_num'], temp['gaussian_covariance_type'], temp['samplesize'],
                temp['n_hidden_node'], temp['n_hidden_layer'], temp['activation_fun'], temp['batch_size'],
                temp['epochs'], temp['lr'], temp['MC'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
