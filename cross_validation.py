import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    # dataset_name could be 'muris_tabula'
    # nuisance_variable is 'batch'
    # conf_estimator means estimator for confounding effect, it is 'MMD' for cross validation
    hyperparameter_config = {
        'dataset_name': ['tabula_muris'],
        'confounder': ['batch'],
        'n_layers_encoder': [2],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'batch_size': [128],
        'adv_estimator': ['stdz_MMD'],
        'MMD_kernel_mul': [2],
        'MMD_kernel_num': [15],
        'epochs': [150],
        'lr': [1e-3],
        'MCs': [20],
        'MC': list(range(20)),
        'n_weights': [10],
        'regularize_weight': [0.001, 5, 10, 50, 100, 400, 800, 1000, 2000, 4000]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 pareto_front_paretoMTL_main.py --regularize "
              "--taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --use_batches --batch_size %s "
              "--adv_estimator %s --MMD_kernel_mul %s --MMD_kernel_num %s "
              "--epochs %s --lr %s --MCs %s --MC %s --n_weights %s --regularize_weight %s "
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'], temp['adv_estimator'], temp['MMD_kernel_mul'],
                 temp['MMD_kernel_num'], temp['epochs'], temp['lr'], temp['MCs'], temp['MC'], temp['n_weights'], temp['regularize_weight'])
              )
if __name__ == "__main__":
    main(sys.argv[1:])
