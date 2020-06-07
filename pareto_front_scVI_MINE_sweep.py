import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    # dataset_name could be 'muris_tabula', 'pbmc'
    # nuisance_variable could be 'batch', 'library_size'
    # conf_estimator means estimator for confounding effect, it could be 'MINE', 'NN' (NN stands for nearest neighbor), 'aggregated_posterior'
    hyperparameter_config = {
        'dataset_name': ['muris_tabula'],
        'confounder': ['batch'],
        'n_layers_encoder': [2],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'batch_size': [128],
        'conf_estimator': ['MINE_MI'],
        'adv_n_hidden': [128],
        'adv_n_layers': [10],
        'activation_fun': ['ELU'],
        'pre_n_epochs': [50],
        'pre_lr': [1e-3],
        'pre_adv_epochs': [100],
        'adv_lr': [5e-5],
        'n_epochs': [600],
        'main_lr': [1e-3],
        'min_obj1': [12000],
        'max_obj1': [20000],
        'min_obj2': [-0.1],
        'max_obj2': [0.9],
        'scale': [ele/10 for ele in range(0, 11)],
        'MCs': 20*[1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 pareto_front_scVI_MINE_main.py --taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --use_batches --batch_size %s "
              "--conf_estimator %s --adv_n_hidden %s --adv_n_layers %s --activation_fun %s "
              "--pre_n_epochs %s --pre_lr %s --pre_adv_epochs %s --adv_lr %s --n_epochs %s --main_lr %s "
              "--std --min_obj1 %s --max_obj1 %s --min_obj2 %s --max_obj2 %s --scale %s --MCs 20"
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'],
                 temp['conf_estimator'], temp['adv_n_hidden'], temp['adv_n_layers'], temp['activation_fun'],
                 temp['pre_n_epochs'], temp['pre_lr'], temp['pre_adv_epochs'], temp['adv_lr'], temp['n_epochs'], temp['main_lr'],
                 temp['min_obj1'], temp['max_obj1'], temp['min_obj2'], temp['max_obj2'], temp['scale'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
