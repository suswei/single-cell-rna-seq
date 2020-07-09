import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    # dataset_name could be 'muris_tabula'
    # nuisance_variable is 'batch'
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
        'pre_epochs': [300, 500, 800],
        'pre_adv_epochs': [100],
        'adv_lr': [5e-4],
        'n_epochs': [50, 100, 200],
        'lr': [1e-2],
        'obj2_scale': [0.3, 0.5],
        'n_tasks': [2],
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 pareto_front_paretoMTL_main.py --taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --use_batches --batch_size %s "
              "--conf_estimator %s --adv_n_hidden %s --adv_n_layers %s --activation_fun %s "
              "--pre_epochs %s --pre_adv_epochs %s --adv_lr %s --n_epochs %s --lr %s --obj2_scale %s "
              "--n_tasks %s --npref %s --pref_idx %s"
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'],
                 temp['conf_estimator'], temp['adv_n_hidden'], temp['adv_n_layers'], temp['activation_fun'],
                 temp['pre_epochs'], temp['pre_adv_epochs'], temp['adv_lr'], temp['n_epochs'], temp['lr'], temp['obj2_scale'],
                 temp['n_tasks'], temp['npref_prefidx']['npref'], temp['npref_prefidx']['pref_idx'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
