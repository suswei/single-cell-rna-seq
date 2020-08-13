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
        'adv_estimator': ['MINE'],
        'adv_n_hidden': [128],
        'adv_n_layers': [10],
        'adv_activation_fun': ['ELU'],
        'lr': [1e-3],
        'adv_lr': [5e-5],
        'alpha': [2],
        'gradnorm_lr': [1e-8],
        'gradnorm_weight_lowlimit': [1e-6],
        'n_epochs': [50],
        'n_tasks': [2],
        'MC': list(range(10)),
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 pareto_front_paretoMTL_main.py --gradnorm_paretoMTL --MCs 10 "
              "--taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --use_batches --batch_size %s "
              "--adv_estimator %s --adv_n_hidden %s --adv_n_layers %s --adv_activation_fun %s "
              "--lr %s --adv_lr %s  --alpha %s --gradnorm_lr %s --gradnorm_weight_lowlimit %s --n_epochs %s "
              "--n_tasks %s --MC %s --npref %s --pref_idx %s"
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'],
                 temp['adv_estimator'], temp['adv_n_hidden'], temp['adv_n_layers'], temp['adv_activation_fun'],
                 temp['lr'], temp['adv_lr'], temp['alpha'], temp['gradnorm_lr'], temp['gradnorm_weight_lowlimit'], temp['n_epochs'],
                 temp['n_tasks'], temp['MC'], temp['npref_prefidx']['npref'], temp['npref_prefidx']['pref_idx'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
