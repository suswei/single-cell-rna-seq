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
        'adv_estimator': ['MINE'],
        'adv_n_hidden': [128],
        'adv_n_layers': [10],
        'adv_activation_fun': ['ELU'],
        'epochs': [150],
        'adv_epochs': [1],
        'lr': [1e-3],
        'adv_lr': [5e-5],
        'obj1_nadir': [19664],
        'obj1_ideal': [11513],
        'obj2_nadir': [0.693],
        'obj2_ideal': [-0.1],
        'MC': list(range(20)),
        'weights_total': [12],
        'nweight_weight': [{'nweight': i, 'weight': j} for i,j in zip(list(range(12)), [0, 1/11, 2/11, 3/11, 4/11, 5/11, 6/11, 7/11, 8/11, 9/11, 10/11, 1])],
        'num_workers': [1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 paretoMTL_main.py --regularize --use_batches --MCs 20 "
              "--taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s  --batch_size %s "
              "--adv_estimator %s --adv_n_hidden %s --adv_n_layers %s --adv_activation_fun %s "
              "--epochs %s --adv_epochs %s --lr %s --adv_lr %s --obj1_nadir %s --obj1_ideal %s --obj2_nadir %s --obj2_ideal %s "
              "--MC %s --weights_total %s --nweight %s --weight %s --num_workers %s "
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'], temp['adv_estimator'],
                 temp['adv_n_hidden'], temp['adv_n_layers'], temp['adv_activation_fun'],
                 temp['epochs'], temp['adv_epochs'], temp['lr'], temp['adv_lr'],
                 temp['obj1_nadir'], temp['obj1_ideal'], temp['obj2_nadir'], temp['obj2_ideal'],
                 temp['MC'], temp['weights_total'], temp['nweight_weight']['nweight'],
                 temp['nweight_weight']['weight'], temp['num_workers'])
              )
if __name__ == "__main__":
    main(sys.argv[1:])
