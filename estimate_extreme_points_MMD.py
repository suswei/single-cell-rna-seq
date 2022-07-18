import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    # dataset_name could be 'tabula_muris', 'TM_MCA_Lung', 'macaque_retina'
    # nuisance_variable is 'batch'
    # adv_estimator means estimator for confounding effect, it could be 'MINE', 'MMD'.
    hyperparameter_config = {
        'dataset_name': ['TM_MCA_Lung'], #tabula_muris
        'confounder': ['batch'],
        'n_layers_encoder': [2],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'batch_size': [128], #2 GPUs
        'adv_estimator': ['MMD'],
        'MMD_bandwidths': ['1,2,5,8,10'],
        'lr': [1e-3],
        'adv_lr': [5e-5],
        'epochs': [250], #250
        'adv_epochs': [1],
        'MC': list(range(10)),
        'weight': [0,1],
        'num_workers': [1] #4
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 main.py --extreme_points --use_batches "
              "--taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --batch_size %s "
              "--adv_estimator %s --MMD_bandwidths %s "
              "--lr %s --adv_lr %s --epochs %s --adv_epochs %s --MC %s --weight %s --num_workers %s"
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'], temp['adv_estimator'], temp['MMD_bandwidths'],
                 temp['lr'], temp['adv_lr'], temp['epochs'], temp['adv_epochs'], temp['MC'], temp['weight'], temp['num_workers'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
