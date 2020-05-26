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
        'nuisance_variable': ['batch'],
        'n_layers_encoder': [10],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'dropout_rate': [0.1],
        'reconstruction_loss': ['zinb'],
        'use_batches': [True],
        'train_size': [0.8],
        'std': [True],
        'max_neg_ELBO': [17000],
        'lr': [1e-2],
        'pre_n_epochs': [100],
        'n_epochs': [200],
        'nsamples_z': [200],
        'conf_estimator': ['MINE'],
        'adv_n_latents': [128],
        'adv_n_layers': [10],
        'adv_pre_epochs': [300],
        'adv_epochs': [1],
        'activation_fun': ['ELU'],
        'unbiased_loss': [True],
        'adv_lr': [5e-3],
        'scale': [0, 0.2 , 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        'MCs': 20*[1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 pareto_front_scVI_MINE_main.py --taskid %s --dataset_name %s --nuisance_variable %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --dropout_rate %s --reconstruction_loss %s "
              "--use_batches %s --train_size %s --std %s --max_neg_ELBO %s --lr %s "
              "--pre_n_epochs %s --n_epochs %s --nsamples_z %s --conf_estimator %s --adv_n_latents %s "
              "--adv_n_layers %s --adv_pre_epochs %s --adv_epochs %s --activation_fun %s --unbiased_loss %s "
              "--adv_lr %s --scale %s --MCs %s"
              % (taskid, temp['dataset_name'], temp['nuisance_variable'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['dropout_rate'], temp['reconstruction_loss'],
                 temp['use_batches'], temp['train_size'], temp['std'], temp['max_neg_ELBO'],
                 temp['lr'], temp['pre_n_epochs'], temp['n_epochs'], temp['nsamples_z'], temp['conf_estimator'],
                 temp['adv_n_latents'], temp['adv_n_layers'], temp['adv_pre_epochs'], temp['adv_epochs'], temp['activation_fun'],
                 temp['unbiased_loss'], temp['adv_lr'], temp['scale'], temp['MCs'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])


#the hyperparameter_config already tried when the MINE estimator is changed into classifier or dataset is changed into pbmc
'''
    hyperparameter_config = {
        'dataset_name': ['muris_tabula'],
        'nuisance_variable': ['batch'],
        'adv_model': ['Classifier'],
        'n_layers_encoder': [2,10],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'dropout_rate': [0.1],
        'reconstruction_loss': ['zinb'],
        'use_batches': [True],
        'use_cuda': [False],
        'MIScale': [0.2],
        'train_size': [0.8],
        'lr': [1e-2],  # 1e-3, 5e-3, 1e-4
        'adv_lr': [1, 1e-1, 1e-2, 5e-3, 1e-3, 5e-4],  # 5e-4, 1e-8
        'pre_n_epochs' : [50],
        'n_epochs': [350],  # 350
        'nsamples_z': [200],
        'adv': [True],
        'Adv_Net_architecture': [[256] * 10],
        'pre_adv_epochs': [5],
        'adv_epochs': [1],
        'activation_fun': ['ELU'], 
        'unbiased_loss': [True], 
        'initial': ['xavier_normal'],
        'adv_model': ['Classifier'],
        'adv_drop_out': [0.2],
        'std': [True],
        'taskid': [0]
    }
    hyperparameter_config = {
        'dataset_name': ['pbmc'],
        'nuisance_variable': ['batch'],
        'adv_model': ['MINE'],
        'n_layers_encoder': [1],
        'n_layers_decoder': [1],
        'n_hidden': [256],
        'n_latent': [14],
        'dropout_rate': [0.5],
        'reconstruction_loss': ['zinb'],
        'use_batches': [True],
        'use_cuda': [False],
        'MIScale': [200, 500, 800, 1000, 2000, 5000, 10000, 100000],
        'train_size': [0.8],
        'lr': [0.01],
        'adv_lr': [0.001],
        'n_epochs': [170],
        'nsamples_z': [200],
        'adv': [False]
    }
'''
