import sys
import os
import itertools

def main(taskid):

    taskid = int(taskid[0])

    # dataset_name could be 'tabula_muris'
    # nuisance_variable is 'batch'
    # adv_estimator means estimator for confounding effect, it could be 'MINE', 'MMD', 'stdz_MMD' (stdz_MMD means standardize the dimension of z, then use MMD)
    hyperparameter_config = {
        'dataset_name': ['tabula_muris'],
        'confounder': ['batch'],
        'n_layers_encoder': [2], #2
        'n_layers_decoder': [2], #2
        'n_hidden': [128], #128
        'n_latent': [10], #
        'batch_size': [128], #4 GPUs
        'adv_estimator': ['MINE'], #
        'adv_n_hidden': [128], #128
        'adv_n_layers': [10],
        'adv_activation_fun': ['ELU'],
        'lr': [1e-3],
        'adv_lr': [5e-5],
        'obj1_nadir': [15531], #
        'obj1_ideal': [14777], #
        'obj2_nadir': [0.428], #
        'obj2_ideal': [0], #
        'epochs': [150], #
        'adv_epochs': [1],
        'n_tasks': [2],
        'MC': list(range(20)),
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))],
        'num_workers': [1]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    temp = hyperparameter_experiments[taskid]

    os.system("python3 paretoMTL_main.py --paretoMTL --use_batches --MCs 20 "
              "--taskid %s --dataset_name %s --confounder %s --n_layers_encoder %s "
              "--n_layers_decoder %s --n_hidden %s --n_latent %s --batch_size %s "
              "--adv_estimator %s --adv_n_hidden %s --adv_n_layers %s --adv_activation_fun %s "
              "--lr %s --adv_lr %s  --obj1_nadir %s --obj1_ideal %s --obj2_nadir %s --obj2_ideal %s --epochs %s --adv_epochs %s "
              "--n_tasks %s --MC %s --npref %s --pref_idx %s --num_workers %s"
              % (taskid, temp['dataset_name'], temp['confounder'], temp['n_layers_encoder'], temp['n_layers_decoder'],
                 temp['n_hidden'], temp['n_latent'], temp['batch_size'], temp['adv_estimator'], temp['adv_n_hidden'],
                 temp['adv_n_layers'], temp['adv_activation_fun'], temp['lr'], temp['adv_lr'], temp['obj1_nadir'],
                 temp['obj1_ideal'], temp['obj2_nadir'], temp['obj2_ideal'], temp['epochs'], temp['adv_epochs'], temp['n_tasks'],
                 temp['MC'], temp['npref_prefidx']['npref'], temp['npref_prefidx']['pref_idx'], temp['num_workers'])
              )

if __name__ == "__main__":
    main(sys.argv[1:])
