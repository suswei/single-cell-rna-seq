import itertools
import os
import pickle
import pandas as pd

def cross_validation_summary( ):
    dir_path = './result/cross_validation/tabula_muris/batch'
    hyperparameter_config = {
        'MC': list(range(10)),
        'regularize_weight': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 100],
        'n_fold': [1, 2, 3, 4, 5, 6, 7, 8]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(hyperparameter_experiments)):
        MC = hyperparameter_experiments[i]['MC']
        regularize_weight = hyperparameter_experiments[i]['regularize_weight']
        n_fold = hyperparameter_experiments[i]['n_fold']
        config_path = dir_path + '/MC{}/weight{}/nfold{}/config.pkl'.format(MC, regularize_weight, n_fold)
        results_path = dir_path + '/MC{}/weight{}/nfold{}/results.pkl'.format(MC, regularize_weight, n_fold)

        if os.path.isfile(config_path) and os.path.isfile(results_path):
            config = pickle.load(open(config_path, "rb"))
            results = pickle.load(open(results_path, "rb"))

            results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator', 'MC', 'regularize_weight', 'n_fold'])}
            results_config.update(results)

            if i == 0:
                results_config_total = pd.DataFrame.from_dict(results_config)
            else:
                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
        else:
            print('taskid{}, MC{}, weight{}, n_fold{}'.format(i, MC, regularize_weight, n_fold))

        results_config_total['loss_test'] = results_config_total.apply(lambda row: row.obj1_test + row.regularize_weight * row.obj2_test, axis=1)
        results_config_total_notna = results_config_total[results_config_total['loss_test'].notna()]
        grouped_results_config = results_config_total_notna.groupby(['MC', 'regularize_weight']).agg({'loss_test': ['count', 'mean']})
