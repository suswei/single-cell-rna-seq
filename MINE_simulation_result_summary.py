import os
import pandas as pd
import itertools
import pickle
import plotly.graph_objects as go

def read_file(hyperparameter_config, save_path, type):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(hyperparameter_experiments)):
        config_file_path = save_path + '/taskid{}/config.pkl'.format(i)
        results_file_path = save_path + '/taskid{}/results.pkl'.format(i)
        if os.path.isfile(config_file_path) and os.path.isfile(results_file_path):
            config = pickle.load(open(config_file_path, "rb"))
            results = pickle.load(open(results_file_path, "rb"))

            if type == 'simulation1':
                results_config = {key: [value] for key, value in config.items() if key in tuple(['case_idx','MC'])}
            else:
                results_config = {key: [value] for key, value in config.items() if key in hyperparameter_config.keys()}
            results_config.update(results)

            if i == 0:
                results_config_total = pd.DataFrame.from_dict(results_config)
            else:
                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)

    return results_config_total

def MINE_simulation1_summary(hyperparameter_config, save_path):

    results_config_total = read_file(hyperparameter_config, save_path, 'simulation1')

    results_config_mean_std = results_config_total.groupby('case_idx').agg(
        empirical_MI_mean=('empirical_MI', 'mean'),
        empirical_MI_std=('empirical_MI', 'std'),
        NN_mean=('NN_estimator', 'mean'),
        NN_std=('NN_estimator', 'std'),
        MINE_train_mean=('MI_MINE_train', 'mean'),
        MINE_train_std=('MI_MINE_train', 'std'),
        MINE_test_mean=('MI_MINE_test', 'mean'),
        MINE_test_std=('MI_MINE_test', 'std')
    ).reset_index()

    trueMI = pd.DataFrame.from_dict(pickle.load(open(save_path + '/trueMI.pkl', "rb")))

    results_config_mean_std = pd.concat([results_config_mean_std, trueMI], axis=1)

    case_idx_list = ['case{}'.format(k) for k in range(7)]

    for type in ['train','test']:
        fig = go.Figure(data=[
            go.Bar(name='true_MI', x=case_idx_list, y=results_config_mean_std.loc[:,'trueMI'].values.tolist()),
            go.Bar(name='empirical_MI', x=case_idx_list, y=results_config_mean_std.loc[:, 'empirical_MI_mean'].values.tolist(),
                   error_y=dict(type='data', array=results_config_mean_std.loc[:, 'empirical_MI_std'].values.tolist())),
            go.Bar(name='NN_MI', x=case_idx_list, y=results_config_mean_std.loc[:, 'NN_mean'].values.tolist(),
                   error_y=dict(type='data',array=results_config_mean_std.loc[:, 'NN_std'].values.tolist())),
            go.Bar(name='MINE_MI', x=case_idx_list, y=results_config_mean_std.loc[:,'MINE_{}_mean'.format(type)].values.tolist(),
                   error_y=dict(type='data', array=results_config_mean_std.loc[:,'MINE_{}_std'.format(type)].values.tolist()))
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        fig.update_layout(
            title={'text': type,
                   'y': 0.92,
                   'x': 0.47,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(size=10, color='black', family='Arial, sans-serif')
        )
        fig.write_image(save_path + '/{}.png'.format(type))

def MINE_simulation2_summary(hyperparameter_config, save_path):

    results_config_total = read_file(hyperparameter_config, save_path, 'simulation2')

    del hyperparameter_config['MC']

    results_config_mean_std = results_config_total.groupby(list(hyperparameter_config.keys())).agg(
        NN_estimator_mean=('NN_estimator', 'mean'),
        NN_estimator_std=('NN_estimator', 'std'),
        empirical_MI_mean = ('empirical_mutual_info', 'mean'),
        empirical_MI_std =('empirical_mutual_info', 'std'),
        MI_MINE_train_mean=('MI_MINE_train', 'mean'),
        MI_MINE_train_std=('MI_MINE_train', 'std'),
        MI_MINE_test_mean=('MI_MINE_test', 'mean'),
        MI_MINE_test_std=('MI_MINE_test', 'std'),
        empirical_CD_KL_0_1_mean = ('empirical_CD_KL_0_1','mean'),
        empirical_CD_KL_0_1_std =('empirical_CD_KL_0_1', 'std'),
        CD_KL_0_1_MINE_train_mean=('CD_KL_0_1_MINE_train', 'mean'),
        CD_KL_0_1_MINE_train_std=('CD_KL_0_1_MINE_train', 'std'),
        CD_KL_0_1_MINE_test_mean=('CD_KL_0_1_MINE_test', 'mean'),
        CD_KL_0_1_MINE_test_std=('CD_KL_0_1_MINE_test', 'std'),
        empirical_CD_KL_1_0_mean = ('empirical_CD_KL_1_0', 'mean'),
        empirical_CD_KL_1_0_std=('empirical_CD_KL_1_0', 'std'),
        CD_KL_1_0_MINE_train_mean = ('CD_KL_1_0_MINE_train', 'mean'),
        CD_KL_1_0_MINE_train_std =('CD_KL_1_0_MINE_train', 'std'),
        CD_KL_1_0_MINE_test_mean = ('CD_KL_1_0_MINE_test', 'mean'),
        CD_KL_1_0_MINE_test_std=('CD_KL_1_0_MINE_test', 'std')
    ).reset_index()

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    case_idx_list = ['case{}'.format(k) for k in range(len(hyperparameter_experiments))]

    for type in ['train', 'test']:
        fig = go.Figure(data=[
            go.Bar(name='empirical_MI', x=case_idx_list, y=results_config_mean_std.loc[:, 'empirical_MI_mean'].values.tolist(),
                   error_y=dict(type='data', array=results_config_mean_std.loc[:, 'empirical_MI_std'].values.tolist())),
            go.Bar(name='NN_MI', x=case_idx_list,
                   y=results_config_mean_std.loc[:, 'NN_estimator_mean'].values.tolist(),
                   error_y=dict(type='data', array=results_config_mean_std.loc[:, 'NN_estimator_std'].values.tolist())),
            go.Bar(name='MINE_MI', x=case_idx_list,
                   y=results_config_mean_std.loc[:, 'MI_MINE_{}_mean'.format(type)].values.tolist(),
                   error_y=dict(type='data', array=results_config_mean_std.loc[:, 'MI_MINE_{}_std'.format(type)].values.tolist()))
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        fig.update_layout(
            title={'text': type,
                   'y': 0.92,
                   'x': 0.47,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(size=10, color='black', family='Arial, sans-serif')
        )
        fig.write_image(save_path + '/{}.png'.format(type))


def MINE_simulation_summary( ):

    #for simulation 1
    hyperparameter_config = {
        'case_idx': list(range(7)),
        'MC': list(range(20))
    }
    dir_path = './result/MINE_simulation1'
    MINE_simulation1_summary(hyperparameter_config=hyperparameter_config, save_path=dir_path)

    #for simulation 2
    hyperparameter_config = {
        'category_num': [2],
        'gaussian_dim': [2, 10],
        'mean_diff': [1, 5],
        'mixture_component_num': [10, 128],
        'gaussian_covariance_type': ['all_identity', 'partial_identity'],
        'MC': list(range(20))
    }
    dir_path = './result/MINE_simulation2'

    MINE_simulation2_summary(hyperparameter_config= hyperparameter_config, save_path=dir_path)

