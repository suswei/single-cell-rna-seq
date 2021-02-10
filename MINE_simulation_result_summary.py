import os
import pandas as pd
import itertools
import pickle
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

            results['NN_estimator'] = [results['NN_estimator'][0].item()]
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
        NN_MI_mean=('NN_estimator', 'mean'),
        NN_MI_std=('NN_estimator', 'std'),
        MINE_MI_train_mean=('MI_MINE_train', 'mean'),
        MINE_MI_train_std=('MI_MINE_train', 'std'),
        MINE_MI_test_mean=('MI_MINE_test', 'mean'),
        MINE_MI_test_std=('MI_MINE_test', 'std')
    ).reset_index()

    trueMI = pd.DataFrame.from_dict(pickle.load(open(save_path + '/trueMI.pkl', "rb")))

    results_config_mean_std = pd.concat([results_config_mean_std, trueMI], axis=1)

    case_idx_list = ['case{}'.format(k+1) for k in range(7)]

    methods_list = ['true','empirical', 'NN','MINE']
    if len(methods_list)<6:
        colors_list = ['rgba(0, 0, 255, .9)','rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)', 'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of color')

    fig = make_subplots(rows=1, cols=2, subplot_titles=('train', 'test'))
    for (i, type) in enumerate(['train', 'test']):
        if i==0:
            showlegend=False
        else:
            showlegend=True
        for (j, method) in enumerate(methods_list):
            if method == 'true':
                fig.add_trace(
                    go.Bar(x=case_idx_list, y=results_config_mean_std.loc[:, 'trueMI'].values.tolist(), name='true_MI',
                           marker_color=colors_list[j], showlegend=showlegend),1,i+1)
            elif method in ['empirical','NN']:
                fig.add_trace(
                    go.Bar(x=case_idx_list, y=results_config_mean_std.loc[:, '{}_MI_mean'.format(method)].values.tolist(),
                       error_y=dict(type='data',array=results_config_mean_std.loc[:, '{}_MI_std'.format(method)].values.tolist()),
                       name='{}_MI'.format(method), marker_color=colors_list[j],showlegend=showlegend),1,i+1)
            else:
                fig.add_trace(
                    go.Bar(x=case_idx_list, y=results_config_mean_std.loc[:, '{}_MI_{}_mean'.format(method,type)].values.tolist(),
                       error_y=dict(type='data',array=results_config_mean_std.loc[:, '{}_MI_{}_std'.format(method,type)].values.tolist()),
                       name='{}_MI'.format(method), marker_color=colors_list[j],showlegend=showlegend),1,i+1)


    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(
        width=1400,
        height=700,
        margin=dict(
            l=1,
            r=1,
            b=1,
            t=25,
            pad=1
        ),
        font=dict(size=20, color='black', family='Times New Roman'),
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=25,
                color="black"
            )
        ),
        plot_bgcolor='rgb(255,255,255)'
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    fig.update_xaxes(title_font=dict(size=20, family='Times New Roman', color='black'), ticks='outside')
    fig.update_yaxes(title_font=dict(size=20, family='Times New Roman', color='black'), ticks='outside')
    fig.write_image(save_path + '/MINE_simulation1.png')

def MINE_simulation2_summary(hyperparameter_config, save_path):

    results_config_total = read_file(hyperparameter_config, save_path, 'simulation2')

    del hyperparameter_config['MC']

    results_config_mean_std = results_config_total.groupby(list(hyperparameter_config.keys())).agg(
        NN_MI_mean=('NN_estimator', 'mean'),
        NN_MI_std=('NN_estimator', 'std'),
        empirical_MI_mean = ('empirical_mutual_info', 'mean'),
        empirical_MI_std =('empirical_mutual_info', 'std'),
        MINE_MI_train_mean=('MI_MINE_train', 'mean'),
        MINE_MI_train_std=('MI_MINE_train', 'std'),
        MINE_MI_test_mean=('MI_MINE_test', 'mean'),
        MINE_MI_test_std=('MI_MINE_test', 'std'),
        empirical_CD_KL_0_1_mean = ('empirical_CD_KL_0_1','mean'),
        empirical_CD_KL_0_1_std =('empirical_CD_KL_0_1', 'std'),
        MINE_CD_KL_0_1_train_mean=('CD_KL_0_1_MINE_train', 'mean'),
        MINE_CD_KL_0_1_train_std=('CD_KL_0_1_MINE_train', 'std'),
        MINE_CD_KL_0_1_test_mean=('CD_KL_0_1_MINE_test', 'mean'),
        MINE_CD_KL_0_1_test_std=('CD_KL_0_1_MINE_test', 'std'),
        empirical_CD_KL_1_0_mean = ('empirical_CD_KL_1_0', 'mean'),
        empirical_CD_KL_1_0_std=('empirical_CD_KL_1_0', 'std'),
        MINE_CD_KL_1_0_train_mean = ('CD_KL_1_0_MINE_train', 'mean'),
        MINE_CD_KL_1_0_train_std =('CD_KL_1_0_MINE_train', 'std'),
        MINE_CD_KL_1_0_test_mean = ('CD_KL_1_0_MINE_test', 'mean'),
        MINE_CD_KL_1_0_test_std=('CD_KL_1_0_MINE_test', 'std')
    ).reset_index()

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    case_idx_list = ['case{}'.format(k+1) for k in range(len(hyperparameter_experiments))]

    methods_list = ['empirical', 'NN', 'MINE']
    if len(methods_list) < 6:
        colors_list = ['rgba(0, 0, 255, .9)', 'rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)',
                   'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of color')

    fig = make_subplots(rows=1, cols=2, subplot_titles=('train', 'test'))
    for (i, type) in enumerate(['train', 'test']):
        if i == 0:
            showlegend = False
        else:
            showlegend = True
        for (j, method) in enumerate(methods_list):
            if method in ['empirical','NN']:
                fig.add_trace(
                    go.Bar(x=case_idx_list, y=results_config_mean_std.loc[:, '{}_MI_mean'.format(method)].values.tolist(),
                           error_y=dict(type='data', array=results_config_mean_std.loc[:, '{}_MI_std'.format(method)].values.tolist()),
                           name='{}_MI'.format(method), marker_color=colors_list[j], showlegend=showlegend), 1, i+1)
            else:
                fig.add_trace(
                    go.Bar(x=case_idx_list, y=results_config_mean_std.loc[:, '{}_MI_{}_mean'.format(method,type)].values.tolist(),
                           error_y=dict(type='data',array=results_config_mean_std.loc[:, '{}_MI_{}_std'.format(method,type)].values.tolist()),
                           name='{}_MI'.format(method), marker_color=colors_list[j], showlegend=showlegend), 1, i+1)

    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(
        width=1400,
        height=700,
        margin=dict(
            l=1,
            r=1,
            b=1,
            t=25,
            pad=1
        ),
        font=dict(size=20, color='black', family='Times New Roman'),
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=25,
                color="black"
            )
        ),
        plot_bgcolor='rgb(255,255,255)'
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    fig.update_xaxes(title_font=dict(size=20, family='Times New Roman', color='black'), ticks='outside')
    fig.update_yaxes(title_font=dict(size=20, family='Times New Roman', color='black'), ticks='outside')
    fig.write_image(save_path + '/MINE_simulation2.png')

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

