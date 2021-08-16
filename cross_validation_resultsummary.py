import itertools
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def draw_plot(save_path, dataframe):

    regularize_weight_indicator = [k+1 for k in list(range(11))]
    loss_test_mean = list(dataframe.loc[:,'loss_test_mean'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=regularize_weight_indicator, y=loss_test_mean, mode='markers + text',  marker_size=15, marker_symbol='circle',
                             marker_color='rgba(255, 25, 52, .9)', text=list(dataframe.loc[:,'regularize_weight']),
                             textfont_size=10, textposition='top center', showlegend=False))

    MC = np.unique(list(dataframe.loc[:,'MC']))[0]
    fig_title = 'MC: {}, best_weight: {}'.format(MC, list(dataframe.sort_values('loss_test_mean').loc[:,'regularize_weight'])[0])

    fig.update_layout(
        font=dict(color='black', family='Times New Roman'),
        title={
            'text': fig_title,
            'font': {
                'size': 25
            },
            'y': 0.94,
            'x': 0.42,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    fig.write_image(save_path + '/MC{}.png'.format(MC))

def cross_validation_summary(MCs):
    dir_path = './result/cross_validation/tabula_muris/batch'
    hyperparameter_config = {
        'MC': list(range(MCs)),
        'regularize_weight': [5000, 4000, 3000, 2000, 1000, 100, 80, 60, 40, 20, 10],
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
        grouped_results_config = results_config_total_notna.groupby(['MC', 'regularize_weight']).agg({'loss_test': ['count', 'mean']}).reset_index()
        grouped_results_config.columns = ['MC','regularize_weight','loss_test_count','loss_test_mean']

    for MC in range(MCs):
        dataframe = grouped_results_config[grouped_results_config.MC.eq(MC)]
        draw_plot(dir_path, dataframe)
