import itertools
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def simple_cull(inputPoints, dominates, return_index: bool=False, min_max: str='min'):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row, return_index, min_max):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, return_index, min_max):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow, return_index, min_max):
    #check if row dominates candidateRow
    if return_index == True:
        if min_max == 'min':
            return sum([row[x + 1] <= candidateRow[x + 1] for x in range(len(row)-1)]) == len(row) - 1
        elif min_max == 'max':
            return sum([row[x + 1] >= candidateRow[x + 1] for x in range(len(row)-1)]) == len(row) - 1
    else:
        if min_max == 'min':
            return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)
        elif min_max == 'max':
            return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

def draw_plot(save_path, dataframe, MC_idx):

    dataframe = dataframe.sort_values('regularize_weight')
    weight_list = [k+1 for k in list(range(10))]
    fig = go.Figure()
    obj1_all, obj2_all = [], []
    for (i, type) in enumerate(['train', 'test']):
        obj1 = dataframe.loc[:, 'obj1_{}'.format(type)].values.tolist()
        obj2 = dataframe.loc[:, 'obj2_{}'.format(type)].values.tolist()

        obj1_all += obj1
        obj2_all += obj2

        inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
        paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')
        pp = np.array(list(paretoPoints1))
        dp = np.array(list(dominatedPoints1))

        if i == 0:
            marker_symbol = 'cross'
            opacity = 1
        else:
            marker_symbol = 'circle'
            opacity = 0.5

        fig.add_trace(go.Scatter(x=obj1, y=obj2, opacity=opacity, text=['{}'.format(i) for i in weight_list],
                                 mode='markers+text', textfont_size=12, textposition='top center', marker_size=10,
                                 marker_symbol=marker_symbol,
                                 marker_color='rgba(0, 0, 0, 0)', marker_opacity=opacity, showlegend=False))
        if dp.shape[0] > 0:
            fig.add_trace(go.Scatter(x=dp[:, 0].tolist(), y=dp[:, 1].tolist(),
                                     mode='markers', marker_size=10, marker_symbol=marker_symbol,
                                     name='{}, yes'.format(type),
                                     marker_color='rgba(0, 0, 255, .9)', marker_opacity=opacity, showlegend=True))
        if pp.shape[0] > 0:
            fig.add_trace(go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(),
                                     mode='markers', marker_size=10, marker_symbol=marker_symbol,
                                     name='{}, no'.format(type),
                                     marker_color='rgba(255, 25, 52, .9)', marker_opacity=opacity, showlegend=True))

    fig_title = r'$\Large V_n(\phi)=MMD_{n}(\phi)$'
    fig.update_layout(
        width=600,
        height=400,
        margin=dict(
            l=80,
            r=90,
            b=70,
            t=40,
            pad=1
        ),
        font=dict(color='black', family='Times New Roman'),
        title={
            'text': fig_title,
            'font': {
                'size': 25
            },
            'y': 0.94,
            'x': 0.42,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend_title=dict(font=dict(family='Times New Roman', size=20, color="black"), text='train/test, dominated'),
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=20,
                color='black'
            )
        ),
        xaxis_tickformat='s'
    )
    fig.update_xaxes(tickfont=dict(size=20), title_text=r'$\large \text{Loss }U_n(\phi, \theta)$',
                     title_font=dict(size=25, family='Times New Roman', color='black'),
                     range=[min(obj1_all) - 50, max(obj1_all) + 50], autorange=False)  # tick0=15200,dtick=100
    fig.update_yaxes(tickfont=dict(size=20), title_text=r'$\large \text{Batch effect }V_n(\phi)$',
                     title_font=dict(size=25, family='Times New Roman', color='black'),
                     range=[min(obj2_all) - 0.01, max(obj2_all) + 0.01], autorange=False)

    fig.write_image(save_path + '/MC{}.png'.format(MC_idx))

def regularize_summary(MCs):
    dir_path = './result/regularize/tabula_muris/batch'
    hyperparameter_config = {
        'MC': list(range(MCs)),
        'regularize_weight': [10, 40, 80, 100, 400, 800, 1000, 2000, 4000, 6000],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(hyperparameter_experiments)):
        MC = hyperparameter_experiments[i]['MC']
        regularize_weight = hyperparameter_experiments[i]['regularize_weight']
        config_path = dir_path + '/MC{}/weight{}/config.pkl'.format(MC, regularize_weight)
        results_path = dir_path + '/MC{}/weight{}/results.pkl'.format(MC, regularize_weight)

        if os.path.isfile(config_path) and os.path.isfile(results_path):
            config = pickle.load(open(config_path, "rb"))
            results = pickle.load(open(results_path, "rb"))

            results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator', 'MC', 'regularize_weight'])}
            results_config.update(results)

            if i == 0:
                results_config_total = pd.DataFrame.from_dict(results_config)
            else:
                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
        else:
            print('taskid{}, MC{}, weight{}'.format(i, MC, regularize_weight))

    for MC in range(MCs):
        dataframe = results_config_total[results_config_total.MC.eq(MC)]
        draw_plot(dir_path, dataframe, MC)
