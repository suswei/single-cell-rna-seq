import os
import itertools
import numpy as np
import pandas as pd
from scvi.dataset.tabula_muris import TabulaMuris
from scipy import sparse
import pickle
from pygmo import hypervolume
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import statistics

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

def draw_pareto_front(dataframe, methods_list, pareto_front_x, pareto_front_y, cal_metric, save_path):

    if cal_metric:
        hypervolume_dict = {}
        percentage_dict = {}
        if pareto_front_x =='obj1':
            obj1_max = dataframe.loc[:, ['obj1_train', 'obj1_test']].max(axis=0).max()
            if pareto_front_y in ['obj2','NN']:
                obj2_max = dataframe.loc[:,['{}_train'.format(pareto_front_y), '{}_test'.format(pareto_front_y)]].max(axis=0).max()
            elif pareto_front_y == 'be':
                obj2_max = dataframe.loc[:, ['be_train', 'be_test']].min(axis=0).min() * (-1)
            ref_point = [obj1_max + 5, obj2_max + 0.01]

        else :
            obj1_max = dataframe.loc[:,['{}_train'.format(pareto_front_x), '{}_test'.format(pareto_front_x)]].min(axis=0).min() * (-1)
            if pareto_front_y in ['obj2','NN']:
                obj2_max = dataframe.loc[:, ['{}_train'.format(pareto_front_y), '{}_test'.format(pareto_front_y)]].max(axis=0).max()
            elif pareto_front_y == 'be':
                obj2_max = dataframe.loc[:, ['be_train', 'be_test']].min(axis=0).min() * (-1)
            ref_point = [obj1_max + 0.01, obj2_max + 0.01]

    for MC in range(dataframe.MC.max()+1):
        dataframe_oneMC = dataframe[dataframe.MC.eq(MC)]

        fig = go.Figure()
        obj1_all, obj2_all = [], []
        image_save_path = save_path

        for (i, method) in enumerate(methods_list):
            image_save_path = image_save_path + '{}_'.format(method)
            dataframe_oneMC_oneMethod = dataframe_oneMC[dataframe_oneMC.method.eq(method)]

            for (j,type) in enumerate(['train', 'test']):

                obj1 = dataframe_oneMC_oneMethod.loc[:, '{}_{}'.format(pareto_front_x, type)].values.tolist()
                obj2 = dataframe_oneMC_oneMethod.loc[:, '{}_{}'.format(pareto_front_y, type)].values.tolist()

                if pareto_front_x == 'obj1' and pareto_front_y == 'be':
                    obj2 = [(-1) * k for k in obj2]
                elif pareto_front_x != 'obj1' and pareto_front_y != 'be':
                    obj1 = [(-1) * k for k in obj1]
                elif pareto_front_x != 'obj1' and pareto_front_y == 'be':
                    obj1 = [(-1) * k for k in obj1]
                    obj2 = [(-1) * k for k in obj2]

                obj1_all += obj1
                obj2_all += obj2

                inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
                paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')
                pp = np.array(list(paretoPoints1))
                #dp = np.array(list(dominatedPoints1))

                inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
                index_list = []
                for paretoPoint in paretoPoints1:
                    for index, inputPoint in enumerate(inputPoints1):
                        if paretoPoint[0] == inputPoint[0] and paretoPoint[1] == inputPoint[1]:
                            index_list += [index+1]

                if cal_metric:
                    # when hypervolume is calculated, the rectangles of dominated points will be covered by those of non-dominated points
                    hv = hypervolume(inputPoints1)
                    hypervolume_dict.update({'{}_{}_MC{}'.format(method, type, MC): [hv.compute(ref_point)]})

                    percentage_dict.update({'{}_{}_MC{}'.format(method, type, MC): [len(list(paretoPoints1))/10]})

                if i == 0:
                    marker_symbol = 'circle'
                elif i == 1:
                    marker_symbol = 'cross'
                elif i == 2:
                    marker_symbol = 'triangle-up'
                elif i == 3:
                    marker_symbol = 'diamond'

                if j==0 and pp.shape[0] > 0:
                    fig.add_trace(go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(), text=['{}'.format(i) for i in index_list],
                                             mode='markers+text', textfont_size=12, textposition='top center',
                                             marker_size=10, marker_symbol=marker_symbol,
                                             marker_color='rgba(0, 0, 0, 0)', showlegend=False))
                elif j==1 and pp.shape[0] > 0:
                    fig.add_trace(go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(), text=['{}'.format(i) for i in index_list],
                                             mode='markers+text', textfont_size=12, textposition='top center',
                                             marker_size=10, marker_symbol=marker_symbol, name='{},{}'.format(method, type),
                                             marker_color='rgba(0, 0,0, 0)', showlegend=False))

                if j == 0 and pp.shape[0] > 0:
                    fig.add_trace(
                        go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(),
                                   mode='markers', textfont_size=12, textposition='top center',
                                   marker_size=10, marker_symbol=marker_symbol, name='{},{}'.format(method, type),
                                   marker_color='rgba(255, 25, 52, .9)', showlegend=True))
                elif j == 1 and pp.shape[0] > 0:
                    fig.add_trace(
                        go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(),
                                   mode='markers', textfont_size=12, textposition='top center',
                                   marker_size=10, marker_symbol=marker_symbol, name='{},{}'.format(method, type),
                                   marker_color='rgba(0, 0, 255, .9)', showlegend=True))

        fig.update_layout(
            width=600,
            height=400,
            margin=dict(
                l=80,
                r=90,
                b=70,
                t=50,
                pad=1
            ),
            font=dict(color='black', family='Times New Roman'),
            title={
                'text': 'non-dominated points',
                'font': {
                    'size': 25
                },
                'y': 0.94,
                'x': 0.42,
                'xanchor': 'center',
                'yanchor': 'top'},
            legend=dict(
                font=dict(
                    family='Times New Roman',
                    size=20,
                    color='black'
                )
            )
        )

        if pareto_front_type == 'obj2':
            fig.update_xaxes(tickfont=dict(size=20),title_text=r'$\large \text{Loss }U_n(\phi, \theta)$' ,
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj1_all) - 50, max(obj1_all) + 50], autorange=False) #tick0=15200,dtick=100
            fig.update_yaxes(tickfont=dict(size=20),title_text=r'$\large \text{Batch effect }V_n(\phi)$' ,
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj2_all) - 0.1, max(obj2_all) + 0.1], autorange=False)
        elif pareto_front_type == 'NN':
            fig.update_xaxes(tickfont=dict(size=20), title_text=r'$\large \text{Loss }U_n(\phi, \theta)$',
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj1_all) - 50, max(obj1_all) + 50], autorange=False)  # tick0=15200,dtick=100
            fig.update_yaxes(tickfont=dict(size=20), title_text=r'$\large \text{Batch effect }NN_n(\phi)$',
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj2_all) - 0.08, max(obj2_all) + 0.08], autorange=False)
        else:
            fig.update_xaxes(tickfont=dict(size=20), title_text='negative {}'.format(pareto_front_type),
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj1_all) - 0.08, max(obj1_all) + 0.08], autorange=False)  # tick0=15200,dtick=100
            fig.update_yaxes(tickfont=dict(size=20), title_text='negative BE',
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj2_all) - 0.08, max(obj2_all) + 0.08], autorange=False)

        fig.write_image(image_save_path + '{}_MC{}.png'.format(pareto_front_type, MC))

    if cal_metric:
        return hypervolume_dict, percentage_dict
    else:
        return None, None

def compare_hypervolume_percent(methods_list, hypervolume_dict, percentage_dict, pareto_front_type, save_path):

    for metric in ['hypervolume','percentage']:
        if metric == 'hypervolume':
            metric_dict = hypervolume_dict
        else:
            metric_dict = percentage_dict

        fig = go.Figure()
        for type in ['train', 'test']:
            metric_mean_list, metric_std_list= [], []
            for method in methods_list:
                metric_oneMethod_oneType = [value[0] for key, value in metric_dict.items() if '{}_{}'.format(method,type) in key]

                metric_mean_list += [statistics.mean(metric_oneMethod_oneType)]
                metric_std_list += [statistics.stdev(metric_oneMethod_oneType)]

            fig.add_trace(go.Scatter(x=methods_list, y=metric_mean_list, mode='lines+markers+text',
                                     text=['{}'.format(round(metric_mean_list[i],2)) + u"\u00B1" + '{}'.format(round(metric_std_list[i],2)) for i in list(range(len(methods_list)))],
                                     textfont_size=12, textposition='top center',marker_size=10,marker_color='rgba(0, 0, 0, 0)',
                                     line=dict(color='rgba(0, 0, 0, 0)'), showlegend=False))
            if type == 'train':
                fig.add_trace(go.Scatter(x=methods_list, y=metric_mean_list, error_y=dict(type='data', array=metric_std_list, visible=True),
                                         mode='lines+markers', marker_size=10, marker_color='rgba(255, 25, 52, .9)',
                                         line=dict(color='rgba(255, 25, 52, .9)'), name='{}'.format(type),showlegend=True))
            else:
                fig.add_trace(go.Scatter(x=methods_list, y=metric_mean_list, error_y=dict(type='data', array=metric_std_list, visible=True),
                                         mode='lines+markers', marker_size=10, marker_color='rgba(0, 0, 255, .9)',
                                         line=dict(color='rgba(0, 0, 255, .9)'), name='{}'.format(type), showlegend=True))

        fig.write_image(save_path + '{}_compare_{}.png'.format(pareto_front_type,metric))

'''
def draw_pareto_front2(dataframe, MC_idx, methods_list, pareto_front_type, save_path):

    if pareto_front_type == 'NN':
        xtitle = r'$\large \text{Loss }U_n(\phi, \theta)$'
        ytitle = r'$\large \text{Batch effect }NN_n(\phi)$'
    else:
        xtitle = '-{}'.format(pareto_front_type.upper())
        ytitle = '-BE'

    dataframe_oneMC = dataframe[dataframe.MC.eq(MC_idx)]
    percent_list_train, percent_list_test = [], []

    obj1_all, obj2_all = [], []
    fig = make_subplots(rows=math.ceil(len(methods_list) / 2), cols=2, subplot_titles=tuple(methods_list),
                        horizontal_spacing=0.03, vertical_spacing=0.06, x_title=xtitle, y_title=ytitle, shared_yaxes=True, shared_xaxes=True)
    for (i,type) in enumerate(['train','test']):
        for (j, method) in enumerate(methods_list):
            dataframe_adv = dataframe_oneMC[dataframe_oneMC.method.eq(method)]
            pref_idx_list = [k + 1 for k in dataframe_adv.loc[:, 'pref_idx'].values.tolist()]

            if pareto_front_type in ['obj2', 'NN']:
                obj1 = dataframe_adv.loc[:, 'obj1_{}'.format(type)].values.tolist()
                obj2 = dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
            else:
                obj1 = dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
                obj2 = dataframe_adv.loc[:, 'be_{}'.format(type)].values.tolist()

                obj1 = [(-1)*k for k in obj1]
                obj2 = [(-1)*k for k in obj2]

            obj1_all += obj1
            obj2_all += obj2

            inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
            paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')

            pp = np.array(list(paretoPoints1))
            dp = np.array(list(dominatedPoints1))
            if type == 'train':
                percent_list_train += [pp.shape[0] / len(obj1)]
            else:
                percent_list_test += [pp.shape[0] / len(obj1)]

            if i == 0:
                marker_symbol = 'cross'
                opacity = 1
            else:
                marker_symbol = 'circle'
                opacity = 0.5

            if j < len(methods_list)-1:
                showlegend=False
            else:
                showlegend=True
            fig.add_trace(go.Scatter(x=obj1, y=obj2, opacity=opacity, text=['{}'.format(i) for i in pref_idx_list], marker_symbol=marker_symbol,
                                     mode='markers+text',textfont_size=12, textposition='top center',marker_size=10, marker_color='rgba(0, 0, 0, 0)', showlegend=False), math.ceil((j+1)/2), j - math.floor(j/2)*2 + 1)
            if dp.shape[0]>0:
                fig.add_trace(go.Scatter(x = dp[:,0].tolist(), y = dp[:,1].tolist(), marker_symbol=marker_symbol,
                                     mode='markers',marker_size=10, name='{}, yes'.format(type),marker_color='rgba(0, 0, 255, .9)', marker_opacity=opacity, showlegend=showlegend), math.ceil((j+1)/2), j - math.floor(j/2)*2 + 1)
            if pp.shape[0]>0:
                fig.add_trace(go.Scatter(x = pp[:,0].tolist(), y = pp[:,1].tolist(), marker_symbol=marker_symbol,
                                     mode='markers',marker_size=10, name='{}, no'.format(type), marker_color='rgba(255, 25, 52, .9)',marker_opacity=opacity, showlegend=showlegend), math.ceil((j+1)/2), j - math.floor(j/2)*2 + 1)

    fig.update_layout(
            width=800,
            height=350*math.ceil(len(methods_list)/2),
            margin=dict(
                l=80,
                r=90,
                b=70,
                t=30,
                pad=1
            ),
            font=dict(color='black', family='Times New Roman'),
            legend_title=dict(font=dict(family='Times New Roman', size=25, color="black"), text='train/test, dominated'),
            legend=dict(
                font=dict(
                    family='Times New Roman',
                    size=25,
                    color="black"
                )
            )
        )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    if pareto_front_type in ['obj2','NN']:
        fig.update_xaxes(tickfont=dict(size=20), title_font=dict(size=25, family='Times New Roman', color='black'),range=[min(obj1_all)-20,max(obj1_all)+20],autorange=False)
    else:
        fig.update_xaxes(tickfont=dict(size=20), title_font=dict(size=25, family='Times New Roman', color='black'),range=[min(obj1_all) - 0.05, max(obj1_all) + 0.05], autorange=False)

    if pareto_front_type == 'NN':
        fig.update_yaxes(tickfont=dict(size=20), title_font=dict(size=25, family='Times New Roman', color='black'),range=[min(obj2_all) - 0.08, max(obj2_all) + 0.08], autorange=False)
    else:
        fig.update_yaxes(tickfont=dict(size=20), title_font=dict(size=25, family='Times New Roman', color='black'), range=[min(obj2_all) - 0.08, max(obj2_all) + 0.08], autorange=False)

    fig.write_image(save_path + '/paretofront_{}_MC{}.png'.format(pareto_front_type,MC_idx))

    return percent_list_train, percent_list_test

def draw_barplot1(percent_dict, hypervolume_dict, methods_list, save_path, pareto_front_type):

    if len(methods_list)<6:
        colors_list = ['rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)','rgb(158,202,225)', 'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of colors')

    fig = make_subplots(rows=2, cols=2, subplot_titles=('train','test','train','test'), shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.06)

    if pareto_front_type == 'NN':
        fig_title = r'$\Large U_n(\phi,\theta) \text{ versus } NN_n(\phi)$'
    elif pareto_front_type == 'asw':
        fig_title = r'$\Large -\text{ASW}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'ari':
        fig_title = r'$\Large -\text{ARI}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'nmi':
        fig_title = r'$\Large -\text{NMI}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'uca':
        fig_title = r'$\Large -\text{UCA}\text{ versus }-\text{BE}$'

    for row_num in range(2):
        if row_num == 0:
            data_dict = percent_dict
        else:
            data_dict = hypervolume_dict

        for (i,data_array) in enumerate([value[0] for key, value in data_dict.items() if key in tuple(['train','test'])]):
            type = ['train','test'][i]
            data_array[~np.isfinite(data_array)] = np.nan
            y_list = np.nanmean(data_array, axis=1).tolist()
            stdev_list = np.nanstd(data_array, axis=1, ddof=1).tolist()

            print(pareto_front_type)
            if row_num == 0:
                print('percent')
            else:
                print('hypervolume')
            print(type)
            print(methods_list)
            print(y_list)
            print(stdev_list)

            fig.add_trace(go.Bar(x=methods_list, y=y_list,
                                 error_y=dict(type='data', array=stdev_list), name=type, width=[1]*len(methods_list),
                                 marker_color=colors_list[0:len(methods_list)],opacity=0.8,showlegend=False
                                 ),row=row_num+1,col=i+1
                          )

    fig.update_layout(barmode='group')
    fig.update_layout(
        width=800,
        height=700,
        margin=dict(
            l=80,
            r=1,
            b=60,
            t=80,
            pad=1
        ),
        font=dict(size=23, color='black', family='Times New Roman'),
        title={
            'text': fig_title,
            'font': {
                'size': 25
            },
            'y': 0.96,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top'},
        bargap=0,
        plot_bgcolor='rgb(255,255,255)'
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    fig.update_yaxes(title_text='percent of non-dominated points', title_font=dict(size=25, family='Times New Roman', color='black'), ticks='outside', row=1, col=1)
    fig.update_yaxes(title_text='hypervolume', title_font=dict(size=25, family='Times New Roman', color='black'), ticks='outside', row=2,col=1)
    fig.write_image(save_path + '/compare_percent_hypervolume_{}.png'.format(pareto_front_type))

def draw_barplot2(percent_dict, hypervolume_dict, methods_list, save_path, pareto_front_type):

    if len(methods_list)<6:
        colors_list = ['rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)','rgb(158,202,225)', 'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of colors')

    fig = make_subplots(rows=2, cols=2, subplot_titles=('train','test','train','test'), shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.03, vertical_spacing=0.06)

    if pareto_front_type == 'NN':
        fig_title = r'$\Large U_n(\phi,\theta) \text{ versus } NN_n(\phi)$'
    elif pareto_front_type == 'asw':
        fig_title = r'$\Large -\text{ASW}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'ari':
        fig_title = r'$\Large -\text{ARI}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'nmi':
        fig_title = r'$\Large -\text{NMI}\text{ versus }-\text{BE}$'
    elif pareto_front_type == 'uca':
        fig_title = r'$\Large -\text{UCA}\text{ versus }-\text{BE}$'

    for row_num in range(2):
        if row_num == 0:
            data_dict = percent_dict
        else:
            data_dict = hypervolume_dict

        for (i,data_array) in enumerate([value[0] for key, value in data_dict.items() if key in tuple(['train','test'])]):
            type = ['train','test'][i]
            data_array[~np.isfinite(data_array)] = np.nan
            y_list = np.nanmean(data_array, axis=1).tolist()
            stdev_list = np.nanstd(data_array, axis=1, ddof=1).tolist()

            print(pareto_front_type)
            if row_num == 0:
                print('percent')
            else:
                print('hypervolume')
            print(type)
            print(methods_list)
            print(y_list)
            print(stdev_list)

            if row_num == 1 and i == 1:
                showlegend = True
            else:
                showlegend = False

            fig.add_trace(go.Bar(x=['z10','z50'], y=y_list[0:2],
                                 error_y=dict(type='data', array=stdev_list[0:2]), name='MINE', width=[0.4]*2,
                                 marker_color=colors_list[0], opacity=0.8, showlegend=showlegend),row=row_num+1,col=i+1)
            fig.add_trace(go.Bar(x=['z10', 'z50'], y=y_list[2:4],
                                 error_y=dict(type='data', array=stdev_list[2:4]), name='MMD', width=[0.4] * 2,
                                 marker_color=colors_list[1], opacity=0.8, showlegend=showlegend), row=row_num + 1,col=i + 1)

    fig.update_layout(barmode='group')
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(
            l=80,
            r=90,
            b=60,
            t=80,
            pad=1
        ),
        font=dict(size=23, color='black', family='Times New Roman'),
        title={
            'text': fig_title,
            'font': {
                'size': 25
            },
            'y': 0.96,
            'x': 0.48,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend=dict(
            font=dict(
                family='Times New Roman',
                size=25,
                color="black"
            )
        ),
        #bargap=0,
        #plot_bgcolor='rgb(255,255,255)'
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    fig.update_yaxes(title_text='percent of non-dominated points', title_font=dict(size=25, family='Times New Roman', color='black'), ticks='outside', row=1, col=1)
    fig.update_yaxes(title_text='hypervolume', title_font=dict(size=25, family='Times New Roman', color='black'), ticks='outside', row=2,col=1)
    fig.write_image(save_path + '/compare_percent_hypervolume_{}.png'.format(pareto_front_type))

def pareto_front(dataframe, methods_list, pareto_front_type, dir_path):

    paretoPoints_percent_array_train = np.empty((len(methods_list), 0))
    paretoPoints_percent_array_test = np.empty((len(methods_list), 0))

    for MC_idx in range(dataframe.MC.max()+1):
        percent_list_train, percent_list_test = draw_pareto_front(dataframe, MC_idx, methods_list, pareto_front_type, dir_path)

        paretoPoints_percent_array_train = np.concatenate((paretoPoints_percent_array_train,np.array([percent_list_train]).reshape((len(methods_list),1))),axis=1)
        paretoPoints_percent_array_test = np.concatenate((paretoPoints_percent_array_test, np.array([percent_list_test]).reshape((len(methods_list),1))), axis=1)

    paretoPoints_percent_dict = {'train':[paretoPoints_percent_array_train],'test':[paretoPoints_percent_array_test]}
    return paretoPoints_percent_dict

def hypervolume_calculation(dataframe, methods_list, pareto_front_type):

    #get the reference point to calculate hypervolume, the hypervolume function in the pygmo package only work for minimization problem
    if pareto_front_type == 'NN':
        obj1_max = dataframe.loc[:,['obj1_train','obj1_test']].max(axis=0).max()
        obj2_max = dataframe.loc[:,['NN_train','NN_test']].max(axis=0).max()
        ref_point = [obj1_max + 5, obj2_max + 0.01]
    else:
        obj1_max = dataframe.loc[:, ['{}_train'.format(pareto_front_type), '{}_test'.format(pareto_front_type)]].min(axis=0).min()*(-1)
        obj2_max = dataframe.loc[:, ['be_train', 'be_test']].min(axis=0).min() * (-1)
        ref_point = [obj1_max + 0.01, obj2_max + 0.01]

    hypervolume_array_train = np.empty((len(methods_list), 0))
    hypervolume_array_test = np.empty((len(methods_list), 0))
    for type in ['train','test']:
        for MC_idx in range(dataframe.MC.max() + 1):
            hypervolume_list = []
            dataframe_oneMC = dataframe[dataframe.MC.eq(MC_idx)]

            for method in methods_list:
                dataframe_adv = dataframe_oneMC[dataframe_oneMC.method.eq(method)]
                if pareto_front_type == 'NN':
                    obj1_list = dataframe_adv.loc[:, 'obj1_{}'.format(type)].values.tolist()
                    obj2_list = dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
                else:
                    obj1_list = [k*(-1) for k in dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()]
                    obj2_list = [k * (-1) for k in dataframe_adv.loc[:, 'be_{}'.format(type)].values.tolist()]

                inputPoints1 = [[obj1_list[k], obj2_list[k]] for k in range(len(obj1_list))]
                hv = hypervolume(inputPoints1) #when hypervolume is calculated, the rectangles of dominated points will be covered by those of non-dominated points
                hypervolume_list += [hv.compute(ref_point)]

            if type == 'train':
                hypervolume_array_train = np.concatenate((hypervolume_array_train, np.array([hypervolume_list]).reshape((len(methods_list),1))),axis=1)
            else:
                hypervolume_array_test = np.concatenate((hypervolume_array_test, np.array([hypervolume_list]).reshape((len(methods_list),1))),axis=1)

    hypervolume_dict = {'train': [hypervolume_array_train], 'test':[hypervolume_array_test]}

    return hypervolume_dict

def compare_hypervolume_percent(dataframe, methods_list, dir_path):
    #percent is paretoPoint percent

    for pareto_front_type in ['NN', 'asw', 'nmi', 'ari', 'uca']:
        hypervolume_dict =  hypervolume_calculation(dataframe, methods_list, pareto_front_type)
        percent_dict = pareto_front(dataframe, methods_list, pareto_front_type, dir_path)

        draw_barplot1(percent_dict=percent_dict, hypervolume_dict=hypervolume_dict, methods_list=methods_list, pareto_front_type=pareto_front_type, save_path=dir_path)
        #draw_barplot2(percent_dict=percent_dict, hypervolume_dict=hypervolume_dict, methods_list=methods_list, pareto_front_type=pareto_front_type, save_path=dir_path)

def cell_type_composition(dataset_name, change_composition, save_path):
    if dataset_name == 'tabula_muris':
        dataset1 = TabulaMuris('facs', save_path=save_path)
        dataset2 = TabulaMuris('droplet', save_path=save_path)
        dataset1.subsample_genes(dataset1.nb_genes)
        dataset2.subsample_genes(dataset2.nb_genes)
        if change_composition == True:
            dataset1_labels = dataset1.__dict__['labels']
            dataset1_labels_df = pd.DataFrame.from_dict({'label': dataset1_labels[:, 0].tolist()})
            dataset1_celltypes = dataset1.__dict__['cell_types']
            dataset1_celltypes_df = pd.DataFrame.from_dict({'cell_type': dataset1_celltypes.tolist()})
            dataset1_celltypes_df['label'] = pd.Series(np.array(list(range(dataset1_celltypes_df.shape[0]))),index=dataset1_celltypes_df.index)

            delete_labels_celltypes = dataset1_celltypes_df[dataset1_celltypes_df.cell_type.isin(['granulocyte', 'nan', 'monocyte', 'hematopoietic precursor cell', 'granulocytopoietic cell'])]
            dataset1.__dict__['_X'] = sparse.csr_matrix(dataset1.__dict__['_X'].toarray()[~dataset1_labels_df.label.isin(delete_labels_celltypes.loc[:, 'label'].values.tolist())])
            for key in ['local_means', 'local_vars', 'batch_indices', 'labels']:
                dataset1.__dict__[key] = dataset1.__dict__[key][~dataset1_labels_df.label.isin(delete_labels_celltypes.loc[:,'label'].values.tolist())]
            dataset1.__dict__['n_labels'] = 18
            dataset1.__dict__['cell_types'] = np.delete(dataset1.__dict__['cell_types'], delete_labels_celltypes.loc[:, 'label'].values.tolist())

            dataset1_celltypes_new = dataset1.__dict__['cell_types']
            dataset1_celltypes_df_new = pd.DataFrame.from_dict({'cell_type': dataset1_celltypes_new.tolist()})
            dataset1_celltypes_df_new['label'] = pd.Series(np.array(list(range(dataset1_celltypes_df_new.shape[0]))),index=dataset1_celltypes_df_new.index)
            label_change = dataset1_celltypes_df.merge(dataset1_celltypes_df_new, how='right', left_on='cell_type', right_on='cell_type').iloc[:, 1:]
            label_change.columns = ['label','new_label']

            dataset1_labels_new = dataset1.__dict__['labels']
            dataset1_labels_df_new = pd.DataFrame.from_dict({'label': dataset1_labels_new[:, 0].tolist()})

            dataset1.__dict__['labels'] = dataset1_labels_df_new.merge(label_change, how='left', left_on='label', right_on='label').loc[:,'new_label'].values.reshape(dataset1.__dict__['labels'].shape[0],1)

    dataset1_labels = dataset1.__dict__['labels']
    dataset2_labels = dataset2.__dict__['labels']
    dataset1_celltypes = dataset1.__dict__['cell_types']
    dataset2_celltypes = dataset2.__dict__['cell_types']

    dataset1_labels_df = pd.DataFrame.from_dict({'label': dataset1_labels[:, 0].tolist()})
    dataset1_celltypes_df = pd.DataFrame.from_dict({'cell_type': dataset1_celltypes.tolist()})
    dataset2_labels_df = pd.DataFrame.from_dict({'label': dataset2_labels[:, 0].tolist()})
    dataset2_celltypes_df = pd.DataFrame.from_dict({'cell_type': dataset2_celltypes.tolist()})

    dataset1_celltypes_df['label'] = pd.Series(np.array(list(range(dataset1_celltypes_df.shape[0]))),index=dataset1_celltypes_df.index)
    dataset2_celltypes_df['label'] = pd.Series(np.array(list(range(dataset2_celltypes_df.shape[0]))),index=dataset2_celltypes_df.index)

    dataset1_labels_celltypes = dataset1_labels_df.merge(dataset1_celltypes_df, how='left', left_on='label',right_on='label')
    dataset2_labels_celltypes = dataset2_labels_df.merge(dataset2_celltypes_df, how='left', left_on='label',right_on='label')

    dataset1_percentage = (dataset1_labels_celltypes['cell_type'].value_counts(normalize=True) * 100).reset_index()
    dataset2_percentage = (dataset2_labels_celltypes['cell_type'].value_counts(normalize=True) * 100).reset_index()
    dataset1_percentage.columns = ['cell_type','percentage']
    dataset2_percentage.columns = ['cell_type', 'percentage']

    compare_percentage = dataset1_percentage.merge(dataset2_percentage, how='outer', left_on='cell_type', right_on='cell_type')
    print(compare_percentage.fillna(0))
'''
def main( ):

    parser = argparse.ArgumentParser(description='resultsummary')

    parser.add_argument('--dataset', type=str, default='tabula_muris',
                        help='name of dataset')

    parser.add_argument('--confounder', type=str, default='batch',
                        help='name of confounder')

    parser.add_argument('--MCs', type=int, default=20,
                        help='number of Monte Carlos')

    parser.add_argument('--pareto_front_x', type=str, default='asw',
                        help='xaxis value') #asw, ari, uca, nmi,obj1

    parser.add_argument('--pareto_front_y', type=str, default='obj2',
                        help='yaxis value') #obj2, NN, be

    parser.add_argument('--methods_list', type=str, default='MINE,MMD',
                        help='list of methods')

    parser.add_argument('--cal_metric', action='store_true', default=False,
                        help='whether to calculate hypervolume and percentage of non-dominated points')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)
    args = parser.parse_args()

    args.methods_list = args.methods_list.split(',')

    for method in args.methods_list:

        if method == 'regularize':
            dir_path = './result/{}/{}/regularize'.format(args.dataset, args.confounder)
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'nweights_weight': [{'n_weights': n, 'weight': i} for n, i in zip([10] * 10, [0, 5, 10, 50, 100, 400, 800, 1000, 2000, 4000])]
            }
        else:
            dir_path = './result/{}/{}/pareto{}'.format(args.dataset, args.confounder, method)
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10] * 10, list(range(10)))]
            }

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i in range(len(hyperparameter_experiments)):
            config_path = dir_path + '/taskid{}/config.pkl'.format(i)
            results_path = dir_path + '/taskid{}/results.pkl'.format(i)

            if os.path.isfile(config_path) and os.path.isfile(results_path):
                config = pickle.load(open(config_path, "rb"))
                results = pickle.load(open(results_path, "rb"))

                if 'obj1_minibatch_list' in results.keys():
                    del results['obj1_minibatch_list']
                if 'obj2_minibatch_list' in results.keys():
                    del results['obj2_minibatch_list']
                if method == 'regularize':
                    results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator', 'MC', 'regularize_weight'])}
                else:
                    results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator','MC', 'pref_idx'])}
                results_config.update({'method': [method]})
                results_config.update(results)

                if method == args.methods_list[0] and i == 0:
                    results_config_total = pd.DataFrame.from_dict(results_config)
                else:
                    results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
            else:
                print('method:{},taskid{}'.format(method, i))

    hypervolume_dict, percentage_dict = draw_pareto_front(dataframe=results_config_total, methods_list=args.methods_list,
                                        pareto_front_x=args.pareto_front_x, pareto_front_y=args.pareto_front_y,
                                        cal_metric=args.cal_metric, save_path=os.path.dirname(dir_path)+'/')

    compare_hypervolume_percent(methods_list=args.methods_list, hypervolume_dict=hypervolume_dict, percentage_dict=percentage_dict,
                                pareto_front_type=args.pareto_front_type, save_path=os.path.dirname(dir_path)+'/')

# Run the actual program
if __name__ == "__main__":
    main()
