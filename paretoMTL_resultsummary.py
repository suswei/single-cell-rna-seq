import os
import itertools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scvi.dataset.tabula_muris import TabulaMuris
from scipy import sparse
import pickle
from pygmo import hypervolume
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import statistics
import cv2

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

    subset_dataframe = dataframe[(dataframe == np.inf).any(axis=1)]
    if subset_dataframe.shape[0]>0:
        dataframe = pd.concat([dataframe, subset_dataframe]).drop_duplicates(keep=False)

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

            if 'pareto' in method:
                subset_string = method.replace('pareto', 'regularize')
                if subset_string in methods_list:
                    dataframe_oneMC_extreme = dataframe_oneMC[dataframe_oneMC.method.eq(subset_string) & dataframe_oneMC.nweight.isin([0,11])]
                    if dataframe_oneMC_extreme.shape[0]>0:
                        dataframe_oneMC_extreme.pref_idx = dataframe_oneMC_extreme.nweight
                        dataframe_oneMC_oneMethod['pref_idx'] = dataframe_oneMC_oneMethod.apply(lambda row: row.pref_idx + 1, axis=1)
                        dataframe_oneMC_oneMethod = pd.concat([dataframe_oneMC_extreme, dataframe_oneMC_oneMethod],axis=0).sort_values('pref_idx')

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
                            
                if 'pareto' in method:
                    index_list = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:,'pref_idx'])]
                else:
                    index_list = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:, 'nweight'])]

                inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
                index_list_Pareto = []
                for paretoPoint in paretoPoints1:
                    for index, inputPoint in zip(index_list, inputPoints1):
                        if paretoPoint[0] == inputPoint[0] and paretoPoint[1] == inputPoint[1]:
                            index_list_Pareto += [index]

                if cal_metric:
                    # when hypervolume is calculated, the rectangles of dominated points will be covered by those of non-dominated points
                    hv = hypervolume(inputPoints1)
                    hypervolume_dict.update({'{}_{}_MC{}'.format(method, type, MC): [hv.compute(ref_point)]})

                    percentage_dict.update({'{}_{}_MC{}'.format(method, type, MC): [len(list(paretoPoints1))/12]})

                if 'pareto' in method:
                    index_list_Pareto = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:, 'pref_idx'])]
                else:
                    index_list_Pareto = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:, 'nweight'])]

                if i == 0:
                    marker_symbol = 'circle'
                elif i == 1:
                    marker_symbol = 'cross'
                elif i == 2:
                    marker_symbol = 'triangle-up'
                elif i == 3:
                    marker_symbol = 'diamond'

                if j == 0:
                    marker_color = 'rgba(255, 25, 52, .9)'
                elif j == 1:
                    marker_color = 'rgba(0, 0, 255, .9)'

                if pp.shape[0] > 0:
                    fig.add_trace(go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(), text=['{}'.format(i) for i in index_list_Pareto],
                                             mode='markers+text', textfont_size=12, textposition='top center', marker_size=10,
                                             marker_symbol=marker_symbol, marker_color='rgba(0, 0, 0, 0)', showlegend=False))

                if pp.shape[0] > 0:
                    fig.add_trace(go.Scatter(x=pp[:, 0].tolist(), y=pp[:, 1].tolist(), mode='markers',
                                   marker_size=10, marker_symbol=marker_symbol, name='{},{}'.format(method, type),
                                   marker_color=marker_color, opacity=0.75, showlegend=True))


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

        if pareto_front_x == 'obj1':
            xaxes_title = r'$\large \text{Loss }U_n(\phi, \theta)$'
            xvalue_adjust = 50
        else:
            xaxes_title = 'negative {}'.format(pareto_front_x)
            xvalue_adjust = 0.1

        if pareto_front_y == 'obj2':
            yaxes_title = r'$\large \text{Batch effect }V_n(\phi)$'
        elif pareto_front_y == 'NN':
            yaxes_title = r'$\large \text{Batch effect }NN_n(\phi)$'
        else:
            yaxes_title = 'negative BE'
        yvalue_adjust = 0.1

        fig.update_xaxes(tickfont=dict(size=20),title_text=xaxes_title,
                         title_font=dict(size=25, family='Times New Roman', color='black'),
                         range=[min(obj1_all) - xvalue_adjust, max(obj1_all) + xvalue_adjust], autorange=False) #tick0=15200,dtick=100
        fig.update_yaxes(tickfont=dict(size=20),title_text=yaxes_title,
                         title_font=dict(size=25, family='Times New Roman', color='black'),
                         range=[min(obj2_all) - yvalue_adjust, max(obj2_all) + yvalue_adjust], autorange=False)

        fig.write_image(image_save_path + '{}_{}_MC{}_paretofront.png'.format(pareto_front_x, pareto_front_y, MC))

    if cal_metric:
        return hypervolume_dict, percentage_dict
    else:
        return None, None

def compare_hypervolume_percent(methods_list, hypervolume_dict, percentage_dict, pareto_front_x, pareto_front_y, save_path):

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

            if type == 'train':
                marker_color = 'rgba(255, 25, 52, .9)'
            elif type == 'test':
                marker_color = 'rgba(0, 0, 255, .9)'

            fig.add_trace(go.Bar(x=methods_list,y=metric_mean_list,name='{}'.format(type),
                                 error_y= dict(type='data', array=metric_std_list, visible=True),
                                 marker_color=marker_color))

        if pareto_front_x == 'obj1' and pareto_front_y == 'obj2':
            title_text = r'$\Large \text{loss }U_n \text{ vs } \text{loss} V_n$'
        elif pareto_front_x == 'obj1' and pareto_front_y == 'NN':
            title_text = r'$\Large \text{loss }U_n \text{ vs } NN_n$'
        elif pareto_front_x == 'obj1' and pareto_front_y == 'be':
            title_text = r'$\Large \text{loss } U_n \text{ vs negative be}$'
        elif pareto_front_x != 'obj1' and pareto_front_y == 'obj2':
            title_text = r'$\Large \text{{negative {} vs loss }} V_n$'.format(pareto_front_x)
        elif pareto_front_x != 'obj1' and pareto_front_y == 'NN':
            title_text = r'$\Large \text{{negative {} vs NN_n}}$'.format(pareto_front_x)
        elif pareto_front_x != 'obj1' and pareto_front_y == 'be':
            title_text = r'$\Large \text{{negative {} vs negative be}}$'.format(pareto_front_x)

        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          title_font_family="Times New Roman",
                          title={'text': title_text,
                                 'y':0.85,
                                 'x':0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'},
                          legend = dict(font = dict(family = "Times New Roman",
                                                    size = 20)))
        fig.update_xaxes(tickfont=dict(size=20, family='Times New Roman'))
        fig.update_yaxes(tickfont=dict(size=20, family='Times New Roman'), title_text='{}'.format(metric),
                         title_font=dict(size=20, family='Times New Roman', color='black'))

        img_save_path = save_path
        for method in methods_list:
            img_save_path = img_save_path + '{}_'.format(method)
        fig.write_image(img_save_path + '{}_{}_{}.png'.format(pareto_front_x, pareto_front_y, metric))

def draw_inputPoints(dataframe, methods_list, pareto_front_x, pareto_front_y, save_path):

    #check any rows that contain infinity value
    subset_dataframe = dataframe[(dataframe == np.inf).any(axis=1)]
    if subset_dataframe.shape[0] > 0:
        dataframe = pd.concat([dataframe, subset_dataframe]).drop_duplicates(keep=False)

    for MC in range(dataframe.MC.max() + 1):
        dataframe_oneMC = dataframe[dataframe.MC.eq(MC)]

        for (i, type) in enumerate(['train', 'test']):
            image_save_path = save_path
            fig = go.Figure()
            obj1_all, obj2_all = [], []
            for (j, method) in enumerate(methods_list):
                image_save_path = image_save_path + '{}_'.format(method)

                dataframe_oneMC_oneMethod = dataframe_oneMC[dataframe_oneMC.method.eq(method)]

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
                inputPoints1 = np.array(list(inputPoints1))

                if 'pareto' in method:
                    index_list_Pareto = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:, 'pref_idx'])]
                else:
                    index_list_Pareto = [int(k) for k in list(dataframe_oneMC_oneMethod.loc[:, 'nweight'])]

                if j == 0:
                    marker_symbol = 'circle'
                    marker_color = 'rgba(255, 0, 0)'
                elif j == 1:
                    marker_symbol = 'cross'
                    marker_color = 'rgb(0,255,0)'
                elif j == 2:
                    marker_symbol = 'triangle-up'
                    marker_color = 'rgba(0, 0, 255)'
                elif j == 3:
                    marker_symbol = 'diamond'
                    marker_color = 'rgb(0,255,255)'

                if inputPoints1.shape[0] > 0:
                    fig.add_trace(go.Scatter(x=inputPoints1[:, 0].tolist(), y=inputPoints1[:, 1].tolist(), mode='markers',
                                             marker_size=[k*2+10 for k in index_list_Pareto], marker_symbol=marker_symbol,
                                             name='{},{}'.format(method, type), marker_color=marker_color,
                                             opacity=0.5, showlegend=True))

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
                    'text': 'all input points',
                    'font': {
                        'size': 25
                    },
                    'y': 0.94,
                    'x': 0.42,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                legend=dict(
                    x=0.5,
                    y=0.8,
                    font=dict(
                        family='Times New Roman',
                        size=20,
                        color='black'
                    )
                )
            )

            if pareto_front_x == 'obj1':
                xaxes_title = r'$\large \text{Loss }U_n(\phi, \theta)$'
                xvalue_adjust = 100
            else:
                xaxes_title = 'negative {}'.format(pareto_front_x)
                xvalue_adjust = 0.1

            if pareto_front_y == 'obj2':
                yaxes_title = r'$\large \text{Batch effect }V_n(\phi)$'
            elif pareto_front_y == 'NN':
                yaxes_title = r'$\large \text{Batch effect }NN_n(\phi)$'
            else:
                yaxes_title = 'negative BE'
            yvalue_adjust = 0.1

            fig.update_xaxes(tickfont=dict(size=20), title_text=xaxes_title,
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj1_all) - xvalue_adjust, max(obj1_all) + xvalue_adjust],
                             autorange=False)  # tick0=15200,dtick=100
            fig.update_yaxes(tickfont=dict(size=20), title_text=yaxes_title,
                             title_font=dict(size=25, family='Times New Roman', color='black'),
                             range=[min(obj2_all) - yvalue_adjust, max(obj2_all) + yvalue_adjust], autorange=False)

            fig.write_image(image_save_path + '{}_{}_MC{}_{}_allInputPoints.png'.format(pareto_front_x, pareto_front_y, MC, type))

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
                        help='yaxis value') #obj2 (MINE or stdMMD), NN, be

    parser.add_argument('--methods_list', type=str, default='paretoMINE,paretoMMD',
                        help='list of methods')

    parser.add_argument('--cal_metric', action='store_true', default=False,
                        help='whether to calculate hypervolume and percentage of non-dominated points')

    parser.add_argument('--diagnosis', action='store_true', default=False,
                        help='whether to visualize diagnosis plot')

    parser.add_argument('--draw_all_inputPoints', action='store_true', default=False,
                        help='whether to draw all input points or not')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)
    args = parser.parse_args()

    args.methods_list = args.methods_list.split(',')

    for method in args.methods_list:
        dir_path = './result/{}/{}/{}'.format(args.dataset, args.confounder, method)
        if 'regularize' in method:
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'nweight_weight': [{'n_weight': n, 'weight': i} for n, i in zip(list(range(12)), [0, 1/11, 2/11, 3/11, 4/11, 5/11, 6/11, 7/11, 8/11, 9/11, 10/11, 1])]
            }
        else:
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10] * 10, list(range(10)))]
            }

        keys, values = zip(*hyperparameter_config.items())
        hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if args.diagnosis == True:
            initial_frame = False

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
                if 'regularize' in method:
                    results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator', 'MC', 'nweight'])}
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

            diagnosis_img_path = dir_path + '/taskid{}/totalloss_train_test_error.png'.format(i)
            if args.diagnosis == True:
                if os.path.isfile(diagnosis_img_path) and initial_frame==False:
                    frame = cv2.imread(dir_path + '/taskid0/totalloss_train_test_error.png')
                    height, width, layers = frame.shape
                    video = cv2.VideoWriter(dir_path + '/diagnosis_video.avi', 0, 1, (width, height))
                    initial_frame = True
                elif os.path.isfile(diagnosis_img_path) and initial_frame==True:
                    img = cv2.imread(dir_path + '/taskid{}/totalloss_train_test_error.png'.format(i))
                    if 'pareto' in method:
                        cv2.putText(img, 'MC:{}, pref_idx: {}'.format(hyperparameter_experiments[i]['MC'],
                                    hyperparameter_experiments[i]['npref_prefidx']['pref_idx']),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    else:
                        cv2.putText(img, 'MC:{}, weight_idx: {}'.format(hyperparameter_experiments[i]['MC'],
                                    hyperparameter_experiments[i]['nweight_weight']['n_weight']),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    video.write(img)
                else:
                    continue

        if args.diagnosis == True and initial_frame==True:
            cv2.destroyAllWindows()
            video.release()
    '''
    hypervolume_dict, percentage_dict = draw_pareto_front(dataframe=results_config_total, methods_list=args.methods_list,
                                       pareto_front_x=args.pareto_front_x, pareto_front_y=args.pareto_front_y,
                                       cal_metric=args.cal_metric, save_path=os.path.dirname(dir_path)+'/')
    if args.cal_metric==True:
        compare_hypervolume_percent(methods_list=args.methods_list, hypervolume_dict=hypervolume_dict, percentage_dict=percentage_dict,
                               pareto_front_x=args.pareto_front_x, pareto_front_y=args.pareto_front_y, save_path=os.path.dirname(dir_path)+'/')
    '''
    if args.draw_all_inputPoints == True:
        draw_inputPoints(dataframe=results_config_total, methods_list=args.methods_list,
                         pareto_front_x=args.pareto_front_x, pareto_front_y=args.pareto_front_y,
                         save_path=os.path.dirname(dir_path)+'/')

# Run the actual program
if __name__ == "__main__":
    main()
