import os
import itertools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
from pygmo import hypervolume
import plotly.graph_objects as go
import argparse
import cv2

'''
for x_axis in ['ari']:
    for ParetoCandidates_ParetoPoints in ['candidates', 'paretopoints']:
        image_path = './result/tabula_muris/batch/pretrain200_lr0.001/paretoMINE_paretoMMD_{}_be_MC3_{}'.format(x_axis, ParetoCandidates_ParetoPoints)
        image = cv2.imread(image_path + '.png')
        start_point = (510, 530)
        end_point = (360, 530)
        color = (178, 190, 181)
        thickness = 2
        image = cv2.arrowedLine(image, start_point, end_point,
                            color, thickness, tipLength = 0.05)
        start_point = (510, 350)
        end_point = (510, 500)
        color = (178, 190, 181)
        thickness = 2
        image = cv2.arrowedLine(image, start_point, end_point,
                            color, thickness, tipLength = 0.05)
        cv2.imwrite(image_path + '_modified.png', image)
'''

def max_min_value(dataframe, pareto_front_x, pareto_front_y):

    if pareto_front_x == 'obj1':
        obj1_max = dataframe.loc[:, ['obj1_train_std', 'obj1_test_std']].max(axis=0).max() #_std
        obj1_min = dataframe.loc[:, ['obj1_train_std', 'obj1_test_std']].min(axis=0).min()
    else:
        obj1_max = dataframe.loc[:, ['{}_train'.format(pareto_front_x), '{}_test'.format(pareto_front_x)]].min(axis=0).min() * (-1)
        obj1_min = dataframe.loc[:, ['{}_train'.format(pareto_front_x), '{}_test'.format(pareto_front_x)]].max(axis=0).max() * (-1)

    if pareto_front_y == 'obj2':
        obj2_max = dataframe.loc[:, ['obj2_train_std', 'obj2_test_std']].max(axis=0).max()
        obj2_min = dataframe.loc[:, ['obj2_train_std', 'obj2_test_std']].min(axis=0).min()
    elif pareto_front_y == 'NN':
        obj2_max = dataframe.loc[:, ['NN_train', 'NN_test']].max(axis=0).max()
        obj2_min = dataframe.loc[:, ['NN_train', 'NN_test']].min(axis=0).min()
    elif pareto_front_y == 'be':
        obj2_max = dataframe.loc[:, ['be_train', 'be_test']].min(axis=0).min() * (-1)
        obj2_min = dataframe.loc[:, ['be_train', 'be_test']].max(axis=0).max() * (-1)

    return obj1_max, obj2_max, obj1_min, obj2_min

def Reference_GridValues(dataframe_dict, methods_list, pareto_front_x, pareto_front_y, draw_extreme_points, mu):

    results_config_AllMethods = dataframe_dict['results_config_AllMethods']
    results_config_subset = results_config_AllMethods[results_config_AllMethods.method.isin(methods_list)]
    results_obj1_max, results_obj2_max, results_obj1_min, results_obj2_min = max_min_value(results_config_subset, pareto_front_x, pareto_front_y)
    obj1_max = results_obj1_max
    obj2_max = results_obj2_max
    obj1_min = results_obj1_min
    obj2_min = results_obj2_min

    if draw_extreme_points:
        if any('MINE' in s for s in methods_list):
            results_config_ExtremePointsMINE = dataframe_dict['results_config_ExtremePointsMINE']
            ExtremePointsMINE_obj1_max, ExtremePointsMINE_obj2_max, ExtremePointsMINE_obj1_min, ExtremePointsMINE_obj2_min = max_min_value(results_config_ExtremePointsMINE, pareto_front_x, pareto_front_y)
            obj1_max = max(obj1_max, ExtremePointsMINE_obj1_max)
            obj2_max = max(obj2_max, ExtremePointsMINE_obj2_max)
            obj1_min = min(obj1_min, ExtremePointsMINE_obj1_min)
            obj2_min = min(obj2_min, ExtremePointsMINE_obj2_min)

        if any('MMD' in s for s in methods_list):
            results_config_ExtremePointsMMD = dataframe_dict['results_config_ExtremePointsMMD']
            ExtremePointsMMD_obj1_max, ExtremePointsMMD_obj2_max, ExtremePointsMMD_obj1_min, ExtremePointsMMD_obj2_min = max_min_value(results_config_ExtremePointsMMD, pareto_front_x, pareto_front_y)
            obj1_max = min(max(obj1_max, ExtremePointsMMD_obj1_max),1000)
            obj2_max = max(obj2_max, ExtremePointsMMD_obj2_max)
            obj1_min = min(obj1_min, ExtremePointsMMD_obj1_min)
            obj2_min = min(obj2_min, ExtremePointsMMD_obj2_min)

    obj1_max += 0.001
    obj2_max += 0.001
    obj1_min -= 0.001
    obj2_min -= 0.001
    print('obj1_max:{}, obj1_min: {}, obj2_max: {}, obj2_min: {}'.format(obj1_max, obj1_min, obj2_max, obj2_min))
    ref_point = [obj1_max, obj2_max]
    GridValues = [[obj1_min + k*mu for k in range(int((obj1_max - obj1_min)/mu)+1)] + [obj1_max], [obj2_min + k*mu for k in range(int((obj2_max - obj2_min)/mu)+1)] + [obj2_max]]

    return ref_point, GridValues

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

def NDC_fun(ParetoPoints, GridValues):
    Grid_Index = []
    for ParetoPoint in ParetoPoints:
        for i in range(len(GridValues[0])):
            if ParetoPoint[0]>=GridValues[0][i] and ParetoPoint[0]<GridValues[0][i+1]:
                Grid_Index_Obj1 = i
                break
        for j in range(len(GridValues[1])):
            if ParetoPoint[1]>=GridValues[1][j] and ParetoPoint[1]<GridValues[1][j+1]:
                Grid_Index_Obj2 = j
                break
        Grid_Index += [[Grid_Index_Obj1, Grid_Index_Obj2]]

    NDC_value = len(set(tuple(x) for x in Grid_Index))
    return NDC_value

def pareto_front(inputPoints, index_list, ReferencePoints, GridValues):

    inputPoints_copy = inputPoints.copy()
    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints_copy, dominates, False, 'min')
    pp = np.array(list(paretoPoints1))
    # dp = np.array(list(dominatedPoints1))

    index_list_Pareto = []
    print('inputPoints_copy: {}'.format(inputPoints_copy))
    print('inputPoints: type: {}, value: {}'.format(type(inputPoints), inputPoints))
    for (index, inputPoint) in zip(index_list, inputPoints):
        print('index: {}'.format(index))
        print('inputPoint: {}'.format(inputPoint))
    for paretoPoint in list(paretoPoints1):
        for (index, inputPoint) in zip(index_list, inputPoints):
            if paretoPoint[0] == inputPoint[0] and paretoPoint[1] == inputPoint[1]:
                index_list_Pareto += [index]

    percentage_value = len(list(paretoPoints1)) / 12
    # when hypervolume is calculated, the rectangles of dominated points will be covered by those of non-dominated points
    hv = hypervolume(inputPoints)
    hypervolume_value = hv.compute(ReferencePoints)

    NDC_value = NDC_fun(list(paretoPoints1), GridValues)

    return pp, index_list_Pareto, percentage_value, hypervolume_value, NDC_value

def objective_list(objective_name, train_test, data_frame):
    if objective_name in ['obj1','obj2']:
        obj_list = data_frame.loc[:, '{}_{}_std'.format(objective_name, train_test)].values.tolist() #_std
    else:
        obj_list = data_frame.loc[:, '{}_{}'.format(objective_name, train_test)].values.tolist()
    return obj_list

def CollectPoints_AllMethods(dataframe_dict, MC, methods_list, pareto_front_x, pareto_front_y, draw_extreme_points,
                             ParetoCandidates_ParetoPoints, ReferencePoints, GridValues):

    results_config_AllMethods = dataframe_dict['results_config_AllMethods']

    '''
    subset_results_config_AllMethods = results_config_AllMethods[(results_config_AllMethods == np.inf).any(axis=1)]
    if subset_results_config_AllMethods.shape[0] > 0:
        results_config_AllMethods= pd.concat([results_config_AllMethods, subset_results_config_AllMethods]).drop_duplicates(keep=False)
    '''

    if draw_extreme_points:
        if any('MINE' in s for s in methods_list):
            results_config_ExtremePointsMINE = dataframe_dict['results_config_ExtremePointsMINE']
            results_config_ExtremePointsMINE.sort_values(['MC', 'index'],ascending=[True, True], inplace=True)
        if any('MMD' in s for s in methods_list):
            results_config_ExtremePointsMMD = dataframe_dict['results_config_ExtremePointsMMD']
            results_config_ExtremePointsMMD.sort_values(['MC', 'index'], ascending=[True, True], inplace=True)

    results_config_oneMC_AllMethods = results_config_AllMethods[results_config_AllMethods.MC.eq(MC)]

    if ParetoCandidates_ParetoPoints == 'ParetoCandidates':
        ParetoCandidates_AllMethods = {}
        ParetoCandidatesIndices_AllMethods = {}

    if ParetoCandidates_ParetoPoints == 'ParetoPoints':
        ParetoPoints_AllMethods = {}
        ParetoPointsIndices_AllMethods = {}
        percentage_AllMethods = {}
        hypervolume_AllMethods = {}
        NDC_AllMethods = {}

    for (i, method) in enumerate(methods_list):

        results_config_oneMC_oneMethod = results_config_oneMC_AllMethods[results_config_oneMC_AllMethods.method.eq(method)]
        if draw_extreme_points:
            if 'MINE' in method:
                results_config_ExtremePointsMINE_oneMC = results_config_ExtremePointsMINE[results_config_ExtremePointsMINE.MC.eq(MC)]
            elif 'MMD' in method:
                results_config_ExtremePointsMMD_oneMC = results_config_ExtremePointsMMD[results_config_ExtremePointsMMD.MC.eq(MC)]
                print('MC: {}, method: {}'.format(MC, method))
                print('results_config_ExtremePointsMMD_oneMC: {}'.format(results_config_ExtremePointsMMD_oneMC))

        for train_test in ['train','test']:

            obj1_list = objective_list(pareto_front_x, train_test, results_config_oneMC_oneMethod)
            obj2_list = objective_list(pareto_front_y, train_test, results_config_oneMC_oneMethod)

            obj1_list_test_convergence = objective_list('obj1', train_test, results_config_oneMC_oneMethod)

            if draw_extreme_points:
                if 'MINE' in method:
                    Input_DataFrame = results_config_ExtremePointsMINE_oneMC
                elif 'MMD' in method:
                    Input_DataFrame = results_config_ExtremePointsMMD_oneMC

                obj1_first_last = objective_list(pareto_front_x, train_test, Input_DataFrame)
                obj2_first_last = objective_list(pareto_front_y, train_test, Input_DataFrame)

                obj1_first_last_test_convergence = objective_list('obj1', train_test, Input_DataFrame)

                obj1 = [obj1_first_last[0]] + obj1_list + [obj1_first_last[-1]]
                obj2 = [obj2_first_last[0]] + obj2_list + [obj2_first_last[-1]]

                obj1_test_convergence = [obj1_first_last_test_convergence[0]] + obj1_list_test_convergence + [obj1_first_last_test_convergence[-1]]

                if 'MMD' in method:
                    print('obj1: {}'.format(obj1))
                    print('obj2: {}'.format(obj2))
            else:
                obj1 = obj1_list
                obj2 = obj2_list

                obj1_test_convergence = obj1_list_test_convergence

            if pareto_front_x == 'obj1' and pareto_front_y == 'be':
                obj2 = [(-1) * k for k in obj2]
            elif pareto_front_x != 'obj1' and pareto_front_y != 'be':
                obj1 = [(-1) * k for k in obj1]
            elif pareto_front_x != 'obj1' and pareto_front_y == 'be':
                obj1 = [(-1) * k for k in obj1]
                obj2 = [(-1) * k for k in obj2]


            #check if obj1_train_std or obj1_test_std>1000, if so, we think the training process is not convergent
            non_convergent_index = [i for i, j in enumerate(obj1_test_convergence) if j > 1000]
            if len(non_convergent_index)>0:
                for i in reversed(non_convergent_index):
                    del obj1[i]
                for i in reversed(non_convergent_index):
                    del obj2[i]

            inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]

            if draw_extreme_points:
                index_list = [0] + [int(k)+1 for k in list(results_config_oneMC_oneMethod.loc[:, 'index'])] + [11]
            else:
                index_list = [int(k) for k in list(results_config_oneMC_oneMethod.loc[:, 'index'])]

            if len(non_convergent_index)>0:
                for i in reversed(non_convergent_index):
                    del index_list[i]

            if ParetoCandidates_ParetoPoints == 'ParetoCandidates':
                ParetoCandidates_AllMethods.update({'{}_{}'.format(method, train_test): np.array(inputPoints1)})
                ParetoCandidatesIndices_AllMethods.update({'{}_{}'.format(method, train_test): index_list})

            if ParetoCandidates_ParetoPoints == 'ParetoPoints':
                ParetoPoints1, index_list_Pareto, percentage_value, hypervolume_value, NDC_value = pareto_front(inputPoints = inputPoints1,
                                        index_list = index_list, ReferencePoints = ReferencePoints, GridValues=GridValues)
                print('MC:{}, method: {}, hypervolume: {}'.format(MC, method, hypervolume_value))
                ParetoPoints_AllMethods.update({'{}_{}'.format(method, train_test): ParetoPoints1})
                ParetoPointsIndices_AllMethods.update({'{}_{}'.format(method, train_test): index_list_Pareto})
                percentage_AllMethods.update({'{}_{}'.format(method, train_test): [percentage_value]})
                hypervolume_AllMethods.update({'{}_{}'.format(method, train_test): [hypervolume_value]})
                NDC_AllMethods.update({'{}_{}'.format(method, train_test): [NDC_value]})

    if ParetoCandidates_ParetoPoints == 'ParetoCandidates':
        return ParetoCandidates_AllMethods, ParetoCandidatesIndices_AllMethods
    if ParetoCandidates_ParetoPoints == 'ParetoPoints':
        return ParetoPoints_AllMethods, ParetoPointsIndices_AllMethods, percentage_AllMethods, hypervolume_AllMethods, NDC_AllMethods

def draw_scatter_plot(points_dict, index_dict, methods_list, xaxis, yaxis, MC, save_path, ParetoCandidates_ParetoPoints):

    fig = go.Figure()
    obj1_all, obj2_all = [], []

    for (i, method) in enumerate(methods_list):
        save_path = save_path + '{}_'.format(method)

        if i == 0:
            marker_symbol = 'circle'
        elif i == 1:
            marker_symbol = 'cross'
        elif i == 2:
            marker_symbol = 'triangle-up'
        elif i == 3:
            marker_symbol = 'diamond'

        for train_test in ['train','test']:

            if train_test == 'train':
                marker_color = 'rgba(255, 25, 52, .9)'
            else:
                marker_color = 'rgba(0, 0, 255, .9)'

            key = method + '_' + train_test
            points = points_dict[key]
            index_list = index_dict[key]

            if method == 'paretoMINE':
                method_name = 'ParetoMTL+MINE'
            elif method == 'paretoMMD':
                method_name = 'ParetoMTL+MMD'
            elif method == 'regularizeMINE':
                method_name = 'scalarization+MINE'
            elif method == 'regularizeMMD':
                method_name = 'scalarization+MMD'

            if points.shape[0] > 0:
                if train_test == 'test':
                    y_jitter = [k+0.01 for k in points[:, 1].tolist()]
                else:
                    y_jitter = points[:, 1].tolist()
                fig.add_trace(go.Scatter(x=points[:, 0].tolist(), y=y_jitter, mode='markers',
                                         marker_size=[k * 1 + 10 for k in index_list], marker_symbol=marker_symbol,
                                         name='{}, {}'.format(method_name, train_test), marker_color=marker_color,
                                         opacity=0.7, showlegend=True))

            obj1_all += list(points[:, 0])
            obj2_all += y_jitter

    if ParetoCandidates_ParetoPoints == 'ParetoCandidates':
        title_text = 'all Pareto candidates'
        save_path_text = 'candidates'
    elif ParetoCandidates_ParetoPoints == 'ParetoPoints':
        title_text = 'non-dominated points'
        save_path_text = 'paretopoints'

    fig.update_layout(
        width=600,
        height=600,
        margin=dict(
            l=80,
            r=90,
            b=70,
            t=70,
            pad=1
        ),
        font=dict(color='black', family='Times New Roman'),
        title={
            'text': title_text,
            'font': {
                'size': 25
            },
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend=dict(
            x=0.52,
            y=0.98,
            font=dict(
                family='Times New Roman',
                size=16,
                color='black'
            )
        )
    )

    if xaxis == 'obj1':
        xaxes_title = r'$\large \text{Loss }\overline{U}_{n}(\phi, \theta)$'
    else:
        xaxes_title = 'negative {}'.format(xaxis.upper())
    xvalue_adjust = 0.02

    if yaxis == 'obj2':
        yaxes_title = r'$\large \text{Batch effect }\overline{V}_{n}(\phi)$'
    elif yaxis == 'NN':
        yaxes_title = r'$\large \text{Batch effect }NN$'
    else:
        yaxes_title = 'negative BE'
    yvalue_adjust = 0.05

    fig.update_xaxes(tickfont=dict(size=20), title_text=xaxes_title,
                     title_font=dict(size=25, family='Times New Roman', color='black'),
                     range=[min(obj1_all) - xvalue_adjust, max(obj1_all) + xvalue_adjust],
                     autorange=False)  # tick0=15200,dtick=100
    fig.update_yaxes(tickfont=dict(size=20), title_text=yaxes_title,
                     title_font=dict(size=25, family='Times New Roman', color='black'),
                     range=[min(obj2_all) - yvalue_adjust, max(obj2_all) + yvalue_adjust], autorange=False)

    fig.write_image(save_path + '{}_{}_MC{}_{}.png'.format(xaxis, yaxis, MC, save_path_text))

def std_obj1_obj2(dataframe):
    dataframe['obj1_train_std']=dataframe.apply(lambda row: (row.obj1_train - row.obj1_min)/ (row.obj1_max - row.obj1_min), axis=1)
    dataframe['obj1_test_std'] = dataframe.apply(lambda row: (row.obj1_test - row.obj1_min) / (row.obj1_max - row.obj1_min), axis=1)
    dataframe['obj2_train_std']=dataframe.apply(lambda row: (row.obj2_train - row.obj2_min)/ (row.obj2_max - row.obj2_min), axis=1)
    dataframe['obj2_test_std'] = dataframe.apply(lambda row: (row.obj2_test - row.obj2_min) / (row.obj2_max - row.obj2_min), axis=1)
    return dataframe

def load_result(dir_path, hyperparameter_config, method):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(hyperparameter_experiments)):

        config_path = dir_path + '/taskid{}/config.pkl'.format(i)
        results_path = dir_path + '/taskid{}/results.pkl'.format(i)

        if os.path.isfile(config_path) and os.path.isfile(results_path):
            config = pickle.load(open(config_path, "rb"))
            results = pickle.load(open(results_path, "rb"))

            print(config)

            if 'regularize' in method:
                config_keys = ['adv_estimator', 'MC', 'nweight', 'obj1_min', 'obj1_max', 'obj2_min', 'obj2_max']
            elif 'pareto' in method:
                config_keys = ['adv_estimator', 'MC', 'pref_idx', 'obj1_min', 'obj1_max', 'obj2_min', 'obj2_max']
            elif 'extreme_points' in method:
                config_keys = ['adv_estimator', 'MC', 'weight']

            results_config = {key: [value] for key, value in config.items() if key in tuple(config_keys)}
            results_config.update({'method': [method]})
            results_config.update(results)

            if i == 0:
                results_config_total = pd.DataFrame.from_dict(results_config)
            else:
                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
        else:
            print('method:{},taskid{}'.format(method, i))

    if 'regularize' in method:
        results_config_total_std = std_obj1_obj2(results_config_total)
        results_config_total_std.rename(columns={'nweight': 'index'}, inplace=True)
        return results_config_total_std
    elif 'pareto' in method:
        results_config_total_std = std_obj1_obj2(results_config_total)
        results_config_total_std.rename(columns={'pref_idx': 'index'}, inplace=True)
        return results_config_total_std
    else:
        return results_config_total

def load_result_ExtremePoints(adv_estimator, MCs, dir_path, methods_list):
    method = 'extreme_points_{}'.format(adv_estimator)
    new_dir_path = dir_path + '/{}'.format(method)
    hyperparameter_config = {
        'MC': list(range(MCs)),
        'weight': [0, 1]
    }
    results_config_ExtremePoints = load_result(new_dir_path, hyperparameter_config, method)

    #results for two extreme points do not contain the max and min values of obj1 and obj2 for standardization
    #read in such values for standardization
    min_max_method = [k for k in methods_list if adv_estimator in k][0]
    min_max_config_path = dir_path + '/{}/taskid0/config.pkl'.format(min_max_method)
    min_max_config = pickle.load(open(min_max_config_path, "rb"))

    results_config_ExtremePoints['obj1_max'] = min_max_config['obj1_max']
    results_config_ExtremePoints['obj1_min'] = min_max_config['obj1_min']
    results_config_ExtremePoints['obj2_max'] = min_max_config['obj2_max']
    results_config_ExtremePoints['obj2_min'] = min_max_config['obj2_min']

    results_config_ExtremePoints_std = std_obj1_obj2(results_config_ExtremePoints)

    results_config_ExtremePoints_std['index'] = results_config_ExtremePoints_std.apply(lambda row: int(row.weight * 11), axis=1)

    print('results_config_total_std shape: {},{}'.format(results_config_ExtremePoints_std.shape[0],results_config_ExtremePoints_std.shape[1]))
    print('columns are: {}'.format(results_config_ExtremePoints_std.columns))
    print('obj1_max: {}'.format(set(results_config_ExtremePoints_std.loc[:,'obj1_max'])))
    print('obj2_min: {}'.format(set(results_config_ExtremePoints_std.loc[:, 'obj2_min'])))

    return results_config_ExtremePoints_std

def diagnosis(dir_path, hyperparameter_config, method):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    initial_frame = False

    for i in range(len(hyperparameter_experiments)):
        diagnosis_img_path = dir_path + '/taskid{}/train_totalloss.png'.format(i)

        if os.path.isfile(diagnosis_img_path) and initial_frame == False:
            frame = cv2.imread(dir_path + '/taskid0/train_totalloss.png')
            height, width, layers = frame.shape
            video = cv2.VideoWriter(dir_path + '/diagnosis_video.avi', 0, 1, (width, height))
            initial_frame = True
        elif os.path.isfile(diagnosis_img_path) and initial_frame == True:
            img = cv2.imread(diagnosis_img_path)
            if 'pareto' in method:
                cv2.putText(img, 'MC:{}, pref_idx: {}'.format(hyperparameter_experiments[i]['MC'],
                            hyperparameter_experiments[i]['npref_prefidx']['pref_idx']), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 2)
            else:
                cv2.putText(img, 'MC:{}, weight_idx: {}'.format(hyperparameter_experiments[i]['MC'],
                            hyperparameter_experiments[i]['nweight_weight']['n_weight']), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            video.write(img)
        else:
            continue

    if initial_frame == True:
        cv2.destroyAllWindows()
        video.release()

def main( ):

    parser = argparse.ArgumentParser(description='resultsummary')

    parser.add_argument('--dataset', type=str, default='tabula_muris',
                        help='name of dataset')

    parser.add_argument('--confounder', type=str, default='batch',
                        help='name of confounder')

    parser.add_argument('--MCs', type=int, default=10,
                        help='number of Monte Carlos')

    parser.add_argument('--pareto_front_x', type=str, default='obj1',
                        help='xaxis value') #asw, ari, uca, nmi,obj1

    parser.add_argument('--pareto_front_y', type=str, default='obj2',
                        help='yaxis value') #obj2 (MINE or stdMMD), NN, be

    parser.add_argument('--methods_list', type=str, default='paretoMMD,regularizeMMD',
                        help='list of methods')

    parser.add_argument('--diagnosis', action='store_true', default=False,
                        help='whether to visualize diagnosis plot')

    parser.add_argument('--draw_extreme_points', action='store_true', default=False,
                        help='whether to draw all input points or not')

    parser.add_argument('--ParetoCandidates_ParetoPoints', type=str, default='ParetoCandidates',
                        help='choose to draw all Pareto Candidates or all Pareto points')

    parser.add_argument('--mu', type=float, default=0.05,
                        help='mu for NDC calculation')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=62364)
    args = parser.parse_args()

    args.methods_list = args.methods_list.split(',')

    for method in args.methods_list:
        dir_path = './result/{}/{}/pretrain150_lr0.001/{}'.format(args.dataset, args.confounder, method)
        if 'regularize' in method:
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'nweight_weight': [{'n_weight': n, 'weight': i} for n, i in zip(list(range(10)), [1/11, 2/11, 3/11, 4/11, 5/11, 6/11, 7/11, 8/11, 9/11, 10/11])]
            }
        elif 'pareto' in method:
            hyperparameter_config = {
                'MC': list(range(args.MCs)),
                'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10] * 10, list(range(10)))]
            }

        results_config_onemethod = load_result(dir_path, hyperparameter_config, method)
        if method == args.methods_list[0]:
            results_config_AllMethods = results_config_onemethod
        else:
            results_config_AllMethods = pd.concat([results_config_AllMethods, results_config_onemethod], axis=0)

        if args.diagnosis:
            diagnosis(dir_path, hyperparameter_config, method)

    dataframe_dict = {'results_config_AllMethods': results_config_AllMethods}
    if args.draw_extreme_points:
        dir_path = './result/{}/{}/pretrain150_lr0.001'.format(args.dataset, args.confounder)
        if any('MINE' in s for s in args.methods_list):
            results_config_ExtremePointsMINE = load_result_ExtremePoints('MINE', 10, dir_path, args.methods_list)
            dataframe_dict.update({'results_config_ExtremePointsMINE': results_config_ExtremePointsMINE})

        if any('MMD' in s for s in args.methods_list):
            results_config_ExtremePointsMMD = load_result_ExtremePoints('MMD', 10, dir_path, args.methods_list)
            dataframe_dict.update({'results_config_ExtremePointsMMD': results_config_ExtremePointsMMD})

    if args.ParetoCandidates_ParetoPoints == 'ParetoPoints':
        ReferencePoints, GridValues = Reference_GridValues(dataframe_dict, args.methods_list, args.pareto_front_x, args.pareto_front_y, args.draw_extreme_points, args.mu)

    dir_path = './result/{}/{}/pretrain150_lr0.001/'.format(args.dataset, args.confounder)

    for MC in range(args.MCs):
        if args.ParetoCandidates_ParetoPoints == 'ParetoCandidates':
            ParetoCandidates_AllMethods, ParetoCandidatesIndices_AllMethods = CollectPoints_AllMethods(
                dataframe_dict, MC, args.methods_list, args.pareto_front_x, args.pareto_front_y, args.draw_extreme_points, args.ParetoCandidates_ParetoPoints,
                ReferencePoints=None, GridValues=None)
            draw_scatter_plot(ParetoCandidates_AllMethods, ParetoCandidatesIndices_AllMethods, args.methods_list, args.pareto_front_x, args.pareto_front_y,
                              MC, dir_path, args.ParetoCandidates_ParetoPoints)

        if args.ParetoCandidates_ParetoPoints == 'ParetoPoints':
            ParetoPoints_AllMethods, ParetoPointsIndices_AllMethods, percentage_AllMethods, hypervolume_AllMethods, NDC_AllMethods = CollectPoints_AllMethods(
                dataframe_dict, MC, args.methods_list, args.pareto_front_x, args.pareto_front_y, args.draw_extreme_points, args.ParetoCandidates_ParetoPoints,
                ReferencePoints, GridValues)
            draw_scatter_plot(ParetoPoints_AllMethods, ParetoPointsIndices_AllMethods, args.methods_list, args.pareto_front_x, args.pareto_front_y,
                              MC, dir_path, args.ParetoCandidates_ParetoPoints)

            percentage_dataframe_oneMC = pd.DataFrame.from_dict(percentage_AllMethods)
            hypervolume_dataframe_oneMC = pd.DataFrame.from_dict(hypervolume_AllMethods)
            NDC_dataframe_oneMC = pd.DataFrame.from_dict(NDC_AllMethods)

            if MC == 0:
                percentage_dataframe = percentage_dataframe_oneMC
                hypervolume_dataframe = hypervolume_dataframe_oneMC
                NDC_dataframe = NDC_dataframe_oneMC
            else:
                percentage_dataframe = pd.concat([percentage_dataframe, percentage_dataframe_oneMC], axis=0)
                hypervolume_dataframe = pd.concat([hypervolume_dataframe, hypervolume_dataframe_oneMC],axis=0)
                NDC_dataframe = pd.concat([NDC_dataframe, NDC_dataframe_oneMC], axis=0)

    if args.ParetoCandidates_ParetoPoints == 'ParetoPoints':
        for metric in ['percentage', 'hypervolume', 'NDC']:
            if metric == 'percentage':
                metric_dataframe = percentage_dataframe
            elif metric == 'hypervolume':
                metric_dataframe = hypervolume_dataframe
            else:
                metric_dataframe = NDC_dataframe

            metric_dataframe_mean = metric_dataframe.mean(axis=0).reset_index(name='mean')
            metric_dataframe_std = metric_dataframe.std(axis=0).reset_index(name='std')
            metric_dataframe_mean_std = metric_dataframe_mean.merge(metric_dataframe_std, how='inner', on='index').round(2)

            metric_dataframe_mean_std_sorted = metric_dataframe_mean_std.iloc[[0,2,1,3],:]
            metric_dataframe_mean_std_sorted=metric_dataframe_mean_std_sorted.reset_index(drop=True)
            print('{}: {}'.format(args.dataset, metric))
            print('{} versus {}'.format(args.pareto_front_x, args.pareto_front_y))
            print(metric_dataframe_mean_std_sorted)

            metric_dataframe_mean_std_sorted['range_start'] = metric_dataframe_mean_std_sorted['mean'] - metric_dataframe_mean_std_sorted['std']
            metric_dataframe_mean_std_sorted['range_end'] = metric_dataframe_mean_std_sorted['mean'] + metric_dataframe_mean_std_sorted['std']
            if metric_dataframe_mean_std_sorted.loc[0,'range_start']<=metric_dataframe_mean_std_sorted.loc[1,'range_end'] and metric_dataframe_mean_std_sorted.loc[1,'range_start']<=metric_dataframe_mean_std_sorted.loc[0,'range_end']:
                print('train, not significant')
            else:
                print('train, significant')

            if metric_dataframe_mean_std_sorted.loc[2, 'range_start'] <= metric_dataframe_mean_std_sorted.loc[
                3, 'range_end'] and metric_dataframe_mean_std_sorted.loc[3, 'range_start'] <= \
                metric_dataframe_mean_std_sorted.loc[2, 'range_end']:
                print('test, not significant')
            else:
                print('test, significant')


# Run the actual program
if __name__ == "__main__":
    main()
