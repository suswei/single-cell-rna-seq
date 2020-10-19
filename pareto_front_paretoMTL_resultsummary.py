import os
import itertools
import numpy as np
import pandas as pd
import pickle
from pygmo import hypervolume
import statistics
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            return sum([row[x+1] <= candidateRow[x+1] for x in range(len(row)-1)]) == len(row)-1
        elif min_max == 'max':
            return sum([row[x + 1] >= candidateRow[x + 1] for x in range(len(row) - 1)]) == len(row) - 1
    else:
        if min_max == 'min':
            return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)
        elif min_max == 'max':
            return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

def draw_pareto_front(dataframe, MC, methods_list, pareto_front_type, save_path):

    if pareto_front_type == 'NN':
        xtitle = 'negative ELBO'
        ytitle = 'NN'
    else:
        xtitle = pareto_front_type.upper()
        ytitle = 'BE'

    dataframe_oneMC = dataframe[dataframe.MC.eq(MC)]
    percent_list_train, percent_list_test = [], []

    for type in ['train','test']:
        obj1_all, obj2_all = [], []
        fig = make_subplots(rows=math.ceil(len(methods_list)/2), cols=2, subplot_titles=tuple(methods_list), horizontal_spacing=0.02, vertical_spacing=0.06,
                            x_title=xtitle, y_title=ytitle, shared_yaxes=True, shared_xaxes=True)
        for (i, balance_advestimator) in enumerate(methods_list):
            dataframe_adv = dataframe_oneMC[dataframe_oneMC.balance_advestimator.eq(balance_advestimator)]
            pref_idx_list = [k + 1 for k in dataframe_adv.loc[:, 'pref_idx'].values.tolist()]

            if pareto_front_type in ['obj2', 'NN']:
                obj1 = dataframe_adv.loc[:, 'obj1_{}'.format(type)].values.tolist()
                obj2 = dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
            else:
                obj1 = dataframe_adv.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
                obj2 = dataframe_adv.loc[:, 'be_{}'.format(type)].values.tolist()

            obj1_all += obj1
            obj2_all += obj2

            inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
            if pareto_front_type in ['obj2','NN']:
                paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')
            else:
                paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'max')
            pp = np.array(list(paretoPoints1))
            dp = np.array(list(dominatedPoints1))
            if type == 'train':
                percent_list_train += [pp.shape[0] / len(obj1)]
            else:
                percent_list_test += [pp.shape[0] / len(obj1)]

            if i < len(methods_list)-1:
                showlegend=False
            else:
                showlegend=True
            fig.add_trace(go.Scatter(x=obj1, y=obj2, text=['{}'.format(i) for i in pref_idx_list],
                                     mode='markers+text',marker_size=15, marker_color='rgba(0, 0, 255, .9)', showlegend=False), math.ceil((i+1)/2), i - math.floor(i/2)*2 + 1)
            if dp.shape[0]>0:
                fig.add_trace(go.Scatter(x = dp[:,0].tolist(), y = dp[:,1].tolist(),
                                     mode='markers',marker_size=15, name='dominated',marker_color='rgba(0, 0, 255, .9)', showlegend=showlegend), math.ceil((i+1)/2), i - math.floor(i/2)*2 + 1)
            if pp.shape[0]>0:
                fig.add_trace(go.Scatter(x = pp[:,0].tolist(), y = pp[:,1].tolist(),
                                     mode='markers',marker_size=15, name='non-dominated', marker_color='rgba(255, 25, 52, .9)',showlegend=showlegend), math.ceil((i+1)/2), i - math.floor(i/2)*2 + 1)
        fig.update_traces(textposition='top center')

        fig.update_layout(
            width=900,
            height=400*math.ceil(len(methods_list)/2),
            margin=dict(
                l=80,
                r=100,
                b=80,
                t=35,
                pad=1
            ),
            font=dict(size=25, color='black', family='Times New Roman'),
            legend=dict(
                font=dict(
                    family='Times New Roman',
                    size=30,
                    color="black"
                )
            ),
        )

        # change font size for subplot title
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=30)

        if pareto_front_type in ['obj2','NN']:
            fig.update_xaxes(title_font=dict(size=30, family='Times New Roman', color='black'),range=[min(obj1_all)-20,max(obj1_all)+20],autorange=False)
        else:
            fig.update_xaxes(title_font=dict(size=30, family='Times New Roman', color='black'),range=[min(obj1_all) - 0.05, max(obj1_all) + 0.05], autorange=False)

        if pareto_front_type == 'NN':
            fig.update_yaxes(title_font=dict(size=30, family='Times New Roman', color='black'),range=[min(obj2_all) - 0.05, max(obj2_all) + 0.05], autorange=False)
        else:
            fig.update_yaxes(title_font=dict(size=30, family='Times New Roman', color='black'), range=[min(obj2_all) - 0.05, max(obj2_all) + 0.05], autorange=False)

        fig.write_image(save_path + '/paretofront_{}_{}_MC{}.png'.format(pareto_front_type,type,MC))

    return percent_list_train, percent_list_test

def draw_mean_metrics(dataframe_list, methods_list, y_axis_variable, save_path):

    if len(methods_list)<6:
        colors_list = ['rgba(0, 0, 255, .9)','rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)', 'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of colors')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('train', 'test'))
    for (i, type) in enumerate(['train', 'test']):
        dataframe = dataframe_list[i]
        if i==0:
            showlegend=False
        else:
            showlegend=True

        for (j,balance_advestimator) in enumerate(methods_list):
            fig.add_trace(go.Scatter(x=[k+1 for k in dataframe.loc[:, 'pref_idx'].values.tolist()],
                                 y=dataframe.loc[:, '{}_{}_mean_{}'.format(y_axis_variable,type, balance_advestimator)].values.tolist(),
                                 error_y=dict(type='data', array=dataframe.loc[:, '{}_{}_std_{}'.format(y_axis_variable,type, balance_advestimator)].values.tolist()),
                                 mode='lines+markers', name= '{}'.format(balance_advestimator), connectgaps=True, marker_color=colors_list[j], showlegend=showlegend),1,i+1)
    fig.update_layout(
        width=1600,
        height=800,
        margin=dict(
            l=30,
            r=30,
            b=30,
            t=30,
            pad=1
        ),
        font=dict(size=25, color='black', family='Times New Roman'),
        legend=dict(
            font=dict(
                family="Times New Roman",
                size=25,
                color="black"
            )
        )
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=25)

    fig.update_xaxes(title_text='subproblem index', title_font=dict(size=25, family='Times New Roman', color='black'), tick0=0, dtick=1)
    fig.update_yaxes(title_text='{} mean'.format(y_axis_variable.upper), title_font=dict(size=25, family='Times New Roman', color='black'))
    fig.write_image(save_path + '/{}_mean.png'.format(y_axis_variable))

def draw_barplot(data_array1, data_array2, methods_list, y_axis_variable, fig_title, save_path):

    if len(methods_list)<6:
        colors_list = ['rgba(255, 25, 52, .9)', 'rgba(48, 254, 0, .9)', 'rgba(182, 0, 228, .9)','rgb(158,202,225)', 'rgba(255, 149, 25, .9)']
    else:
        print('errors: the number of methods exceeds the number of colors')

    fig = make_subplots(rows=1, cols=2, subplot_titles=('train', 'test'), shared_yaxes=True, horizontal_spacing=0.02)

    for (i,data_array_dict) in enumerate([{'train':[data_array1]},{'test':[data_array2]}]):
        type = ['train','test'][i]
        data_array = data_array_dict[type][0]
        y_list, stdev_list = [], []
        for j in range(data_array.shape[0]):
            y_list += [statistics.mean(data_array[j, :].tolist())]
            stdev_list += [statistics.stdev(data_array[j, :].tolist())]
        fig.add_trace(go.Bar(x=methods_list, y=y_list,
                             error_y=dict(type='data', array=stdev_list), name=type, width=[1]*len(methods_list),
                             marker_color=colors_list[0:len(methods_list)],opacity=0.8,showlegend=False,
                             ),row=1,col=i+1
                      )
    '''
    title = {'text': fig_title,
             'y': 1,
             'x': 0.5,
             'xanchor': 'center',
             'yanchor': 'top',
             'font': dict(
                 family="Times New Roman",
                 size=30,
             )
        }
    '''
    fig.update_layout(barmode='group')
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=35,
            r=1,
            b=35,
            t=35,
            pad=1
        ),
        font=dict(size=22, color='black', family='Times New Roman'),
        bargap=0
    )

    # change font size for subplot title
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=30)

    fig.update_yaxes(title_text=y_axis_variable, title_font=dict(size=28, family='Times New Roman', color='black'),row=1,col=1)
    fig.write_image(save_path)

def pareto_front(hyperparameter_config, dataframe, methods_list, dir_path):

    paretoPoints_percent_array_train = np.empty((len(methods_list), 0))
    paretoPoints_percent_array_test = np.empty((len(methods_list), 0))

    for pareto_front_type in ['NN', 'asw', 'nmi', 'ari', 'uca']:
        for MC in range(dataframe.MC.max()+1):
            percent_list_train, percent_list_test = draw_pareto_front(dataframe, MC, methods_list, pareto_front_type, dir_path)

            paretoPoints_percent_array_train = np.concatenate((paretoPoints_percent_array_train,np.array([percent_list_train]).reshape((len(methods_list),1))),axis=1)
            paretoPoints_percent_array_test = np.concatenate((paretoPoints_percent_array_test, np.array([percent_list_test]).reshape((len(methods_list),1))), axis=1)

        draw_barplot(data_array1=paretoPoints_percent_array_train, data_array2=paretoPoints_percent_array_test, methods_list=methods_list, y_axis_variable='ParetoPoints percent mean',
                 fig_title='Average percent of non-dominated points', save_path=dir_path + '/paretopoints_percent_{}.png'.format(pareto_front_type))

def mean_metrics(hyperparameter_config, dataframe, methods_list, dir_path):

    dataframe['row_idx'] = list(range(dataframe.shape[0]))

    dataframe_train_test = []
    for type in ['train', 'test']:
        for balance_advestimator in methods_list:
            dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

            #for each MC, only keep the ParetoPoints
            for MC in range(dataframe_adv.MC.max()+1):
                dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                row_index = dataframe_oneMC.row_idx.values.tolist()
                obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                obj2 = dataframe_oneMC.loc[:, 'NN_{}'.format(type)].values.tolist()
                inputPoints1 = [[row_index[k], obj1[k], obj2[k]] for k in range(len(obj1))]  # index needed
                paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, True, 'min')
                pp = np.array(list(paretoPoints1))
                dataframe_adv_ParetoPoints_oneMC = dataframe_oneMC[dataframe_oneMC['row_idx'].isin([int(k) for k in pp[:,0].tolist()])]
                if MC==0:
                    dataframe_adv_ParetoPoints = dataframe_adv_ParetoPoints_oneMC
                else:
                    dataframe_adv_ParetoPoints = pd.concat([dataframe_adv_ParetoPoints,dataframe_adv_ParetoPoints_oneMC],axis=0)

            if type=='train':
                dataframe_mean_std = dataframe_adv_ParetoPoints.groupby('pref_idx').agg(
                    obj1_train_mean=('obj1_train', 'mean'),
                    obj1_train_std=('obj1_train', 'std'),
                    obj2_train_mean=('obj2_train', 'mean'),
                    obj2_train_std=('obj2_train', 'std'),
                    NN_train_mean=('NN_train','mean'),
                    NN_train_std=('NN_train','std'),
                    asw_train_mean=('asw_train', 'mean'),
                    asw_train_std=('asw_train', 'std'),
                    nmi_train_mean=('nmi_train', 'mean'),
                    nmi_train_std=('nmi_train', 'std'),
                    ari_train_mean=('ari_train', 'mean'),
                    ari_train_std=('ari_train', 'std'),
                    uca_train_mean=('uca_train', 'mean'),
                    uca_train_std=('uca_train', 'std'),
                    be_train_mean=('be_train', 'mean'),
                    be_train_std=('be_train', 'std')
                ).reset_index()
            else:
                dataframe_mean_std = dataframe_adv_ParetoPoints.groupby('pref_idx').agg(
                    obj1_test_mean=('obj1_test', 'mean'),
                    obj1_test_std=('obj1_test', 'std'),
                    obj2_test_mean=('obj2_test', 'mean'),
                    obj2_test_std=('obj2_test', 'std'),
                    NN_test_mean=('NN_test', 'mean'),
                    NN_test_std=('NN_test', 'std'),
                    asw_test_mean=('asw_test', 'mean'),
                    asw_test_std=('asw_test', 'std'),
                    nmi_test_mean=('nmi_test', 'mean'),
                    nmi_test_std=('nmi_test', 'std'),
                    ari_test_mean=('ari_test', 'mean'),
                    ari_test_std=('ari_test', 'std'),
                    uca_test_mean=('uca_test', 'mean'),
                    uca_test_std=('uca_test', 'std'),
                    be_test_mean=('be_test', 'mean'),
                    be_test_std=('be_test', 'std')
                ).reset_index()

            dataframe_mean_std.rename(columns=dict(zip(dataframe_mean_std.columns[1:], [column + '_{}'.format(balance_advestimator) for column in dataframe_mean_std.columns[1:]])),
                        inplace=True)

            if balance_advestimator == methods_list[0]:
                dataframe_mean_std_total = dataframe_mean_std
            else:
                dataframe_mean_std_total = dataframe_mean_std_total.merge(dataframe_mean_std, how='outer', on='pref_idx')

        dataframe_train_test += [dataframe_mean_std_total]

    for y_axis_variable in ['obj1', 'NN', 'asw','nmi','ari','uca','be']:
        draw_mean_metrics(dataframe_train_test, methods_list, y_axis_variable, dir_path)

def hypervolume_compare(hyperparameter_config, dataframe, methods_list, dir_path):

    #get the reference point to calculate hypervolume, hypervolume function seems only work for minimization problem
    for pareto_front_type in ['NN','asw','nmi','ari','uca']:
        if pareto_front_type == 'NN':
            obj1_max = dataframe.loc[:,['obj1_train','obj1_test']].max(axis=0).max()
            obj2_max = dataframe.loc[:,['NN_train','NN_test']].max(axis=0).max()
            ref_point = [obj1_max + 5, obj2_max+0.01]
        else:
            obj1_max = dataframe.loc[:, ['{}_train'.format(pareto_front_type), '{}_test'.format(pareto_front_type)]].min(axis=0).min()*(-1)
            obj2_max = dataframe.loc[:, ['be_train', 'be_test']].min(axis=0).min() * (-1)
            ref_point = [obj1_max + 0.01, obj2_max + 0.01]

        hypervolume_array_train = np.empty((0, dataframe.MC.max() + 1))
        hypervolume_array_test = np.empty((0, dataframe.MC.max() + 1))
        for type in ['train','test']:
            for balance_advestimator in methods_list:
                hypervolume_list = []
                dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

                for MC in range(dataframe_adv.MC.max() + 1):
                    dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                    if pareto_front_type == 'NN':
                        obj1_list = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                        obj2_list = dataframe_oneMC.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
                    else:
                        obj1_list = [k*(-1) for k in dataframe_oneMC.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()]
                        obj2_list = [k * (-1) for k in dataframe_oneMC.loc[:, 'be_{}'.format(type)].values.tolist()]

                    #when calculate the hypervolume, get rid of the dominated points first, then calculate the hypervolume
                    inputPoints1 = [[obj1_list[k], obj2_list[k]] for k in range(len(obj1_list))]
                    if pareto_front_type == 'NN':
                        paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')
                    else:
                        paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, False, 'min')
                    pp = np.array(list(paretoPoints1))
                    hv = hypervolume([[pp[:,0][k], pp[:,1][k]] for k in range(pp.shape[0])])
                    hypervolume_list += [hv.compute(ref_point)]
                if type == 'train':
                    hypervolume_array_train = np.concatenate((hypervolume_array_train, np.array([hypervolume_list])),axis=0)
                else:
                    hypervolume_array_test = np.concatenate((hypervolume_array_test, np.array([hypervolume_list])),axis=0)
            save_path = dir_path + '/hypervolume_compare_{}.png'.format(pareto_front_type)
            draw_barplot(data_array1=hypervolume_array_train, data_array2=hypervolume_array_test,  methods_list=methods_list, y_axis_variable='hypervolume', fig_title='Averaged hypervolume', save_path=save_path)

def paretoMTL_summary(dataset: str='muris_tabula', confounder: str='batch', methods_list: list=['MINE','MMD']):

    dir_path = './result/pareto_front_paretoMTL/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'pre_epochs': [200],
        'pre_lr': [1e-3],
        'adv_lr': [5e-5],
        'n_epochs': [150],
        'lr': [1e-3],
        'MC': list(range(20)),
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for balance_advestimator in methods_list:
        dir_path_subset = dir_path + '/{}'.format(balance_advestimator)
        for i in range(len(hyperparameter_experiments)):
            config_path = dir_path_subset + '/taskid{}/config.pkl'.format(i)
            results_path = dir_path_subset + '/taskid{}/results.pkl'.format(i)

            if os.path.isfile(config_path) and os.path.isfile(results_path):
                config = pickle.load(open(config_path, "rb"))
                results = pickle.load(open(results_path, "rb"))

                if 'obj1_minibatch_list' in results.keys():
                    del results['obj1_minibatch_list']
                if 'obj2_minibatch_list' in results.keys():
                    del results['obj2_minibatch_list']

                results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator','MC', 'pre_epochs', 'pre_lr', 'adv_lr', 'n_epochs', 'lr', 'pref_idx'])}
                results_config.update({'balance_advestimator': [balance_advestimator]})
                results_config.update(results)

                if balance_advestimator == methods_list[0] and i == 0:
                    results_config_total = pd.DataFrame.from_dict(results_config)
                else:
                    results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)

    del hyperparameter_config['npref_prefidx']
    del hyperparameter_config['MC']

    pareto_front(hyperparameter_config= hyperparameter_config, dataframe=results_config_total, methods_list=methods_list, dir_path=dir_path)

    mean_metrics(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, methods_list=methods_list, dir_path=dir_path)

    hypervolume_compare(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, methods_list=methods_list, dir_path=dir_path)
