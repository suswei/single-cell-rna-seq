import os
import itertools
import numpy as np
import pandas as pd
import pickle
from pygmo import hypervolume
import statistics
import plotly.graph_objects as go

def simple_cull(inputPoints, dominates, return_index: bool=False):
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
            if dominates(candidateRow, row, return_index):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, return_index):
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

def dominates(row, candidateRow, return_index):
    if return_index == True:
        return sum([row[x+1] <= candidateRow[x+1] for x in range(len(row)-1)]) == len(row)-1
    else:
        return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)

def draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path):

    inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]
    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates)
    pp = np.array(list(paretoPoints1))
    dp = np.array(list(dominatedPoints1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obj1, y=obj2,
                             text=['{}'.format(i) for i in pref_idx_list],
                             mode='markers+text', showlegend=False),1,1)
    if dp.shape[0]>0:
        fig.add_trace(go.Scatter(x = dp[:,0].tolist(), y = dp[:,1].tolist(),
                             mode='markers', name='dominated points',marker_color='rgba(0, 0, 255, .9)', showlegend=False),1,1)
    if pp.shape[0]>0:
        fig.add_trace(go.Scatter(x = pp[:,0].tolist(),
                             y = pp[:,1].tolist(),
                             mode='markers', name='non-dominated points', marker_color='rgba(255, 25, 52, .9)',showlegend=False),1,1)

    fig.update_traces(textposition='top center')

    fig.update_layout(
        width=700,
        height=700,
        margin=dict(
            l=25,
            r=25,
            b=25,
            t=110,
            pad=1
        ),
        font=dict(size=20, color='black', family='Times New Roman'),
        title={'text': fig_title,
               'y': 0.94,
               'x': 0.38,
               'xanchor': 'center',
               'yanchor': 'top',
               'font': dict(
                           family="Times New Roman",
                           size=20,
                          )
        },
        legend = dict(
            font=dict(
                family='Times New Roman',
                size=20,
                color="black"
            )
        )
    )
    fig.update_xaxes(title_text='obj1', title_font=dict(size=20, family='Times New Roman', color='black'))
    fig.update_yaxes(title_text='obj2', title_font=dict(size=20, family='Times New Roman', color='black'))
    fig.write_image(save_path)
    return pp.shape[0]/len(obj1)

def draw_mean_metrics(dataframe, methods_list, y_axis_variable, type, save_path):
    fig = go.Figure()

    for balance_advestimator in methods_list:
        fig.add_trace(go.Scatter(x=[k+1 for k in dataframe.loc[:, 'pref_idx'].values.tolist()],
                             y=dataframe.loc[:, '{}_{}_mean_{}'.format(y_axis_variable,type, balance_advestimator)].values.tolist(),
                             error_y=dict(type='data', array=dataframe.loc[:, '{}_{}_std_{}'.format(y_axis_variable,type, balance_advestimator)].values.tolist()),
                             mode='lines+markers', name= '{}'.format(balance_advestimator), connectgaps=True)
                      )
    fig.update_layout(
        width=1000,
        height=800,
        margin=dict(
            l=25,
            r=25,
            b=25,
            t=25,
            pad=1
        ),
        font=dict(size=20, color='black', family='Times New Roman'),
        title={'text': type,
               'y': 1,
               'x': 0.45,
               'xanchor': 'center',
               'yanchor': 'top',
               'font': dict(
                   family="Times New Roman",
                   size=25,
               )
        },
        legend=dict(
            font=dict(
                family="Times New Roman",
                size=20,
                color="black"
            )
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    fig.update_xaxes(title_text='subproblem index', title_font=dict(size=20, family='Times New Roman', color='black'))
    fig.update_yaxes(title_text='{} mean'.format(y_axis_variable), title_font=dict(size=20, family='Times New Roman', color='black'))
    fig.write_image(save_path + '/{}_mean_{}.png'.format(y_axis_variable,type))

def draw_barplot(data_array, methods_list, y_axis_variable, fig_title, save_path):

    y_list, stdev_list = [], []
    for i in range(data_array.shape[0]):
        y_list += [statistics.mean(data_array[i,:].tolist())]
        stdev_list += [statistics.stdev(data_array[i, :].tolist())]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=methods_list, y=y_list,
                         error_y=dict(type='data', array=stdev_list)
                         )
                  ) #text=[statistics.mean(MINE_list), statistics.mean(HSIC_list)]
    #fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
    fig.update_yaxes()
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(
            l=20,
            r=1,
            b=25,
            t=25,
            pad=1
        ),
        font=dict(size=20, color='black', family='Times New Roman'),
        title={'text': fig_title,
               'y': 1,
               'x': 0.52,
               'xanchor': 'center',
               'yanchor': 'top',
               'font': dict(
                   family="Times New Roman",
                   size=25,
               )
        }
    )
    fig.update_yaxes(title_text=y_axis_variable, title_font=dict(size=20, family='Times New Roman', color='black'))
    fig.write_image(save_path)

def pareto_front(hyperparameter_config, dataframe, methods_list, dir_path):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for temp in hyperparameter_experiments:
        for type in ['train', 'test']:
            paretoPoints_percent_array = np.empty((0, dataframe.MC.max()+1))
            for balance_advestimator in methods_list:
                dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]
                paretoPoints_percent_one = []
                for pareto_front_type in ['whole', 'NN']:  # 'asw', 'nmi', 'ari', 'uca'
                    for MC in range(dataframe_adv.MC.max()+1):
                        dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                        pref_idx_list = [k+1 for k in dataframe_oneMC.loc[:, 'pref_idx'].values.tolist()]

                        save_path = dir_path + '/{}'.format(balance_advestimator) + '/paretoMTL_MC{}_{}_{}_{}.png'.format(MC, balance_advestimator.split('_')[1], pareto_front_type, type)

                        if pareto_front_type == 'whole':
                            fig_title = '{}, obj1: negative_ELBO, obj2: {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                                type, balance_advestimator.split('_')[1], temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)
                            obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneMC.loc[:, 'obj2_{}'.format(type)].values.tolist()
                            percent = draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                            paretoPoints_percent_one += [percent]
                        elif pareto_front_type in ['NN']:
                            fig_title = '{}, obj1: negative_ELBO, obj2: NN(for {}),<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                                type, balance_advestimator.split('_')[1], temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)
                            obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneMC.loc[:, '{}_{}'.format(pareto_front_type,type)].values.tolist()
                            _ = draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                    if pareto_front_type == 'whole':
                        paretoPoints_percent_array = np.concatenate((paretoPoints_percent_array,np.array([paretoPoints_percent_one])),axis=0)

            draw_barplot(data_array=paretoPoints_percent_array, methods_list=methods_list, y_axis_variable='ParetoPoints percent',
                         fig_title=type, save_path=dir_path + '/ParetoPoints_percent_{}.png'.format(type))


def mean_metrics(hyperparameter_config, dataframe, methods_list, dir_path):

    dataframe['row_idx'] = list(range(dataframe.shape[0]))

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for temp in hyperparameter_experiments:
        for type in ['train', 'test']:
            for balance_advestimator in methods_list:
                dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

                #for each MC, only keep the ParetoPoints
                for MC in range(dataframe_adv.MC.max()+1):
                    dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                    row_index = dataframe_oneMC.row_idx.values.tolist()
                    obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                    obj2 = dataframe_oneMC.loc[:, 'obj2_{}'.format(type)].values.tolist()
                    inputPoints1 = [[row_index[k], obj1[k], obj2[k]] for k in range(len(obj1))]  # index needed
                    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates, return_index=True)
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

            for y_axis_variable in ['obj1', 'NN', 'asw','nmi','ari','uca','be']:
                draw_mean_metrics(dataframe_mean_std_total, methods_list, y_axis_variable, type, dir_path)

def hypervolume_compare(hyperparameter_config, dataframe, methods_list, dir_path):

    #get the reference point to calculate hypervolume
    obj1_max = dataframe.loc[:,['obj1_train','obj1_test']].max(axis=0).max()
    NN_max = dataframe.loc[:,['NN_train','NN_test']].max(axis=0).max()
    ref_point = [obj1_max + 5, NN_max]

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for temp in hyperparameter_experiments:
        for type in ['train','test']:
            hypervolume_array = np.empty((0,dataframe.MC.max() + 1))

            for balance_advestimator in methods_list:
                hypervolume_list = []
                dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

                for MC in range(dataframe_adv.MC.max() + 1):
                    dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                    obj1_list = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                    NN_list = dataframe_oneMC.loc[:, 'NN_{}'.format(type)].values.tolist()

                    #when calculate the hypervolume, get rid of the dominated points first, then calculate the hypervolume
                    inputPoints1 = [[obj1_list[k], NN_list[k]] for k in range(len(obj1_list))]
                    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates)
                    pp = np.array(list(paretoPoints1))
                    hv = hypervolume([[pp[:,0][k], pp[:,1][k]] for k in range(pp.shape[0])])
                    hypervolume_list += [hv.compute(ref_point)]
                hypervolume_array = np.concatenate((hypervolume_array, np.array([hypervolume_list])),axis=0)
            save_path = dir_path + '/hypervolume_compare_{}.png'.format(type)
            draw_barplot(data_array=hypervolume_array, methods_list=methods_list, y_axis_variable='hypervolume', fig_title=type, save_path=save_path)

def paretoMTL_summary(dataset: str='muris_tabula', confounder: str='batch', methods_list: list=['std_MINE','std_MMD']):

    dir_path = './result/pareto_front_paretoMTL/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'pre_epochs': [200],
        'pre_lr': [1e-3],
        'adv_lr': [5e-5],
        'n_epochs': [100],
        'lr': [1e-3],
        'MC': list(range(10)),
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
