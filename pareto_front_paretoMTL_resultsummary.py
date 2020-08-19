import os
import itertools
import numpy as np
import pandas as pd
import pickle
from pygmo import hypervolume
import statistics
import plotly.graph_objects as go
import cv2

def draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = obj1,
                         y = obj2,
                         text = ['pref_idx:{}'.format(i) for i in pref_idx_list],
                         mode='markers+text')
                  )
    fig.update_traces(textposition='top center')

    fig.update_layout(
        width=700,
        height=700,
        margin=dict(
            l=20,
            r=1,
            b=20,
            t=110,
            pad=1
        ),
        font=dict(size=15, color='black', family='Arial, sans-serif'),
        title={'text': fig_title,
               'y': 0.97,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'}
    )
    fig.update_xaxes(title_text='obj1', title_font=dict(size=15, family='Arial, sans-serif', color='black'))
    fig.update_yaxes(title_text='obj2', title_font=dict(size=15, family='Arial, sans-serif', color='black'))
    fig.write_image(save_path)

def draw_mean_metrics(dataframe, y_axis_variable, save_path):
    fig = go.Figure()

    for balance_advestimator in ['gradnorm_MINE', 'std_MINE', 'std_HSIC']:
        fig.add_trace(go.Scatter(x=dataframe.loc[:, 'pref_idx'].values.tolist(),
                             y=dataframe.loc[:, '{}_mean_{}'.format(y_axis_variable, balance_advestimator)].values.tolist(),
                             error_y=dict(type='data', array=dataframe.loc[:, '{}_std_{}'.format(y_axis_variable, balance_advestimator)].values.tolist()),
                             mode='lines+markers', name= '{}'.format(balance_advestimator))
                      )
    fig.update_layout(
        width=1000,
        height=800,
        margin=dict(
            l=1,
            r=1,
            b=20,
            t=25,
            pad=1
        ),
        font=dict(size=15, color='black', family='Arial, sans-serif'),
        title={'text': '{}_mean'.format(y_axis_variable),
               'y': 1,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    fig.update_xaxes(title_text='preference index', title_font=dict(size=15, family='Arial, sans-serif', color='black'))
    fig.write_image(save_path + '/{}_mean.png'.format(y_axis_variable))

def draw_hypervolume(list1, list2, list3, fig_title, save_path):

    fig = go.Figure()
    fig.add_trace(go.Bar(x=['gradnorm_MINE','std_MINE','std_HSIC'],
                         y=[statistics.mean(list1), statistics.mean(list2), statistics.mean(list3)],
                         error_y=dict(type='data', array=[statistics.stdev(list1), statistics.stdev(list2), statistics.stdev(list3)])
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
            b=1,
            t=25,
            pad=1
        ),
        font=dict(size=15, color='black', family='Arial, sans-serif'),
        title={'text': fig_title,
               'y': 1,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'}
    )
    fig.update_yaxes(title_text='hypervolume', title_font=dict(size=15, family='Arial, sans-serif', color='black'))
    fig.write_image(save_path)

def create_video(video_save_path, image_paths):
    #image_paths is a list of image paths to read images into a video
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_save_path, 0, 1, (width, height))
    for file_path in image_paths:
        video.write(cv2.imread(file_path))
    cv2.destroyAllWindows()
    video.release()

def pareto_front(hyperparameter_config, dataframe, dir_path):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for balance_advestimator in ['gradnorm_MINE', 'std_MINE', 'std_HSIC']:
        dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

        for temp in hyperparameter_experiments:

            for MC in range(dataframe_adv.MC.max()+1):
                dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]

                pref_idx_list = dataframe_oneMC.loc[:, 'pref_idx'].values.tolist()
                for pareto_front_type in ['whole', 'NN']: #'asw', 'nmi', 'ari', 'uca'
                    for type in ['train', 'test']:

                        save_path = dir_path + '/{}'.format(balance_advestimator) + '/paretoMTL_MC{}_{}_{}_{}.png'.format(MC, balance_advestimator.split('_')[1], pareto_front_type, type)

                        if pareto_front_type == 'whole':
                            if balance_advestimator == 'gradnorm_MINE':
                                fig_title = 'obj1: negative_ELBO, obj2: {}, {}, {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {},<br>gradnorm_weight_lowlimit: {:.6E}'.format(
                                    balance_advestimator.split('_')[1], pareto_front_type, type, temp['pre_epochs'],
                                    temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC, temp['gradnorm_weight_lowlimit'])
                            else:
                                fig_title = 'obj1: negative_ELBO, obj2: {}, {}, {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                                    balance_advestimator.split('_')[1], pareto_front_type, type, temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)

                            obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneMC.loc[:, 'obj2_{}'.format(type)].values.tolist()
                            draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                        elif pareto_front_type in ['NN']:
                            if balance_advestimator == 'gradnorm_MINE':
                                fig_title = 'obj1: negative_ELBO, obj2: NN(for {}), {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {},<br>gradnorm_weight_lowlimit: {:.6E}'.format(
                                    balance_advestimator.split('_')[1], type, temp['pre_epochs'], temp['pre_lr'],
                                    temp['adv_lr'], temp['n_epochs'], temp['lr'], MC, temp['gradnorm_weight_lowlimit'])
                            else:
                                fig_title = 'obj1: negative_ELBO, obj2: NN(for {}), {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                                    balance_advestimator.split('_')[1], type, temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)

                            obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneMC.loc[:, '{}_{}'.format(pareto_front_type,type)].values.tolist()

                            draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                        '''
                        else:
                            fig_title = 'obj1: be, obj2: {}, {}, {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}'.format(
                                pareto_front_type, temp['adv_estimator'], type, temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'])
    
                            obj1 = dataframe_oneMC.loc[:, 'be_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneMC.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()
    
                            draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                        '''
    '''
    video_save_path = dir_path + '/paretoMTL.mp4'

    image_paths = []
    for temp in hyperparameter_experiments:
        for pareto_front_type in ['whole', 'NN', 'asw', 'nmi', 'ari', 'uca']:
            for type in ['train','test']:
                one_image_path = dir_path + '/paretoMTL_{}_{}_{}.png'.format(temp['adv_estimator'], pareto_front_type, type)
                if os.path.isfile(one_image_path):
                    image_paths += [one_image_path]

    create_video(video_save_path, image_paths)
    '''

def mean_metrics(hyperparameter_config, dataframe, dir_path):

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for balance_advestimator in ['gradnorm_MINE', 'std_MINE', 'std_HSIC']:
        for temp in hyperparameter_experiments:
            dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]
            dataframe_mean_std = dataframe_adv.groupby('pref_idx').agg(
                obj1_train_mean=('obj1_train', 'mean'),
                obj1_train_std=('obj1_train', 'std'),
                obj2_train_mean=('obj2_train', 'mean'),
                obj2_train_std=('obj2_train', 'std'),
                NN_train_mean=('NN_train','mean'),
                NN_train_std=('NN_train','std'),
                obj1_test_mean=('obj1_test', 'mean'),
                obj1_test_std=('obj1_test', 'std'),
                obj2_test_mean=('obj2_test', 'mean'),
                obj2_test_std=('obj2_test', 'std'),
                NN_test_mean=('NN_test', 'mean'),
                NN_test_std=('NN_test', 'std'),
                asw_train_mean=('asw_train', 'mean'),
                asw_train_std=('asw_train', 'std'),
                nmi_train_mean=('nmi_train', 'mean'),
                nmi_train_std=('nmi_train', 'std'),
                ari_train_mean=('ari_train', 'mean'),
                ari_train_std=('ari_train', 'std'),
                uca_train_mean=('uca_train', 'mean'),
                uca_train_std=('uca_train', 'std'),
                be_train_mean=('be_train', 'mean'),
                be_train_std=('be_train', 'std'),
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

        if balance_advestimator == 'gradnorm_MINE':
            dataframe_mean_std_total = dataframe_mean_std
        else:
            dataframe_mean_std_total = dataframe_mean_std_total.merge(dataframe_mean_std, on='pref_idx')

    for variable in ['obj1', 'NN', 'asw','nmi','ari','uca','be']:
        for type in ['train','test']:
            y_axis_variable = '{}_{}'.format(variable, type)
            draw_mean_metrics(dataframe_mean_std_total, y_axis_variable, dir_path)

def hypervolume_compare(hyperparameter_config, dataframe, dir_path):

    #get the reference point to calculate hypervolume
    obj1_max = dataframe.loc[:,['obj1_train','obj1_test']].max(axis=0).max()
    NN_max = dataframe.loc[:,['NN_train','NN_test']].max(axis=0).max()
    ref_point = [obj1_max + 5, NN_max]

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for type in ['train','test']:
        hypervolume_gradnorm_MINE, hypervolume_std_MINE, hypervolume_std_HSIC = [], [], []

        for balance_advestimator in ['gradnorm_MINE', 'std_MINE', 'std_HSIC']:
            for temp in hyperparameter_experiments:

                dataframe_adv = dataframe[dataframe.balance_advestimator.eq(balance_advestimator)]

                for MC in range(dataframe_adv.MC.max() + 1):
                    dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                    obj1_list = dataframe_oneMC.loc[:,'obj1_{}'.format(type)].values.tolist()
                    NN_list = dataframe_oneMC.loc[:, 'NN_{}'.format(type)].values.tolist()
                    hv = hypervolume([[obj1_list[k], NN_list[k]] for k in range(len(obj1_list))])
                    if balance_advestimator == 'gradnorm_MINE':
                        hypervolume_gradnorm_MINE += [hv.compute(ref_point)]
                    elif balance_advestimator == 'std_MINE':
                        hypervolume_std_MINE += [hv.compute(ref_point)]
                    else:
                        hypervolume_std_HSIC += [hv.compute(ref_point)]
        fig_title = type
        save_path = dir_path + '/hypervolume_compare_{}.png'.format(type)
        draw_hypervolume(list1=hypervolume_gradnorm_MINE, list2=hypervolume_std_MINE, list3= hypervolume_std_HSIC, fig_title=fig_title, save_path=save_path)

def paretoMTL_summary(dataset: str='muris_tabula', confounder: str='batch'):

    dir_path = './result/pareto_front_paretoMTL/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'pre_epochs': [250],
        'pre_lr': [1e-3],
        'adv_lr': [5e-5],
        'n_epochs': [100],
        'lr': [1e-3],
        'gradnorm_weight_lowlimit': [1e-6],
        'MC': list(range(10)),
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for balance_advestimator in ['gradnorm_MINE', 'std_MINE', 'std_HSIC']:
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

                if balance_advestimator == 'gradnorm_MINE' and i == 0:
                    results_config_total = pd.DataFrame.from_dict(results_config)
                else:
                    results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)

    del hyperparameter_config['npref_prefidx']
    del hyperparameter_config['MC']

    pareto_front(hyperparameter_config= hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)

    mean_metrics(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)

    hypervolume_compare(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)
