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
        title={'text': fig_title,
               'y': 0.92,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        xaxis_title="obj1",
        yaxis_title="obj2",
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
    fig.write_image(save_path)

def draw_mean_metrics(dataframe, y_axis_variable, adv_estimator, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.loc[:, 'pref_idx'].values.tolist(),
                         y=dataframe.loc[:, '{}_mean'.format(y_axis_variable)].values.tolist(),
                         error_y=dict(type='data', array=dataframe.loc[:, '{}_std'.format(y_axis_variable)].values.tolist()),
                         mode='lines+markers', name= '{}'.format(y_axis_variable))
                  )
    fig.update_xaxes(title_text='preference index')

    fig.update_layout(
        title={'text': 'adv_estimator: {}, {}_mean'.format(adv_estimator, y_axis_variable),
               'y': 0.9,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
    fig.write_image(save_path + '/{}_mean_{}.png'.format(y_axis_variable, adv_estimator))

def draw_hypervolume(MINE_list, HSIC_list, fig_title, save_path):

    fig = go.Figure()
    fig.add_trace(go.Bar(x=['MINE','HSIC'],
                         y=[statistics.mean(MINE_list), statistics.mean(HSIC_list)],
                         error_y=dict(type='data', array=[statistics.stdev(MINE_list), statistics.stdev(HSIC_list)])
                         )
                  ) #text=[statistics.mean(MINE_list), statistics.mean(HSIC_list)]
    #fig.update_traces(texttemplate='%{text:.0f}', textposition='inside')
    fig.update_layout(
        title={'text': fig_title,
               'y': 0.92,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
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

    for temp in hyperparameter_experiments:

        dataframe_adv = dataframe[dataframe.adv_estimator.eq(temp['adv_estimator'])]

        for MC in range(dataframe_adv.MC.max()+1):
            dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]

            pref_idx_list = dataframe_oneMC.loc[:, 'pref_idx'].values.tolist()
            for pareto_front_type in ['whole', 'NN']: #'asw', 'nmi', 'ari', 'uca'
                for type in ['train', 'test']:

                    save_path = dir_path + '/paretoMTL_MC{}_{}_{}_{}.png'.format(MC, temp['adv_estimator'], pareto_front_type, type)

                    if pareto_front_type == 'whole':

                        fig_title = 'obj1: negative_ELBO, obj2: {}, {}, {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                            temp['adv_estimator'], pareto_front_type, type, temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)

                        obj1 = dataframe_oneMC.loc[:, 'obj1_{}'.format(type)].values.tolist()
                        obj2 = dataframe_oneMC.loc[:, 'obj2_{}'.format(type)].values.tolist()
                        draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                    elif pareto_front_type in ['NN']:

                        fig_title = 'obj1: negative_ELBO, obj2: NN(for {}), {},<br>pre_epochs: {}, pre_lr: {}, adv_lr: {},<br>n_epochs:{}, lr: {}, MC: {}'.format(
                            temp['adv_estimator'], type, temp['pre_epochs'], temp['pre_lr'], temp['adv_lr'], temp['n_epochs'], temp['lr'], MC)

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

    for temp in hyperparameter_experiments:
        dataframe_adv = dataframe[dataframe.adv_estimator.eq(temp['adv_estimator'])]
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

        for variable in ['obj1', 'obj2', 'NN', 'asw','nmi','ari','uca','be']:
            for type in ['train','test']:
                y_axis_variable = '{}_{}'.format(variable, type)
                draw_mean_metrics(dataframe_mean_std, y_axis_variable, temp['adv_estimator'], dir_path)

def hypervolume_compare(hyperparameter_config, dataframe, dir_path):

    #get the reference point to calculate hypervolume
    obj1_max = dataframe.loc[:,['obj1_train','obj1_test']].max(axis=0).max()
    NN_max = dataframe.loc[:,['NN_train','NN_test']].max(axis=0).max()
    ref_point = [obj1_max + 5, NN_max]

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for type in ['train','test']:
        hypervolume_MINE, hypervolume_HSIC = [], []
        for temp in hyperparameter_experiments:

            dataframe_adv = dataframe[dataframe.adv_estimator.eq(temp['adv_estimator'])]

            for MC in range(dataframe_adv.MC.max() + 1):
                dataframe_oneMC = dataframe_adv[dataframe_adv.MC.eq(MC)]
                obj1_list = dataframe_oneMC.loc[:,'obj1_{}'.format(type)].values.tolist()
                NN_list = dataframe_oneMC.loc[:, 'NN_{}'.format(type)].values.tolist()
                hv = hypervolume([[obj1_list[k], NN_list[k]] for k in range(len(obj1_list))])
                if temp['adv_estimator'] == 'MINE':
                    hypervolume_MINE += [hv.compute(ref_point)]
                elif temp['adv_estimator'] == 'HSIC':
                    hypervolume_HSIC += [hv.compute(ref_point)]
        fig_title = 'compare hypervolume of pareto front from MINE and HSIC'
        save_path = dir_path + '/hypervolume_compare_{}.png'.format(type)
        draw_hypervolume(MINE_list=hypervolume_MINE, HSIC_list=hypervolume_HSIC, fig_title=fig_title, save_path=save_path)

def paretoMTL_summary(dataset: str='muris_tabula', confounder: str='batch'):

    dir_path = './result/pareto_front_paretoMTL/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'adv_estimator': ['MINE'],
        'pre_epochs': [200],
        'pre_lr': [1e-3],
        'adv_lr': [5e-5],
        'n_epochs': [50],
        'lr': [1e-3],
        'MCs': 1 * [1],
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10]*10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(hyperparameter_experiments)):
        if i in list(range(200)):
            config_path = dir_path + '/taskid{}/config.pkl'.format(i)
            results_path = dir_path + '/taskid{}/results.pkl'.format(i)
        else:
            config_path = dir_path + '/HSIC/taskid{}/config.pkl'.format(i - int(i/200)*200)
            results_path = dir_path + '/HSIC/taskid{}/results.pkl'.format(i - int(i/200)*200)
        if os.path.isfile(config_path) and os.path.isfile(results_path):
            config = pickle.load(open(config_path, "rb"))
            results = pickle.load(open(results_path, "rb"))

            if 'obj1_minibatch_list' in results.keys():
                del results['obj1_minibatch_list']
            if 'obj2_minibatch_list' in results.keys():
                del results['obj2_minibatch_list']

            results_config = {key: [value] for key, value in config.items() if key in tuple(['adv_estimator','MC', 'pre_epochs', 'pre_lr', 'adv_lr', 'n_epochs', 'lr', 'pref_idx'])}
            results_config.update(results)

            if i == 0:
                results_config_total = pd.DataFrame.from_dict(results_config)
            else:
                results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
        else:
            print(i)

    del hyperparameter_config['npref_prefidx']
    del hyperparameter_config['MCs']

    pareto_front(hyperparameter_config= hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)

    mean_metrics(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)

    hypervolume_compare(hyperparameter_config=hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)
