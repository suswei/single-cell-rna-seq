import os
import itertools
import numpy as np
import pandas as pd
import pickle
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
        dataframe_oneconfig = dataframe[dataframe.pre_epochs.eq(temp['pre_epochs']) & dataframe.n_epochs.eq(temp['n_epochs']) & dataframe.obj2_scale.eq(temp['obj2_scale'])]

        if dataframe_oneconfig.shape[0] > 0:
            pref_idx_list = dataframe_oneconfig.loc[:, 'pref_idx'].values.tolist()
            for pareto_front_type in ['minibatch', 'full_MINE','NN', 'asw', 'nmi', 'ari', 'uca']:

                if pareto_front_type == 'minibatch':
                    save_path = dir_path + '/paretoMTL_preepochs{}_nepochs{}_obj2scale{}_{}_train.png'.format(
                        temp['pre_epochs'],temp['n_epochs'],temp['obj2_scale'], pareto_front_type)

                    fig_title = 'obj1: negative_ELBO, obj2: MINE,<br>{}, {}, <br>pre_epochs: {}, n_epochs:{}, obj2_scale:{}'.format(
                        pareto_front_type, 'train', temp['pre_epochs'], temp['n_epochs'], temp['obj2_scale'])

                    obj1 = dataframe_oneconfig.loc[:, 'obj1_minibatch'].values.tolist()
                    obj2 = dataframe_oneconfig.loc[:, 'obj2_minibatch'].values.tolist()
                    draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                elif pareto_front_type in ['full_MINE','NN', 'asw', 'nmi', 'ari', 'uca']:
                    for type in ['train', 'test']:
                        save_path = dir_path + '/paretoMTL_preepochs{}_nepochs{}_obj2scale{}_{}_{}.png'.format(
                            temp['pre_epochs'], temp['n_epochs'], temp['obj2_scale'], pareto_front_type,type)
                        if pareto_front_type in ['full_MINE','NN']:
                            fig_title = 'obj1: negative_ELBO, obj2: {}, {},<br>pre_epochs: {}, n_epochs:{}, obj2_scale:{}'.format(
                                pareto_front_type, type, temp['pre_epochs'], temp['n_epochs'], temp['obj2_scale'])

                            obj1 = dataframe_oneconfig.loc[:, 'obj1_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneconfig.loc[:, '{}_{}'.format(pareto_front_type,type)].values.tolist()

                            draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)
                        else:
                            fig_title = 'obj1: be, obj2: {},{},<br>pre_epochs: {}, n_epochs:{}, obj2_scale:{}'.format(
                                pareto_front_type, type, temp['pre_epochs'], temp['n_epochs'], temp['obj2_scale'])

                            obj1 = dataframe_oneconfig.loc[:, 'be_{}'.format(type)].values.tolist()
                            obj2 = dataframe_oneconfig.loc[:, '{}_{}'.format(pareto_front_type, type)].values.tolist()

                            draw_pareto_front(obj1, obj2, pref_idx_list, fig_title, save_path)

    video_save_path = dir_path + '/paretoMTL.mp4'

    image_paths = []
    for temp in hyperparameter_experiments:
        for pareto_front_type in ['minibatch', 'full_MINE', 'NN', 'asw', 'nmi', 'ari', 'uca']:
            for type in ['train','test']:
                one_image_path = dir_path + '/paretoMTL_preepochs{}_nepochs{}_obj2scale{}_{}_{}.png'.format(
                            temp['pre_epochs'], temp['n_epochs'], temp['obj2_scale'], pareto_front_type,type)
                if os.path.isfile(one_image_path):
                    image_paths += [one_image_path]

    create_video(video_save_path, image_paths)



def paretoMTL_summary(dataset: str='muris_tabula', confounder: str='batch'):
    dir_path = './result/pareto_front_paretoMTL/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'pre_epochs': [300, 500, 800],
        'n_epochs': [50, 100, 200],
        'obj2_scale': [0.2, 0.3, 0.5],
        'npref_prefidx': [{'npref': n, 'pref_idx': i} for n, i in zip([10] * 10, list(range(10)))]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_config_total = pd.DataFrame()
    for i in range(len(hyperparameter_experiments)):
        config_path = dir_path + '/taskid{}/config.pkl'.format(i)
        results_path = dir_path + '/taskid{}/results.pkl'.format(i)
        if os.path.isfile(config_path) and os.path.isfile(results_path):
            config = pickle.load(open(config_path, "rb"))
            results = pickle.load(open(results_path, "rb"))

            results_config = {key: [value] for key, value in config.items() if key in tuple(['pre_epochs','n_epochs','obj2_scale','pref_idx'])}
            results_config.update(results)

            results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)

    hyperparameter_config.pop('npref_prefidx')

    pareto_front(hyperparameter_config= hyperparameter_config, dataframe=results_config_total, dir_path=dir_path)

