import os
import itertools
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2

#find the parento front when the goal is to minimize two objectives
def simple_cull(inputPoints, dominates):
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
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
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

def dominates(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)


def draw_pareto_front(obj1, obj2, pareto_front_type, type, save_path):

    inputPoints1 = [[obj1[k], obj2[k]] for k in range(len(obj1))]

    paretoPoints1, dominatedPoints1 = simple_cull(inputPoints1, dominates)

    fig = plt.figure()
    dp = np.array(list(dominatedPoints1))
    pp = np.array(list(paretoPoints1))
    plt.scatter(dp[:, 0], dp[:, 1])
    plt.scatter(pp[:, 0], pp[:, 1], color='red')
    if pareto_front_type == 'minibatch':
        plt.title('obj1: std_negative_ELBO, obj2: std_MINE,\n{}, {}'.format(pareto_front_type, type), fontsize=18)
        plt.xlim(0, 1.2)
    elif pareto_front_type == 'full':
        plt.title('obj1: std_negative_ELBO, obj2: MINE,\n{}, {}'.format(pareto_front_type, type), fontsize=18)
        plt.xlim(0, 1.2)
    else:
        plt.title('obj1: be, obj2: {}, {}'.format(pareto_front_type, type), fontsize=18)
        plt.xlim(-0.2, 0.6)
    plt.xlabel('obj1', fontsize=16)
    plt.ylabel('obj2', fontsize=16)
    fig.savefig(save_path + '/pareto_front_{}_{}.png'.format(pareto_front_type, type))
    plt.close(fig)

def create_video(video_save_path, image_paths):
    #image_paths is a list of image paths to read images into a video
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_save_path, 0, 1, (width, height))
    for file_path in image_paths:
        video.write(cv2.imread(file_path))
    cv2.destroyAllWindows()
    video.release()


def pareto_front(dataframe, save_path):

    for pareto_front_type in ['minibatch', 'full', 'asw', 'nmi', 'ari', 'uca']:
        if pareto_front_type == 'minibatch':
            obj1 = dataframe.loc[:, 'std_obj1_minibatch'].values.tolist()
            obj2 = dataframe.loc[:, 'std_obj2_minibatch'].values.tolist()
            draw_pareto_front(obj1, obj2, pareto_front_type, 'train', save_path)
        elif pareto_front_type in ['full', 'asw', 'nmi', 'ari', 'uca']:
            for type in ['train', 'test']:
                if pareto_front_type == 'full':
                    obj1 = dataframe.loc[:, 'std_obj1_{}set'.format(type)].values.tolist()
                    obj2 = dataframe.loc[:, 'full_MINE_estimator_{}set'.format(type)].values.tolist()
                    draw_pareto_front(obj1, obj2, pareto_front_type, type, save_path)
                else:
                    obj1 = dataframe.loc[:, '{}_be'.format(type, pareto_front_type)].values.tolist()
                    obj2 = dataframe.loc[:, '{}_{}'.format(type, pareto_front_type)].values.tolist()
                    draw_pareto_front(obj1, obj2, pareto_front_type, type, save_path)

    video_save_path = save_path + '/pareto_front.mp4'

    image_paths = []
    for pareto_front_type in ['minibatch', 'full', 'asw', 'nmi', 'ari', 'uca']:
        for type in ['train','test']:
            one_image_path = save_path + '/pareto_front_{}_{}.png'.format(pareto_front_type, type)
            if os.path.isfile(one_image_path):
                image_paths += [one_image_path]

    create_video(video_save_path, image_paths)

def draw_mean_metrics_scales(dataframe, y_axis_variable, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe.loc[:, 'scale'].values.tolist(),
                         y=dataframe.loc[:, '{}_mean'.format(y_axis_variable)].values.tolist(),
                         error_y=dict(type='data', array=dataframe.loc[:, '{}_std'.format(y_axis_variable)].values.tolist()),
                         mode='lines+markers', name= '{}'.format(y_axis_variable))
                  )
    fig.update_xaxes(title_text='lambda')
    if y_axis_variable in ['std_obj1_minibatch', 'std_obj1_train', 'std_obj1_test']:
        fig.update_layout(yaxis=dict(range=[0, 4]))
    fig.update_layout(
        title={'text': '{}_mean'.format(y_axis_variable),
               'y': 0.9,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size=10, color='black', family='Arial, sans-serif')
    )
    fig.write_image(save_path + '/{}_mean.png'.format(y_axis_variable))

def mean_metrics_scales(dataframe, save_path):
    dataframe_mean_std = dataframe.groupby(list(tuple(['scale']))).agg(
        std_obj1_minibatch_mean=('std_obj1_minibatch', 'mean'),
        std_obj1_minibatch_std=('std_obj1_minibatch', 'std'),
        std_obj2_minibatch_mean=('std_obj2_minibatch', 'mean'),
        std_obj2_minibatch_std=('std_obj2_minibatch', 'std'),
        std_obj1_train_mean=('std_obj1_trainset', 'mean'),
        std_obj1_train_std=('std_obj1_trainset', 'std'),
        full_MINE_estimator_train_mean=('full_MINE_estimator_trainset', 'mean'),
        full_MINE_estimator_train_std=('full_MINE_estimator_trainset', 'std'),
        std_obj1_test_mean=('std_obj1_testset', 'mean'),
        std_obj1_test_std=('std_obj1_testset', 'std'),
        full_MINE_estimator_test_mean=('full_MINE_estimator_testset', 'mean'),
        full_MINE_estimator_test_std=('full_MINE_estimator_testset', 'std'),
        asw_train_mean=('train_asw', 'mean'),
        asw_train_std=('train_asw', 'std'),
        nmi_train_mean=('train_nmi', 'mean'),
        nmi_train_std=('train_nmi', 'std'),
        ari_train_mean=('train_ari', 'mean'),
        ari_train_std=('train_ari', 'std'),
        uca_train_mean=('train_uca', 'mean'),
        uca_train_std=('train_uca', 'std'),
        be_train_mean=('train_be', 'mean'),
        be_train_std=('train_be', 'std'),
        asw_test_mean=('test_asw', 'mean'),
        asw_test_std=('test_asw', 'std'),
        nmi_test_mean=('test_nmi', 'mean'),
        nmi_test_std=('test_nmi', 'std'),
        ari_test_mean=('test_ari', 'mean'),
        ari_test_std=('test_ari', 'std'),
        uca_test_mean=('test_uca', 'mean'),
        uca_test_std=('test_uca', 'std'),
        be_test_mean=('test_be', 'mean'),
        be_test_std=('test_be', 'std')
    ).reset_index()

    image_paths = []
    for plot_type in ['std_obj1_minibatch', 'std_obj2_minibatch', 'std_obj1', 'full_MINE_estimator', 'asw', 'nmi', 'ari', 'uca', 'be']:
        if plot_type in ['std_obj1_minibatch', 'std_obj2_minibatch']:
            draw_mean_metrics_scales(dataframe=dataframe_mean_std, y_axis_variable=plot_type, save_path=save_path)
            image_paths += [save_path + '/{}_mean.png'.format(plot_type)]
        else:
            for data_type in ['train','test']:
                y_axis_variable = '{}_{}'.format(plot_type, data_type)
                draw_mean_metrics_scales(dataframe=dataframe_mean_std, y_axis_variable=y_axis_variable, save_path=save_path)
                image_paths += [save_path + '/{}_mean.png'.format(y_axis_variable)]

    create_video(video_save_path=save_path + '/mean_metric.mp4', image_paths = image_paths)

def pareto_front_summary(dataset: str='muris_tabula', confounder: str='batch'):
    dir_path = './result/pareto_front_scVI_MINE/{}/{}'.format(dataset, confounder)
    hyperparameter_config = {
        'scale': [ele / 10 for ele in range(0, 11)],
        'MCs': 20 * [1]
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

            results_config = {key: [value] for key, value in config.items() if key in tuple(['scale'])}
            results_config.update(results)

            results_config_total = pd.concat([results_config_total, pd.DataFrame.from_dict(results_config)], axis=0)
    #because of coding error to std std_obj2_minibatch
    #the wrong code is std_obj2_minibatch = (obj2_minibatch_list[-1] - args.min_obj2)/(args.max_obj1 - args.min_obj2)
    results_config_total['std_obj2_minibatch'] = results_config_total['std_obj2_minibatch'].apply(lambda x: x * (20000 + 0.1))
    pareto_front(dataframe=results_config_total, save_path=dir_path)
    mean_metrics_scales(dataframe=results_config_total, save_path=dir_path)
