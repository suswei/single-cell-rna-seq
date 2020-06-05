import os
import pandas as pd
import itertools
import pickle
import plotly.graph_objects as go
import cv2

def draw_plot(dataframe, fig_title, save_path):

    data_dict = dataframe.to_dict('list')
    #measured_values = ['nearest_neighbor_estimate','empirical_mutual_info','MI_MINE_train','MI_MINE_test']

    measured_values = ['nearest_neighbor_estimate','empirical_mutual_info','MI_MINE_train','MI_MINE_test','empirical_CD_KL_0_1','CD_KL_0_1_MINE_train','CD_KL_0_1_MINE_test','empirical_CD_KL_1_0','CD_KL_1_0_MINE_train','CD_KL_1_0_MINE_test']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=measured_values,
                         y=[value[0] for key, value in data_dict.items() if key in [ele + '_mean' for ele in measured_values]],
                         error_y=dict(type='data', array=[value[0] for key, value in data_dict.items() if key in [ele + '_std' for ele in measured_values]]),
                         text=[value[0] for key, value in data_dict.items() if key in [ele + '_mean' for ele in measured_values]], )
                  )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        title={'text': fig_title,
               'y': 0.92,
               'x': 0.47,
               'xanchor': 'center',
               'yanchor': 'top'},
        font=dict(size = 10, color ='black', family = 'Arial, sans-serif')
    )
    fig.write_image(save_path)


def result_summary(confounder_type, hyperparameter_config):

    dir_path = './result/MINE_simulation2/{}/'.format(confounder_type)

    hyperparameter_config_subset = {key: value for key, value in hyperparameter_config.items() if key not in ['MCs']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results_config_total = pd.DataFrame()
    for i in range(len(hyperparameter_experiments)*len(hyperparameter_config['MCs'])):
        config_file_path = dir_path + 'taskid{}/config.pkl'.format(i)
        results_file_path = dir_path + 'taskid{}/results.pkl'.format(i)
        if os.path.isfile(config_file_path) and os.path.isfile(results_file_path):
            config = pickle.load(open(config_file_path, "rb"))
            results = pickle.load(open(results_file_path,"rb"))

            results_config = {key: [value] for key, value in config.items() if key in tuple(['confounder_type']) + keys}
            results_config.update(results)

            results_config_total = pd.concat([results_config_total,pd.DataFrame.from_dict(results_config)],axis=0)
        else:
            print(i)

    results_config_mean_std = results_config_total.groupby(list(tuple(['confounder_type']) + keys)).agg(
        nearest_neighbor_estimate_mean=('nearest_neighbor_estimate', 'mean'),
        nearest_neighbor_estimate_std=('nearest_neighbor_estimate', 'std'),
        empirical_mutual_info_mean = ('empirical_mutual_info', 'mean'),
        empirical_mutual_info_std =('empirical_mutual_info', 'std'),
        MI_MINE_train_mean=('MI_MINE_train', 'mean'),
        MI_MINE_train_std=('MI_MINE_train', 'std'),
        MI_MINE_test_mean=('MI_MINE_test', 'mean'),
        MI_MINE_test_std=('MI_MINE_test', 'std'),
        empirical_CD_KL_0_1_mean = ('empirical_CD_KL_0_1','mean'),
        empirical_CD_KL_0_1_std =('empirical_CD_KL_0_1', 'std'),
        CD_KL_0_1_MINE_train_mean=('CD_KL_0_1_MINE_train', 'mean'),
        CD_KL_0_1_MINE_train_std=('CD_KL_0_1_MINE_train', 'std'),
        CD_KL_0_1_MINE_test_mean=('CD_KL_0_1_MINE_test', 'mean'),
        CD_KL_0_1_MINE_test_std=('CD_KL_0_1_MINE_test', 'std'),
        empirical_CD_KL_1_0_mean = ('empirical_CD_KL_1_0', 'mean'),
        empirical_CD_KL_1_0_std=('empirical_CD_KL_1_0', 'std'),
        CD_KL_1_0_MINE_train_mean = ('CD_KL_1_0_MINE_train', 'mean'),
        CD_KL_1_0_MINE_train_std =('CD_KL_1_0_MINE_train', 'std'),
        CD_KL_1_0_MINE_test_mean = ('CD_KL_1_0_MINE_test', 'mean'),
        CD_KL_1_0_MINE_test_std=('CD_KL_1_0_MINE_test', 'std')
    ).reset_index()

    for config_index in range(len(hyperparameter_experiments)):
        temp = hyperparameter_experiments[config_index]
        results_oneconfig_dataset = results_config_mean_std
        for key in keys:
            results_oneconfig_dataset = results_oneconfig_dataset[results_oneconfig_dataset[key]==temp[key]]

        save_path = dir_path + '{}'.format(confounder_type)
        for key in keys:
            save_path += '_{}{}'.format(key, temp[key])
        save_path += '.png'

        fig_title = ''
        for index, key in enumerate(keys): # if there are too many hyperparameters, exhibit the fig title in several lines.
            if index % 3 == 2:
                if index < len(keys) - 1:
                    fig_title += '{}: {}, <br>'.format(key, temp[key])
                else:
                    fig_title += '{}: {}.'.format(key, temp[key])
            else:
                fig_title += '{}: {}, '.format(key, temp[key])

        draw_plot(results_oneconfig_dataset, fig_title, save_path)

def main( ):

    confounder_type = 'discrete'
    hyperparameter_config = {
        'category_num': [2],
        'gaussian_dim': [2, 10],
        'mean_diff': [1, 5],
        'mixture_component_num': [10, 128],
        'gaussian_covariance_type': ['all_identity', 'partial_identity'],
        'samplesize': [12800],
        'activation_fun': ['Leaky_ReLU', 'ELU'],
        'unbiased_loss': [True],
        'MCs': 40 * [1]
    }

    dir_path = './result/MINE_simulation2/{}/'.format(confounder_type)

    result_summary(confounder_type=confounder_type, hyperparameter_config= hyperparameter_config)

    hyperparameter_config_subset =  {key: value for key, value in hyperparameter_config.items() if key not in ['MCs']}
    keys, values = zip(*hyperparameter_config_subset.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]


    video_name = dir_path + '{}.mp4'.format(confounder_type)

    file_paths = []
    for config_index in range(len(hyperparameter_experiments)):
        temp = hyperparameter_experiments[config_index]
        one_image_path = dir_path + '{}'.format(confounder_type)
        for key in keys:
            one_image_path += '_{}{}'.format(key, temp[key])
        one_image_path += '.png'

        if os.path.isfile(one_image_path):
           file_paths += [one_image_path]

    frame = cv2.imread(file_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for file_path in file_paths:
        video.write(cv2.imread(file_path))
    cv2.destroyAllWindows()
    video.release()
