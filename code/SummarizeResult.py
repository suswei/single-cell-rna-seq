import os
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import itertools
import math
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def SummarizeResult(result_type: str = 'image', file_paths: list = ['Not_Specified'],
                    subfile_titles: list = ['Not_Specified'], figtitle: str = 'Not_Specified', n_column: int = 2):
    if result_type == 'image':
        if file_paths == ['Not_Specified']:
            print('Please specify the paths to read in the images first.')
        else:
            images = []
            for file_path in file_paths:
                images.append(mpimg.imread(file_path))
            if len(images) == 1:
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                ax = plt.subplot(1,1,1)
                ax.axis('off')
                plt.imshow(images[0])
            else:
                columns = n_column
                if len(images)%columns == 0:
                    fig, ax = plt.subplots(len(images) // columns, columns, figsize=(10 * columns, 7 * (len(images) // columns)))
                else:
                    fig, ax = plt.subplots(len(images) // columns + 1, columns, figsize=(10 * columns, 7 * (len(images) // columns + 1)))
                fig.tight_layout()
                if len(images) <= columns:
                    for i,image in enumerate(images):
                        ax[i] = plt.subplot(1, columns, i + 1)
                        ax[i].axis('off')
                        if subfile_titles != ['Not_Specified']:
                            ax[i].set_title(subfile_titles[i])
                        plt.imshow(image)
                else:
                    for i, image in enumerate(images):
                        if len(images)%columns == 0:
                            ax[i // columns, i - (i // columns) * columns] = plt.subplot(len(images) / columns, columns, i + 1)
                        else:
                            ax[i // columns, i - (i // columns) * columns] = plt.subplot(len(images) / columns + 1, columns, i + 1)
                        ax[i//columns,i-(i//columns)*columns].axis('off')
                        if subfile_titles != ['Not_Specified']:
                           ax[i//columns,i-(i//columns)*columns].set_title(subfile_titles[i])
                        plt.imshow(image)
                if len(images)%columns != 0:
                    n_blank = columns - (len(images)%columns)
                    for j in range(n_blank):
                       ax[-1, -(j+1)].axis('off')
                fig.suptitle(figtitle, y=1.02, fontsize=18, verticalalignment='top')

def barplot_list(data, error_data, alg, title, save=None, interest=0, prog=False, figsize=None):
    ind = np.arange(len(alg))  # the x locations for the groups
    width = 0.25  # the width of the bars
    if figsize is None:
        fig = plt.figure()

    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if len(data[0]) == 3:
        width = 0.2  # the width of the bars

    else:
        width = 0.1

    rects = []
    color = ["r", "g", "y", "b", "purple"]
    if prog:
        color = ['darkred', "red", "tomato", "salmon"]
    for i in range(len(data[0])):
        rects.append(ax.barh(ind + i * width, data[:, i], width, xerr=error_data[:,i], color=color[i]))

    anchor_param = (0.8, 0.8)
    leg_rec = [x[0] for x in rects]
    leg_lab = ('ASW', 'NMI', 'ARI', "UCA", "BE")
    if prog:
        leg_lab = ["2", "3", "4", "7"]
    ax.legend(leg_rec, leg_lab[:len(data[0])],loc='center right')

    # add some text for labels, title and axes ticks
    ax.set_xlabel(title)
    ax.set_yticks(ind + width)
    ax.set_yticklabels(alg)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)
    plt.close(fig)

def Average_ClusteringMetric_Barplot(dataset_name: str = "muris_tabula",nuisance_variable: str = 'batch', results_dict: str = 'NA', n_sample: int = 100, hyperparameter_config = {'MIScale':[1000,5000,10000,50000,100000]}):

    cluster_metric_dataframes = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
    for i in range(n_sample):
        file_path = results_dict + '%s_%s_sample%s_ClusterMetric.csv'%(dataset_name, nuisance_variable, i)
        cluster_metric_dataframe = pd.read_csv(file_path)
        cluster_metric_dataframes = pd.concat([cluster_metric_dataframes, cluster_metric_dataframe], axis=0)

    vae_train = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('%s_%s_sample.*._Vae_trainset$'%(dataset_name,nuisance_variable))]
    vae_test = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('%s_%s_sample.*._Vae_testset$'%(dataset_name,nuisance_variable))]

    vae_train_mean = vae_train.mean().values
    vae_train_std = vae_train.std().values

    vae_test_mean = vae_test.mean().values
    vae_test_std = vae_test.std().values

    vae_vae_MI_train_mean_all = vae_train_mean
    vae_vae_MI_train_std_all = vae_train_std

    vae_vae_MI_test_mean_all = vae_test_mean
    vae_vae_MI_test_std_all = vae_test_std

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for j in range(len(hyperparameter_experiments)):
        key, value = zip(*hyperparameter_experiments[j].items())
        MIScale = value[0]

        vae_MI_train = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('%s_%s_MIScale%s_sample.*._VaeMI_trainset$'%(dataset_name, nuisance_variable,MIScale))]
        vae_MI_test = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('%s_%s_MIScale%s_sample.*._VaeMI_testset$'%(dataset_name, nuisance_variable,MIScale))]

        vae_MI_train_mean = vae_MI_train.mean().values
        vae_MI_train_std = vae_MI_train.std().values

        vae_MI_test_mean = vae_MI_test.mean().values
        vae_MI_test_std = vae_MI_test.std().values

        vae_vae_MI_train_mean = np.vstack([vae_train_mean, vae_MI_train_mean])
        vae_vae_MI_train_std = np.vstack([vae_train_std, vae_MI_train_std])

        vae_vae_MI_test_mean = np.vstack([vae_test_mean, vae_MI_test_mean])
        vae_vae_MI_test_std = np.vstack([vae_test_std, vae_MI_test_std])

        vae_vae_MI_mean_dataframe = pd.DataFrame(np.vstack([vae_vae_MI_train_mean, vae_vae_MI_test_mean]), columns=['asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
        vae_vae_MI_mean_dataframe = pd.concat([pd.DataFrame({'label':['vae_train_mean','vae+MI_train_mean','vae_test_mean','vae+MI_test_mean']}),vae_vae_MI_mean_dataframe],axis=1)

        vae_vae_MI_std_dataframe = pd.DataFrame(np.vstack([vae_vae_MI_train_std, vae_vae_MI_test_std]),columns=['asw', 'nmi', 'ari', 'uca', 'be', 'MILoss'])
        vae_vae_MI_std_dataframe = pd.concat([pd.DataFrame({'label': ['vae_train_std', 'vae+MI_train_std', 'vae_test_std', 'vae+MI_test_std']}), vae_vae_MI_std_dataframe], axis=1)

        vae_vae_MI_mean_dataframe.to_csv(results_dict+'mean_clustering_metrics_%s_%s_MIScale%s.csv'%(dataset_name, nuisance_variable, MIScale), index = None, header=True)
        vae_vae_MI_std_dataframe.to_csv(results_dict+'std_clustering_metrics_%s_%s_MIScale%s.csv'%(dataset_name, nuisance_variable, MIScale), index=None, header=True)

        barplot_list(vae_vae_MI_train_mean[:,0:5], error_data=vae_vae_MI_train_std[:,0:5],alg = ["scVI", "scVI+MI"],
                     title = "Train Set Mean Clustering Metrics %s %s MIScale%s"%(dataset_name, nuisance_variable, MIScale),
                     save=results_dict+'trainset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples'%(dataset_name, nuisance_variable, MIScale,n_sample))
        barplot_list(vae_vae_MI_test_mean[:,0:5], error_data=vae_vae_MI_test_std[:,0:5], alg=["scVI", "scVI+MI"],
                     title="Test Set Mean Clustering Metrics %s %s MIScale%s"%(dataset_name, nuisance_variable, MIScale),
                     save=results_dict+'testset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples' %(dataset_name, nuisance_variable, MIScale, n_sample))


        vae_vae_MI_train_mean_all = pd.DataFrame(np.vstack([vae_vae_MI_train_mean_all, vae_MI_train_mean]),columns=['asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
        vae_vae_MI_train_std_all = pd.DataFrame(np.vstack([vae_vae_MI_train_std_all, vae_MI_train_std]),columns=['asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
        vae_vae_MI_test_mean_all = pd.DataFrame(np.vstack([vae_vae_MI_test_mean_all, vae_MI_test_mean]),columns=['asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
        vae_vae_MI_test_std_all = pd.DataFrame(np.vstack([vae_vae_MI_test_std_all, vae_MI_test_std]),columns=['asw', 'nmi', 'ari', 'uca', 'be','MILoss'])

    for type in ['training','testing']:
        if type == 'training':
            mean_dataset = vae_vae_MI_train_mean_all
            std_dataset = vae_vae_MI_train_std_all
        else:
            mean_dataset = vae_vae_MI_test_mean_all
            std_dataset = vae_vae_MI_test_std_all

        xaxis_index = list(range(1, len(hyperparameter_config['MIScale'])+2))
        xtick_labels = ['scvi'] + [str(n) for n in hyperparameter_config['MIScale']]
        fig = plt.figure(figsize=(10, 7))
        plt.plot(xaxis_index, mean_dataset.loc[:,['asw']].values, xaxis_index, mean_dataset.loc[:,['nmi']].values, xaxis_index, mean_dataset.loc[:,['ari']].values, xaxis_index, mean_dataset.loc[:,['uca']].values, xaxis_index, mean_dataset.loc[:,['be']].values)
        plt.errorbar(xaxis_index, mean_dataset.loc[:,['asw']].values, yerr=std_dataset.loc[:,['asw']].values, label='asw')
        plt.errorbar(xaxis_index, mean_dataset.loc[:, ['nmi']].values, yerr=std_dataset.loc[:, ['nmi']].values, label='nmi')
        plt.errorbar(xaxis_index, mean_dataset.loc[:, ['ari']].values, yerr=std_dataset.loc[:, ['ari']].values, label='ari')
        plt.errorbar(xaxis_index, mean_dataset.loc[:, ['uca']].values, yerr=std_dataset.loc[:, ['uca']].values, label='uca')
        plt.errorbar(xaxis_index, mean_dataset.loc[:, ['be']].values, yerr=std_dataset.loc[:, ['be']].values,label='be')
        plt.errorbar(xaxis_index, mean_dataset.loc[:, ['MILoss']].values, yerr=std_dataset.loc[:, ['MILoss']].values,label='MILoss')
        plt.legend(loc='upper right', fontsize=16)
        plt.xticks(xaxis_index, xtick_labels, rotation='horizontal',fontsize=14)
        plt.xlabel('index', fontsize=16)
        plt.yticks([k/10 for k in range(13)], [str(n) for n in [k/10 for k in range(13)]], rotation='horizontal', fontsize=14)
        plt.title('%s, %s, %s, clusteringmetrics' % (dataset_name, nuisance_variable,type), fontsize=18)
        fig.savefig(results_dict + '%s_%s_%s_clusteringmetrics.png' % (dataset_name, nuisance_variable,type))
        plt.close(fig)


def Summarize_EstimatedMI_with_TrueMI(file_path: str = 'NA', method: str = 'NA', distribution: str = 'NA', gaussian_dimensions: list = [None, None]):
    #file path should be in the format r'\', or using '\\'

    dataframe = pd.read_csv(file_path)
    result_dict = os.path.dirname(file_path)

    for gaussian_dimension in gaussian_dimensions:
        if method == 'nearest_neighbor':
            subset_dataframes = [dataframe[(dataframe.method == method) & (dataframe.distribution == distribution) & (dataframe.gaussian_dimension == gaussian_dimension)].sort_values('rho', ascending=True)]
        else:
            subset_dataframes = [dataframe[(dataframe.method==method) & (dataframe.distribution==distribution) & (dataframe.gaussian_dimension==gaussian_dimension)  & (dataframe.training_or_testing=='training')].sort_values('rho', ascending=True)]
            subset_dataframes += [dataframe[(dataframe.method == method) & (dataframe.distribution == distribution) & (dataframe.gaussian_dimension == gaussian_dimension) & (dataframe.training_or_testing == 'testing')].sort_values('rho', ascending=True)]

        for i in range(len(subset_dataframes)):
            subset_dataframe = subset_dataframes[i]

            if subset_dataframe.loc[:,['method']].iloc[0,0] == 'nearest_neighbor':
                true_MI = subset_dataframe.loc[:,['true_MI']].values
                estimated_MI = subset_dataframe.loc[:, ['estimated_MI']].values
                std = subset_dataframe.loc[:, ['standard_deviation']].values

                sorted_index = sorted(range(len(true_MI)), key=true_MI.__getitem__)
                sorted_trueMI = [true_MI[i] for i in sorted_index]
                sorted_estimatedMI = [estimated_MI[i] for i in sorted_index]
                sorted_std = [std[i] for i in sorted_index]

                xaxis_index = list(range(1, len(true_MI) + 1))

                fig = plt.figure(figsize=(10, 7))
                lines1 = plt.plot(xaxis_index, sorted_trueMI, xaxis_index, sorted_estimatedMI)
                plt.errorbar(xaxis_index, sorted_estimatedMI, yerr=sorted_std, fmt='o')
                plt.setp(lines1[0], linewidth=2)
                plt.setp(lines1[1], linewidth=2)
                plt.legend(('true MI', 'estimated MI'), loc='upper right',fontsize=16)
                plt.xlabel('index',fontsize=16)
                plt.title('%s, %s, gaussian_dim%s' % (method, distribution, gaussian_dimension),fontsize=18)
                fig.savefig(result_dict + '\\%s_%s_gaussian_dim%s.png' % (method, distribution, gaussian_dimension))
                plt.close(fig)
            elif subset_dataframe.loc[:,['method']].iloc[0,0] in ['Mine_Net','Mine_Net4'] and subset_dataframe.loc[:,['distribution']].iloc[0,0] =='categorical':
                true_MI = subset_dataframe.loc[:, ['true_MI']].values
                estimated_MI = subset_dataframe.loc[:, ['estimated_MI']].values
                type = subset_dataframe.loc[:, ['training_or_testing']].iloc[0, 0]

                sorted_index = sorted(range(len(true_MI)), key=true_MI.__getitem__)
                sorted_trueMI = [true_MI[i] for i in sorted_index]
                sorted_estimatedMI = [estimated_MI[i] for i in sorted_index]

                xaxis_index = list(range(1, len(true_MI) + 1))

                fig = plt.figure(figsize=(10, 7))
                lines1 = plt.plot(xaxis_index, sorted_trueMI, xaxis_index, sorted_estimatedMI)
                plt.setp(lines1[0], linewidth=2)
                plt.setp(lines1[1], linewidth=2)
                plt.legend(('true MI', 'estimated MI'), loc='upper right', fontsize=16)
                plt.xlabel('index', fontsize=16)
                plt.title('%s, %s, gaussian_dim%s, %s' % (method, distribution, gaussian_dimension, type), fontsize=18)
                fig.savefig(result_dict + '\\%s_%s_gaussian_dim%s_%s.png' % (method, distribution, gaussian_dimension, type))
                plt.close(fig)
            elif subset_dataframe.loc[:,['method']].iloc[0,0] =='Mine_Net4' and subset_dataframe.loc[:,['distribution']].iloc[0,0] in ['gaussian','lognormal']:
                rho = subset_dataframe.loc[:,['rho']].values
                true_MI = subset_dataframe.loc[:,['true_MI']].values
                estimated_MI = subset_dataframe.loc[:,['estimated_MI']].values
                type = subset_dataframe.loc[:,['training_or_testing']].iloc[0,0]

                fig = plt.figure(figsize=(10, 7))
                lines1 = plt.plot(rho, true_MI, rho, estimated_MI)
                plt.setp(lines1[0], linewidth=2)
                plt.setp(lines1[1], linewidth=2)
                plt.legend(('true MI', 'estimated MI'),loc='upper right', fontsize=16)
                plt.xlabel('rho',fontsize=16)
                plt.title('%s, %s, gaussian_dim%s, %s'%(method, distribution, gaussian_dimension,type),fontsize=18)
                fig.savefig(result_dict + '\\%s_%s_gaussian_dim%s_%s.png'%(method, distribution, gaussian_dimension,type))
                plt.close(fig)

def clustermetric_vs_ELBO(input_dir_path: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  results_dict: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  dataset_name: str='muris_tabula', nuisance_variable: str='batch', Label_list: list=['trainset', 'testset'], config_numbers: int=107, hyperparameter_index: int=1,
                  starting_epoch: int=7):

    for i in range(config_numbers):
        clustermetric_filepath_oneconfig = input_dir_path + 'scviconfig%s/clustermetrics_duringtraining.csv'%(i)
        if os.path.isfile(clustermetric_filepath_oneconfig):
            clustermetric_oneconfig = pd.read_csv(clustermetric_filepath_oneconfig)

            for Label in Label_list:
                clustermetric_oneconfig_subset = clustermetric_oneconfig[clustermetric_oneconfig['Label'].str.match(Label)]
                for k in [starting_epoch]:
                    fig = plt.figure(figsize=(10, 7))
                    lines1 = plt.plot(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:], clustermetric_oneconfig_subset.loc[:, ['asw']].values[k:],
                                      clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:], clustermetric_oneconfig_subset.loc[:, ['nmi']].values[k:],
                                      clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:], clustermetric_oneconfig_subset.loc[:, ['ari']].values[k:],
                                      clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:], clustermetric_oneconfig_subset.loc[:, ['uca']].values[k:],
                                      clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:], clustermetric_oneconfig_subset.loc[:, ['be']].values[k:])
                    plt.legend(('asw', 'nmi', 'ari', 'uca', 'be'), loc='upper right', fontsize=16)
                    epochs_list = np.ndarray.tolist(clustermetric_oneconfig_subset.loc[:, ['nth_epochs']].values[k:])
                    for j in range(len(epochs_list)):
                        plt.text(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:][j], clustermetric_oneconfig_subset.loc[:, ['asw']].values[k:][j], '%s' % (epochs_list[j]), horizontalalignment='right')
                        plt.text(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:][j], clustermetric_oneconfig_subset.loc[:, ['nmi']].values[k:][j], '%s' % (epochs_list[j]),horizontalalignment='right')
                        plt.text(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:][j],clustermetric_oneconfig_subset.loc[:, ['ari']].values[k:][j], '%s' % (epochs_list[j]),horizontalalignment='right')
                        plt.text(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:][j],clustermetric_oneconfig_subset.loc[:, ['uca']].values[k:][j], '%s' % (epochs_list[j]),horizontalalignment='right')
                        plt.text(clustermetric_oneconfig_subset.loc[:, ['ELBO']].values[k:][j],clustermetric_oneconfig_subset.loc[:, ['be']].values[k:][j], '%s' % (epochs_list[j]),horizontalalignment='right')

                    plt.title('config%s, %s, clusteringmetrics vs ELBO from epoch%s' % (i, Label, k*10),fontsize=18)
                    fig.savefig(results_dict + 'scviconfig%s/%s_%s_config%s_%s_clustervsELBO_epoch%s.png' % (i, dataset_name, nuisance_variable, i, Label, k*10))
                    plt.close(fig)
        else:
            continue
    if hyperparameter_index==1:
        hyperparameter_config = {
            'n_layers_encoder': [2, 10],
            'n_layers_decoder': [10, 2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [1e-2, 1e-3],
            'n_epochs': [350],
            'nsamples_z': [200],
            'adv': [False],
            'std': [True, False]
        }
    elif hyperparameter_index==2:
        hyperparameter_config = {
            'n_layers_encoder': [2, 10],
            'n_layers_decoder': [10, 2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [1, 1e-1, 1e-2, 1e-3],
            'n_epochs': [350, 800],
            'nsamples_z': [200],
            'adv': [False],
            'std': [True]
        }
    elif hyperparameter_index==3:
        hyperparameter_config = {
            'n_layers_encoder': [2, 10],
            'n_layers_decoder': [10, 2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [1e-2],
            'n_epochs': [1200],
            'nsamples_z': [200],
            'adv': [False],
            'std': [True]
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_dataframe = pd.DataFrame( columns=['n_layers_encoder', 'n_layers_decoder', 'lr', 'std'])

    for m in range(config_numbers):
        key, value = zip(*hyperparameter_experiments[m].items())
        intermediate_dataframe = pd.DataFrame.from_dict({'n_layers_encoder': [value[0]], 'n_layers_decoder': [value[1]],'lr': [value[9]], 'std': [value[13]]})
        hyperparameter_dataframe = pd.concat([hyperparameter_dataframe, intermediate_dataframe], axis=0)

    hyperparameter_dataframe.index = range(0, config_numbers)
    configs = pd.DataFrame.from_dict({'configs': list(range(0, config_numbers))})
    hyperparameter_dataframe = pd.concat([configs, hyperparameter_dataframe], axis=1)
    hyperparameter_dataframe.to_csv(results_dict + '%s_%s_%s_hyperparameters.csv' % (dataset_name, nuisance_variable, 'configs'), index=None,header=True)

def choose_adv_lr_min_max_MI(input_dir_path: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  results_dict: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  dataset_name: str='muris_tabula', nuisance_variable: str='batch', n_layer_number: int=2, adv_lr_number: int=6, MIScale_number: int=2, repetition_id: list=[0]):
    hyperparameter_config = {
        'n_layers_encoder': [2, 10],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'dropout_rate': [0.1],
        'reconstruction_loss': ['zinb'],
        'use_batches': [True],
        'use_cuda': [False],
        'MIScale': [0, 0.9],
        'train_size': [0.8],
        'lr': [1e-2],
        'adv_lr': [1, 1e-1, 1e-2, 5e-3, 1e-3, 5e-4],
        'pre_n_epochs': [100],  # 100
        'n_epochs': [700],
        'nsamples_z': [200],
        'adv': [True],
        'Adv_Net_architecture': [[256] * 10],
        'pre_adv_epochs': [350],
        'adv_epochs': [3],
        'activation_fun': ['ELU'],
        'unbiased_loss': [True],
        'initial': ['xavier_normal'],
        'adv_model': ['MI'],
        'optimiser': ['Adam'],
        'adv_drop_out': [0.2],
        'std': [True],
        'taskid': [0, 10, 50, 80]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    repetition_number = len(repetition_id)
    for k in range(n_layer_number):
        for rep in range(repetition_number):
            smaller_MIScale = pd.DataFrame(columns=['adv_lr', 'MIScale0'])
            higher_MIScale = pd.DataFrame(columns=['adv_lr', 'MIScale0.9'])
            config_list = []
            for Scale_N in range(MIScale_number):
                for adv_lr_N in range(adv_lr_number):
                    config_list += [k*MIScale_number*adv_lr_number*repetition_number + Scale_N*adv_lr_number*repetition_number + adv_lr_N*repetition_number + rep]
            for i in config_list:
                MI_training_oneconfig_path = input_dir_path + 'config%s/%s_%s_config%s_minibatch_info.csv' % (i, dataset_name, nuisance_variable, i)
                key, value = zip(*hyperparameter_experiments[i].items())
                adv_lr = value[11]
                MIScale = value[8]
                if os.path.isfile(MI_training_oneconfig_path):
                    MI_training_oneconfig = pd.read_csv(MI_training_oneconfig_path)
                    if MIScale == 0:
                        intermediate1 = pd.DataFrame.from_dict({'adv_lr': [adv_lr]*(MI_training_oneconfig.shape[0])})
                        intermediate1 = pd.concat([intermediate1, MI_training_oneconfig.loc[:,'minibatch_penalty']], axis=1)
                        intermediate1.columns = ['adv_lr', 'MIScale0']
                        smaller_MIScale = pd.concat([smaller_MIScale, intermediate1], axis=0)
                    elif MIScale == 0.9:
                        intermediate2 = pd.DataFrame.from_dict({'adv_lr': [adv_lr] * (MI_training_oneconfig.shape[0])})
                        intermediate2 = pd.concat([intermediate2, MI_training_oneconfig.loc[:, 'minibatch_penalty']], axis=1)
                        intermediate2.columns = ['adv_lr', 'MIScale0.9']
                        higher_MIScale = pd.concat([higher_MIScale, intermediate2], axis=0)
                else:
                    if MIScale == 0:
                        intermediate1 = pd.DataFrame.from_dict({'adv_lr': [adv_lr] * 42000, 'MIScale0': [0]*42000})
                        smaller_MIScale = pd.concat([smaller_MIScale, intermediate1], axis=0)
                    elif MIScale == 0.9:
                        intermediate2 = pd.DataFrame.from_dict({'adv_lr': [adv_lr] * 42000, 'MIScale0.9': [0] * 42000})
                        higher_MIScale = pd.concat([higher_MIScale, intermediate2], axis=0)

            smaller_higher_MIScale = pd.concat([smaller_MIScale, higher_MIScale.iloc[:,1]], axis=1)
            smaller_higher_MIScale2 = pd.melt(smaller_higher_MIScale, id_vars=['adv_lr'], value_vars=['MIScale0', 'MIScale0.9'], var_name='MIScale')
            fig, ax = plt.subplots(figsize=(10, 7))
            ax = sns.boxplot(x='adv_lr', y='value', data=smaller_higher_MIScale2, hue='MIScale')
            ax.set(ylim=(-3, 0.7))
            if k == 0:
                ax.set_title('n_layer_encoder%s, repid%s' % (2, repetition_id[rep]))
                plt.savefig(results_dict + '%s_%s_encoder%s_repid%s_MI_adv_lr.png' % (dataset_name, nuisance_variable, 2, repetition_id[rep]))
            elif k == 1:
                ax.set_title('n_layer_encoder%s, repid%s' % (10, repetition_id[rep]))
                plt.savefig(results_dict + '%s_%s_encoder%s_repid%s_MI_adv_lr.png' % (dataset_name, nuisance_variable, 10, repetition_id[rep]))

    image_folder = results_dict
    video_name = results_dict + 'stdMI_stdreconstloss.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def choose_config(input_dir_path: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  results_dict: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  dataset_name: str='muris_tabula', nuisance_variable: str='batch', Label_list: list=['trainset', 'testset'], config_numbers: int=107,
                  hyperparameter_config_index: int=2, adv: str='MI', n_layer_number: int=2, repetition_id: list=[0]):
    hyperparameter_config = {
        'n_layers_encoder': [2, 10],
        'n_layers_decoder': [2],
        'n_hidden': [128],
        'n_latent': [10],
        'dropout_rate': [0.1],
        'reconstruction_loss': ['zinb'],
        'use_batches': [True],
        'use_cuda': [False],
        'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'train_size': [0.8],
        'lr': [1e-2],
        'adv_lr': [1e-2],
        'pre_n_epochs': [100],  # 100
        'n_epochs': [700],
        'nsamples_z': [200],
        'adv': [True],
        'Adv_Net_architecture': [[256] * 10],
        'pre_adv_epochs': [350],
        'adv_epochs': [3],
        'activation_fun': ['ELU'],
        'unbiased_loss': [True],
        'initial': ['xavier_normal'],
        'adv_model': ['MI'],
        'optimiser': ['Adam'],
        'adv_drop_out': [0.2],
        'std': [True],
        'taskid': [0, 10, 50, 80]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    fig = plt.figure()

    repetition_number = len(repetition_id)
    for n_layer in range(n_layer_number):
        clustermetric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be', 'std_penalty', 'std_ELBO'])
        valid_config = []
        for i in range(n_layer*10*repetition_number, (n_layer+1)*10*repetition_number):
            clustermetric_filepath_oneconfig = input_dir_path + 'config%s/muris_tabula_batch_config%s_ClusterMetric.csv'%(i,i)
            if os.path.isfile(clustermetric_filepath_oneconfig):
                valid_config = valid_config + [i]
                clustermetric_oneconfig = pd.read_csv(clustermetric_filepath_oneconfig)
                clustermetric = pd.concat([clustermetric, clustermetric_oneconfig], axis=0)
            else:
                continue

        valid_config_rep = []
        for i in valid_config:
            valid_config_rep.append(i)
            valid_config_rep.append(i)

        clustermetric = pd.concat([pd.DataFrame.from_dict({'configs': valid_config_rep}), clustermetric.reset_index(drop=True)], axis=1)

        for rep in range(repetition_number):
            config_list = []
            for i in valid_config_rep:
                if (i % repetition_number == rep):
                    config_list.append(i)

            for Label in ['trainset']:
                clustermetric_half = clustermetric.loc[clustermetric['configs'].isin(config_list)]
                std_ELBO = clustermetric_half[clustermetric_half['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['std_ELBO']].values[0:]
                std_penalty = clustermetric_half[clustermetric_half['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['std_penalty']].values[0:]

                fig = plt.figure(figsize=(10, 7))
                lines1 = plt.plot(std_ELBO, std_penalty)

                unique_list = []
                for i in config_list:
                    if i not in unique_list:
                        unique_list.append(i)

                scale_list = [(i-n_layer*repetition_number*10-rep) / (10*repetition_number) for i in unique_list]

                for i in range(len(scale_list)):
                    plt.text(std_ELBO[i], std_penalty[i], '%s' % (scale_list[i]),horizontalalignment='right', fontsize=12)

                plt.xlabel('std_ELBO', fontsize=16)
                plt.ylabel('std_penalty', fontsize=16)

                if adv == 'MI':
                    plt.title('%s, %s, encoder%s, repid%s, %s, stdMI stdreconstloss' % (dataset_name, nuisance_variable, [2,10][n_layer], repetition_id[rep], Label),fontsize=18)
                    fig.savefig(results_dict + '%s_%s_encoder%s_repid%s_%s_stdMI_stdreconstloss.png' % (dataset_name, nuisance_variable, [2,10][n_layer], repetition_id[rep], Label))
                    plt.close(fig)
    image_folder = results_dict
    video_name = results_dict + 'stdMI_stdreconstloss.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

    '''
        for Label in Label_list:
            xaxis_index = list(range(1, len(valid_config) + 1))
            xtick_labels = valid_config
            fig = plt.figure(figsize=(10, 7))
            lines1=plt.plot(xaxis_index, clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['asw']].values,
                            xaxis_index, clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['nmi']].values,
                            xaxis_index, clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['ari']].values,
                            xaxis_index, clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['uca']].values,
                            xaxis_index, clustermetric[clustermetric['Label'].str.match('muris_tabula_batch_config.*._VaeMI_' + Label)].loc[:, ['be']].values)
            plt.legend(('asw', 'nmi','ari','uca','be'),loc='upper right', fontsize=16)
            plt.xticks(xaxis_index, xtick_labels, rotation='vertical', fontsize=14)
            plt.xlabel('config index', fontsize=16)
            plt.title('%s, %s, encoder%s, %s, clusteringmetrics' % (dataset_name, nuisance_variable, [2,10][n_layer], Label), fontsize=18)
            fig.savefig(results_dict + '%s_%s_encoder%s_%s_clusteringmetrics.png' % (dataset_name, nuisance_variable, [2,10][n_layer], Label))
            plt.close(fig)

        for p in range(n_layer*10*repetition_number, (n_layer+1)*10*repetition_number):
            key, value = zip(*hyperparameter_experiments[p].items())
            n_layer_encoder = value[0]
            MIScale = value[8]
            repid = value[26]
            clustermetric_trainingprocess_filepath = input_dir_path + 'config%s/clustermetrics_duringtraining.csv'%(p)
            if os.path.isfile(clustermetric_trainingprocess_filepath):
                clustermetric_trainingprocess_oneconfig = pd.read_csv(clustermetric_trainingprocess_filepath)
                xaxis_index = list(range(1, clustermetric_trainingprocess_oneconfig.shape[0] + 1))
                for Label in Label_list:
                    fig = plt.figure(figsize=(10, 7))
                    lines1 = plt.plot(xaxis_index, clustermetric_trainingprocess_oneconfig[clustermetric_trainingprocess_oneconfig['Label'].str.match(Label)].loc[:, ['asw']].values,
                                      xaxis_index, clustermetric_trainingprocess_oneconfig[clustermetric_trainingprocess_oneconfig['Label'].str.match(Label)].loc[:, ['nmi']].values,
                                      xaxis_index, clustermetric_trainingprocess_oneconfig[clustermetric_trainingprocess_oneconfig['Label'].str.match(Label)].loc[:, ['ari']].values,
                                      xaxis_index, clustermetric_trainingprocess_oneconfig[clustermetric_trainingprocess_oneconfig['Label'].str.match(Label)].loc[:, ['uca']].values,
                                      xaxis_index, clustermetric_trainingprocess_oneconfig[clustermetric_trainingprocess_oneconfig['Label'].str.match(Label)].loc[:, ['be']].values)
                    plt.legend(('asw', 'nmi', 'ari', 'uca', 'be'), loc='upper right', fontsize=16)
                    plt.xlabel('epoch number', fontsize=16)
                    plt.title('%s, %s, encoder%s, MIScale%s, repid%s, clusteringmetrics' % (dataset_name, nuisance_variable, n_layer_encoder, MIScale, repid), fontsize=18)
                    fig.savefig(results_dict + 'config%s/%s_%s_encoder%s_MIScale%s_repid%s_clusteringmetrics.png' % (p, dataset_name, nuisance_variable, n_layer_encoder, MIScale, repid))
                    plt.close(fig)
    '''
