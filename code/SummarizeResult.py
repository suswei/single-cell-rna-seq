import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import itertools
import math


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

def choose_config(input_dir_path: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  results_dict: str='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/',
                  dataset_name: str='muris_tabula', nuisance_variable: str='batch', Label: str='muris_tabula_batch_config.*._VaeMI_trainset', config_numbers: int=107):
    clustermetric = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be','MILoss'])
    valid_config = []
    for i in range(config_numbers):
        clustermetric_filepath_oneconfig = input_dir_path + 'config%s/muris_tabula_batch_config%s_ClusterMetric.csv'%(i,i)
        if os.path.isfile(clustermetric_filepath_oneconfig):
            clustermetric_oneconfig = pd.read_csv(clustermetric_filepath_oneconfig)
            valid_config = valid_config + ['%s'%(i)]
            clustermetric = pd.concat([clustermetric, clustermetric_oneconfig], axis=0)
        else:
            continue

    xaxis_index = list(range(1, len(valid_config) + 1))
    xtick_labels = valid_config
    fig = plt.figure(figsize=(10, 7))
    lines1=plt.plot(xaxis_index, clustermetric[clustermetric['Label'].str.match(Label)].loc[:, ['asw']].values, xaxis_index, clustermetric[clustermetric['Label'].str.match(Label)].loc[:, ['nmi']].values,
             xaxis_index, clustermetric[clustermetric['Label'].str.match(Label)].loc[:, ['ari']].values, xaxis_index, clustermetric[clustermetric['Label'].str.match(Label)].loc[:, ['uca']].values,
             xaxis_index, clustermetric[clustermetric['Label'].str.match(Label)].loc[:, ['be']].values)
    plt.legend(('asw', 'nmi','ari','uca','be'),loc='upper right', fontsize=16)
    plt.xticks(xaxis_index, xtick_labels, rotation='vertical', fontsize=14)
    plt.xlabel('config index', fontsize=16)
    #plt.yticks([k / 10 for k in range(13)], [str(n) for n in [k / 10 for k in range(13)]], rotation='horizontal', fontsize=14)
    plt.title('%s, %s, %s, clusteringmetrics' % (dataset_name, nuisance_variable, 'configs'), fontsize=18)
    fig.savefig(results_dict + '%s_%s_%s_clusteringmetrics.png' % (dataset_name, nuisance_variable, 'configs'))
    plt.close(fig)
