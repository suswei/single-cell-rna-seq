import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import glob2
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

def Average_ClusteringMetric_Barplot(dataset_name: str = "Pbmc", results_dict: str = 'NA', n_sample: int = 100, hyperparameter_config = {'n_hidden_z':[10,30],'n_layers_z':[3,10],'MineLoss_Scale':[1000,5000,10000,50000,100000]}):

    cluster_metric_dataframes = pd.DataFrame(columns=['Label', 'asw', 'nmi', 'ari', 'uca', 'be'])
    for i in range(n_sample):
        file_path = results_dict + dataset_name + '_Sample%s_ClusterMetric.csv' % (i)
        cluster_metric_dataframe = pd.read_csv(file_path)
        cluster_metric_dataframes = pd.concat([cluster_metric_dataframes, cluster_metric_dataframe], axis=0)

    vae_train = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('^sample.*._Vae_trainset$')]
    vae_test = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('^sample.*._Vae_testset$')]

    vae_train_mean = vae_train.mean().values
    vae_train_std = vae_train.std().values

    vae_test_mean = vae_test.mean().values
    vae_test_std = vae_test.std().values

    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for j in range(len(hyperparameter_experiments)):
        key, value = zip(*hyperparameter_experiments[j].items())
        n_hidden_z = value[0]
        n_layers_z = value[1]
        MineLoss_Scale = value[2]


        vae_Mine_train = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('^sample.*._VaeMine_trainset_Hidden%s_layers%s_MineLossScale%s$'%( n_hidden_z,n_layers_z,MineLoss_Scale))]
        vae_Mine_test = cluster_metric_dataframes[cluster_metric_dataframes['Label'].str.match('^sample.*._VaeMine_testset_Hidden%s_layers%s_MineLossScale%s$'%( n_hidden_z,n_layers_z,MineLoss_Scale))]

        vae_Mine_train_mean = vae_Mine_train.mean().values
        vae_Mine_train_std = vae_Mine_train.std().values

        vae_Mine_test_mean = vae_Mine_test.mean().values
        vae_Mine_test_std = vae_Mine_test.std().values

        vae_vae_Mine_train_mean = np.vstack([vae_train_mean, vae_Mine_train_mean])
        vae_vae_Mine_train_std = np.vstack([vae_train_std, vae_Mine_train_std])

        vae_vae_Mine_test_mean = np.vstack([vae_test_mean, vae_Mine_test_mean])
        vae_vae_Mine_test_std = np.vstack([vae_test_std, vae_Mine_test_std])

        vae_vae_Mine_mean_dataframe = pd.DataFrame(np.vstack([vae_vae_Mine_train_mean, vae_vae_Mine_test_mean]), columns=['asw', 'nmi', 'ari', 'uca', 'be'])
        vae_vae_Mine_mean_dataframe = pd.concat([pd.DataFrame({'label':['vae_train_mean','vae+Mine_train_mean','vae_test_mean','vae+Mine_test_mean']}),vae_vae_Mine_mean_dataframe],axis=1)

        vae_vae_Mine_std_dataframe = pd.DataFrame(np.vstack([vae_vae_Mine_train_std, vae_vae_Mine_test_std]),columns=['asw', 'nmi', 'ari', 'uca', 'be'])
        vae_vae_Mine_std_dataframe = pd.concat([pd.DataFrame({'label': ['vae_train_std', 'vae+Mine_train_std', 'vae_test_std', 'vae+Mine_test_std']}), vae_vae_Mine_std_dataframe], axis=1)

        vae_vae_Mine_mean_dataframe.to_csv(results_dict+'mean_clustering_metrics_Hidden%s_layers%s_MineLossScale%s.csv'%(n_hidden_z, n_layers_z, MineLoss_Scale), index = None, header=True)
        vae_vae_Mine_std_dataframe.to_csv(results_dict+'std_clustering_metrics_Hidden%s_layers%s_MineLossScale%s.csv'%(n_hidden_z, n_layers_z, MineLoss_Scale), index=None, header=True)

        barplot_list(vae_vae_Mine_train_mean, error_data=vae_vae_Mine_train_std,alg = ["scVI", "scVI+MINE"],
                     title = "Train Set Mean Clustering Metrics Hidden%s layers%s MineLossScale%s"%(n_hidden_z, n_layers_z, MineLoss_Scale),
                     save=results_dict+'trainset_mean_clustering_metrics_%s_Hidden%s_layers%s_MineLossScale%s_%ssamples'%(dataset_name, n_hidden_z,n_layers_z, MineLoss_Scale,n_sample))
        barplot_list(vae_vae_Mine_test_mean, error_data=vae_vae_Mine_test_std, alg=["scVI", "scVI+MINE"],
                     title="Test Set Mean Clustering Metrics Hidden%s layers%s MineLossScale%s"%(n_hidden_z, n_layers_z, MineLoss_Scale),
                     save=results_dict+'testset_mean_clustering_metrics_%s_Hidden%s_layers%s_MineLossScale%s_%ssamples' %(dataset_name, n_hidden_z, n_layers_z, MineLoss_Scale, n_sample))


def Summarize_Compare_NNEstimator_TrueMI(results_dict: str = 'NA', MineNet_Info = {'model':['Mine_Net'], 'Hyperparameter':{'n_hidden_z': [10], 'n_layers_z':[10,30,50], 'Gaussian_Dimension': [2, 20], 'sample_size': [14388], 'train_size': [0.5]}}):

    if list(MineNet_Info.values())[0][0] == 'Mine_Net':
       final_dataframe = pd.DataFrame(columns=['variable_dimension', 'sample_size', 'rho', 'true_MI', 'estimated_training_MI', 'estimated_testing_MI'])
       filelist = glob2.glob(results_dict + 'Taskid*_Compare_*.csv')
    else:
        final_dataframe = pd.DataFrame(columns=['variable_dimension', 'sample_size', 'rho', 'true_MI', 'estimated_training_MI', 'estimated_testing_MI', 'final_loss'])
        if list(MineNet_Info.values())[0][0] == 'Mine_Net2':
            filelist = [f for f in glob2.glob(results_dict + "Taskid*_Compare2_*.csv")]
        elif list(MineNet_Info.values())[0][0] == 'Mine_Net3':
            filelist = [f for f in glob2.glob(results_dict + "Taskid*_Compare3_*.csv")]
        elif list(MineNet_Info.values())[0][0] == 'Mine_Net4':
            filelist = [f for f in glob2.glob(results_dict + "Taskid*_Compare4_*.csv")]

    for file_path in filelist:
        intermediate_dataframe = pd.read_csv(file_path)
        final_dataframe = pd.concat([final_dataframe, intermediate_dataframe], sort=True)

    if list(MineNet_Info.values())[0][0] == 'Mine_Net':
        MineNet_Info_Dict = list(MineNet_Info.values())[1]
        keys, values = zip(*MineNet_Info_Dict.items())
        MineNet_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i in range(len(MineNet_experiments)):
            key, value = zip(*MineNet_experiments[i].items())
            n_hidden_z = value[0]
            n_layers_z = value[1]
            Gaussian_Dimension = value[2]
            sample_size = value[3]
            train_size = value[4]

            subset_final_dataframe = final_dataframe[(final_dataframe.variable_dimension==Gaussian_Dimension) & (final_dataframe.n_hidden_z==n_hidden_z) & (final_dataframe.n_layers_z==n_layers_z) & (final_dataframe.sample_size==sample_size)].sort_values('rho', ascending=True)
            rho_2 = subset_final_dataframe.loc[:,['rho']].values
            true_MI2 = subset_final_dataframe.loc[:,['true_MI']].values
            estimated_training_MI2 = subset_final_dataframe.loc[:,['estimated_training_MI']].values
            estimated_testing_MI2 = subset_final_dataframe.loc[:,['estimated_testing_MI']].values

            fig = plt.figure(figsize=(10, 7))
            lines1 = plt.plot(rho_2, true_MI2, rho_2, estimated_training_MI2)
            plt.setp(lines1[0], linewidth=2)
            plt.setp(lines1[1], linewidth=2)

            plt.legend(('true MI', 'estimated training MI'),loc='upper right')
            plt.xlabel('rho')
            plt.title('n_hidden_z = %s, n_layers_z = %s, variable_dimension = %s, sample_size =  %s, training'%(n_hidden_z, n_layers_z, Gaussian_Dimension, math.floor(sample_size*train_size)))
            fig.savefig(results_dict + 'Mine_Net_n_hidden_z%s_n_layers_z%s_variable_dimension%s_sample_size%s_training.png'%(n_hidden_z, n_layers_z, Gaussian_Dimension, math.floor(sample_size*train_size)))
            plt.close(fig)

            fig = plt.figure(figsize=(10, 7))
            lines2 = plt.plot(rho_2, true_MI2, rho_2, estimated_testing_MI2)
            plt.setp(lines2[0], linewidth=2)
            plt.setp(lines2[1], linewidth=2)

            plt.legend(('true MI', 'estimated testing MI'), loc='upper right')
            plt.xlabel('rho')
            plt.title('n_hidden_z = %s, n_layers_z = %s, variable_dimension = %s, sample_size = %s, testing' % (n_hidden_z, n_layers_z, Gaussian_Dimension, math.floor(sample_size*(1-train_size))))
            fig.savefig(results_dict + 'Mine_Net_n_hidden_z%s_n_layers_z%s_variable_dimension%s_sample_size%s_testing.png' %(n_hidden_z, n_layers_z, Gaussian_Dimension, math.floor(sample_size*(1-train_size))))
            plt.close(fig)

    if list(MineNet_Info.values())[0][0] in ['Mine_Net2','Mine_Net3','Mine_Net4']:
        MineNet_Info_Dict = list(MineNet_Info.values())[1]
        keys, values = zip(*MineNet_Info_Dict.items())
        MineNet_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i in range(len(MineNet_experiments)):
            key, value = zip(*MineNet_experiments[i].items())
            Gaussian_Dimension = value[0]
            sample_size = value[1]
            train_size = value[2]

            subset_final_dataframe = final_dataframe[(final_dataframe.variable_dimension == Gaussian_Dimension) & (final_dataframe.sample_size == sample_size)].sort_values('rho', ascending=True)
            rho_2 = subset_final_dataframe.loc[:, ['rho']].values
            true_MI2 = subset_final_dataframe.loc[:, ['true_MI']].values
            estimated_training_MI2 = subset_final_dataframe.loc[:, ['estimated_training_MI']].values
            estimated_testing_MI2 = subset_final_dataframe.loc[:, ['estimated_testing_MI']].values

            fig = plt.figure(figsize=(10, 7))
            lines1 = plt.plot(rho_2, true_MI2, rho_2, estimated_training_MI2)
            plt.setp(lines1[0], linewidth=2)
            plt.setp(lines1[1], linewidth=2)

            plt.legend(('true MI', 'estimated training MI'), loc='upper right')
            plt.xlabel('rho')
            plt.title('variable_dimension = %s, sample_size =  %s, training' % (Gaussian_Dimension, math.floor(sample_size*train_size)))
            fig.savefig(results_dict  + list(MineNet_Info.values())[0][0] + '_variable_dimension%s_sample_size%s_training.png'%(Gaussian_Dimension, math.floor(sample_size*train_size)))
            plt.close(fig)

            fig = plt.figure(figsize=(10, 7))
            lines2 = plt.plot(rho_2, true_MI2, rho_2, estimated_testing_MI2)
            plt.setp(lines2[0], linewidth=2)
            plt.setp(lines2[1], linewidth=2)

            plt.legend(('true MI', 'estimated testing MI'), loc='upper right')
            plt.xlabel('rho')
            plt.title('variable_dimension = %s, sample_size = %s, testing' % (Gaussian_Dimension, math.floor(sample_size*(1-train_size))))
            fig.savefig(results_dict + list(MineNet_Info.values())[0][0] + '_variable_dimension%s_sample_size%s_testing.png'%(Gaussian_Dimension, math.floor(sample_size*(1-train_size))))
            plt.close(fig)

