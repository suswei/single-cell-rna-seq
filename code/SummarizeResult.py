import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

def SummarizeResult(result_type: str = 'image', file_paths: list = ['Not_Specified'],
                    subfile_titles: list = ['Not_Specified'], figtitle: str = 'Not_Specified', n_column: int = 2):
    if result_type == 'image':
        if file_paths == ['Not_Specified']:
            print('Please specify the paths to read in the results first.')
        else:
            images = []
            for file_path in file_paths:
                images.append(mpimg.imread(file_path))
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
                    ax[i//2,i-(i//2)*2] = plt.subplot(len(images) / columns + 1, columns, i + 1)
                    ax[i//2,i-(i//2)*2].axis('off')
                    if subfile_titles != ['Not_Specified']:
                       ax[i//2,i-(i//2)*2].set_title(subfile_titles[i])
                    plt.imshow(image)
            if len(images)%columns != 0:
                n_blank = columns - (len(images)%2)
                for j in range(n_blank):
                   ax[-1, -(j+1)].axis('off')
            fig.suptitle(figtitle, y=1.03, fontsize=18, verticalalignment='top')

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

def Average_ClusteringMetric_Barplot(dataset_name: str = "Pbmc", file_number: int = 100, n_hidden_z: int = 30,
                                     n_layers_z: int =3, MineLossScale: float = 1000):
    cluster_metric_dataframes = pd.DataFrame(columns = ['label', 'asw', 'nmi', 'ari', 'uca', 'be'])
    for i in range(file_number):
        if i not in [31, 39, 46, 47, 49, 55, 59,63,67,70,74,75]:

            file_path = "result/Tune_Hyperparameter_For_MineNet/2019-05-08/" + dataset_name + "_Sample" + str(i) + "_Hidden" + str(n_hidden_z) + "_layers" + str(n_layers_z) + "_MineLossScale" + str(MineLossScale) + "_ClusterMetric.csv"

            cluster_metric_dataframe = pd.read_csv(file_path)
            cluster_metric_dataframe.columns = ['label', 'asw', 'nmi', 'ari', 'uca', 'be']
            cluster_metric_dataframes = cluster_metric_dataframes.append(cluster_metric_dataframe)
    vae_train = cluster_metric_dataframes[cluster_metric_dataframes.label.isin(["vae_train"])]
    vae_Mine_train = cluster_metric_dataframes[cluster_metric_dataframes.label.isin(["vaemine_train"])]
    vae_test = cluster_metric_dataframes[cluster_metric_dataframes.label.isin(["vae_test"])]
    vae_Mine_test = cluster_metric_dataframes[cluster_metric_dataframes.label.isin(["vaemine_test"])]

    vae_train_mean = vae_train.mean().values
    vae_train_std = vae_train.std().values

    vae_Mine_train_mean = vae_Mine_train.mean().values
    vae_Mine_train_std = vae_Mine_train.std().values

    vae_test_mean = vae_test.mean().values
    vae_test_std = vae_test.std().values

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

    vae_vae_Mine_mean_dataframe.to_csv('./result/Tune_Hyperparameter_For_Minenet/2019-05-08/mean_clustering_metrics.csv', index = None, header=True)
    vae_vae_Mine_std_dataframe.to_csv('./result/Tune_Hyperparameter_For_Minenet/2019-05-08/std_clustering_metrics.csv', index=None, header=True)

    barplot_list(vae_vae_Mine_train_mean, error_data=vae_vae_Mine_train_std,alg = ["scVI", "scVI+MINE"],
                 title = "Train Set Mean Clustering Metrics from %s samples"%(file_number),
                 save='./result/Tune_Hyperparameter_For_Minenet/2019-05-08/trainset_mean_clustering_metrics_%s_Hidden%s_layers%s_MineLossScale%s_%ssamples'%(dataset_name, n_hidden_z,n_layers_z,MineLossScale,file_number))
    barplot_list(vae_vae_Mine_test_mean, error_data=vae_vae_Mine_test_std, alg=["scVI", "scVI+MINE"],
                 title="Test Set Mean Clustering Metrics from %s samples" % (file_number),
                 save='./result/Tune_Hyperparameter_For_Minenet/2019-05-08/testset_mean_clustering_metrics_%s_Hidden%s_layers%s_MineLossScale%s_%ssamples' %(dataset_name, n_hidden_z, n_layers_z, MineLossScale, file_number))

