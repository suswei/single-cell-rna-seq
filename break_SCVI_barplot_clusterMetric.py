import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def barplot_list(data, alg, title, save=None, interest=0, prog=False, figsize=None):
    ind = np.arange(len(alg))  # the x locations for the groups
    width = 0.25  # the width of the bars
    if figsize is None:
        fig = plt.figure()

    else:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if len(data[0]) == 3:
        width = 0.25  # the width of the bars

    else:
        width = 0.15

    rects = []
    color = ["r", "g", "y", "b", "purple"]
    if prog:
        color = ['darkred', "red", "tomato", "salmon"]
    for i in range(len(data[0])):
        rects.append(ax.barh(ind + i * width, data[:, i], width, color=color[i]))

    anchor_param = (0.8, 0.8)
    leg_rec = [x[0] for x in rects]
    leg_lab = ('ASW', 'NMI', 'ARI', "UCA", "BE")
    if prog:
        leg_lab = ["2", "3", "4", "7"]
    ax.legend(leg_rec, leg_lab[:len(data[0])])

    # add some text for labels, title and axes ticks
    ax.set_xlabel(title)
    ax.set_yticks(ind + width)
    ax.set_yticklabels(alg)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

def break_SCVI_barplot_clusterMetric(changed_property):
    if changed_property == 'Change_Library_Size':
        ratios = [1 / 10, 3 / 10]
    elif changed_property == 'Change_Expressed_Gene_Number':
        ratios=[1 / 10, 3 / 10] #I do not add in the ratio 4/5 in the plotting, because otherwise the plot will be too crowded.
    elif changed_property == 'Change_Gene_Expression_Proportion':
        proportions=[0.2, 0.4, 0.5]
        ratios=[2 / 10, 5 / 10, 2, 5]

    clusterMetric_path_original = './result/break_SCVI/ClusteringMetric_OriginalPbmc.csv'
    clusterMetric = pd.read_csv(clusterMetric_path_original)

    if changed_property=='Change_Library_Size' or changed_property=='Change_Expressed_Gene_Number':
        for batch in [0,1]:
            for ratio in ratios:
                clusterMetric_path = './result/break_SCVI/%s/ClusteringMetrics_Batch%s_ratio%s.csv'%(changed_property, batch, ratio)
                intermediate_clusterMetric= pd.read_csv(clusterMetric_path)
                clusterMetric = pd.concat([clusterMetric,intermediate_clusterMetric])
        if changed_property=='Change_Library_Size':
            alg = ["Original","b0_r0.1","b0_r0.3","b1_r0.1","b1_r0.3"]
        else:
            alg = ["Original","b0_r0.1","b0_r0.3","b1_r0.1","b1_r0.3"]
        barplot_list(clusterMetric.iloc[:, 1:].values, alg, 'Clustering metrics of train set', save='./result/break_SCVI/%s/ClusteringMetric_%s.png'%(changed_property, changed_property))

    else:
        for proportion in proportions:
            clusterMetric_total = clusterMetric
            for ratio in [2 / 10, 5 / 10, 2, 5]:
                clusterMetric_path = './result/break_SCVI/%s/ClusteringMetrics_Proportion%s_ratio%s.csv'%(changed_property, proportion, ratio)
                intermediate_clusterMetric = pd.read_csv(clusterMetric_path)
                clusterMetric_total = pd.concat([clusterMetric_total, intermediate_clusterMetric])
            alg = ["Original", "r0.2", "r0.5", "r2", "r5"]
            barplot_list(clusterMetric_total.iloc[:, 1:].values, alg, 'Clustering metrics of train set', save='./result/break_SCVI/%s/ClusteringMetric_%s_Proportion%s.png' % (changed_property, changed_property,proportion))

break_SCVI_barplot_clusterMetric(changed_property ='Change_Library_Size')
break_SCVI_barplot_clusterMetric(changed_property ='Change_Expressed_Gene_Number')
break_SCVI_barplot_clusterMetric(changed_property ='Change_Gene_Expression_Proportion')

#draw barplot of clustering metric of tabula muris with 2 encoder, 2 decoder, or  10 encoder, 2 decoder, and pbmc. Note that the clustering metric for tabula muris is from 100 repetitions, for pbmc is just one
#repetition because I just borrow result before.
clusterMetric_Pbmc = pd.read_csv('./result/break_SCVI/ClusteringMetric_OriginalPbmc.csv')
clusterMetric_Pbmc.rename(columns={'memo':'label'}, inplace=True)
clusterMetric_tabula_2encoder = pd.read_csv('./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/mean_clustering_metrics_muris_tabula_batch_MIScale200.csv').iloc[[0],0:6]
clusterMetric_tabula_10encoder = pd.read_csv('./result/tune_hyperparameter_for_SCVI_MI/muris_tabula_encoder10layers/mean_clustering_metrics_muris_tabula_batch_MIScale200.csv').iloc[[0],0:6]
clusterMetric_total = pd.concat([clusterMetric_Pbmc,clusterMetric_tabula_2encoder])
clusterMetric_total = pd.concat([clusterMetric_total ,clusterMetric_tabula_10encoder])
alg = ['Original_Pbmc', 'Tabula_2encoder', 'Tabula_10encoder']
barplot_list(clusterMetric_total.iloc[:, 1:].values, alg, 'Clustering metrics of train set', save='./result/break_SCVI/ClusteringMetric_Pbmc_Tabula')

