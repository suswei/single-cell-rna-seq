import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import rpy2.robjects as robjects
from scipy import sparse
import glob2
import ntpath
import itertools

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
        ratios = [1 / 10, 3 / 10, 3, 10]
    elif changed_property == 'Change_Expressed_Gene_Number':
        ratios=[1 / 10, 3 / 10, 4 / 5]
    elif changed_property == 'Change_Gene_Expression_Proportion':
        proportions=[0.2, 0.4, 0.5]
        ratios=[1 / 10, 3 / 10, 3, 10]

    clusterMetric_path_original = './data/break_SCVI/ClusteringMetric_OriginalPbmc.csv'
    clusterMetric = pd.read_csv(clusterMetric_path_original)

    if changed_property=='Change_Library_Size' or changed_property=='Change_Expressed_Gene_Number':
        for batch in [0,1]:
            for ratio in ratios:
                clusterMetric_path = './result/break_SCVI/%s/ClusteringMetrics_Batch%s_ratio%s.csv'(changed_property, batch, ratio)
                intermediate_clusterMetric= pd.read_csv(clusterMetric_path)
                clusterMetric = pd.concat([clusterMetric,intermediate_clusterMetric])
            #draw barplot

alg = ["Original","0.2","0.5","2","5"]
barplot_list(Modify_Prop1, alg, 'Clustering metrics train set for Changing Proportion', save='result/2019-05-01/trainset_clustering_metrics_for_PropThreshold1')
alg = ["Original","0.2","0.5","2","5"]
barplot_list(Modify_Prop2, alg, 'Clustering metrics train set for Changing Proportion', save='result/2019-05-01/trainset_clustering_metrics_for_PropThreshold2')
alg = ["Original","0.2","0.5","2","5"]
barplot_list(Modify_Prop3, alg, 'Clustering metrics train set for Changing Proportion', save='result/2019-05-01/trainset_clustering_metrics_for_PropThreshold3')

Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Library_Size/2019-04-05/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Library_Size/2019-04-14/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Expressed_Gene_Number/2019-04-14/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Gene_Expression_Proportion/2019-05-01/Pbmc_CellInfo_GeneCount*')
