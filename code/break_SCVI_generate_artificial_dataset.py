import os
from scvi.dataset import *
import numpy as np
import pandas as pd
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import rpy2.robjects as robjects
from scipy import sparse
import glob2
import ntpath

#%matplotlib inline

if not os.path.exists('data/break_SCVI/Change_Library_Size'):
        os.makedirs('data/break_SCVI/Change_Library_Size')
if not os.path.exists('data/break_SCVI/Change_Expressed_Gene_Number'):
        os.makedirs('data/break_SCVI/Change_Expressed_Gene_Number')
if not os.path.exists('data/break_SCVI/Change_Gene_Expression_Proportion'):
        os.makedirs('data/break_SCVI/Change_Gene_Expression_Proportion')

if not os.path.exists('result/break_SCVI/Change_Library_Size'):
        os.makedirs('result/break_SCVI/Change_Library_Size')
if not os.path.exists('result/break_SCVI/Change_Expressed_Gene_Number'):
        os.makedirs('result/break_SCVI/Change_Expressed_Gene_Number')
if not os.path.exists('result/break_SCVI/Change_Gene_Expression_Proportion'):
        os.makedirs('result/break_SCVI/Change_Gene_Expression_Proportion')

#use the built in pbmc_dataset to produce Pbmc_batch.csv, Pbmc_CellName_Label_GeneCount.csv
pbmc_dataset = PbmcDataset(save_path='data/break_SCVI')
Pbmc_Genecount = pd.DataFrame(pbmc_dataset._X.todense())
Pbmc_Label = pd.DataFrame({'Labels':pbmc_dataset.labels[:,0]})
Pbmc_CellName = pd.DataFrame({'PbmcCellName':list(pbmc_dataset.raw_qc.index)})
Pbmc_CellName_Label_Genecount = pd.concat([Pbmc_CellName, Pbmc_Label, Pbmc_Genecount], axis=1)
Pbmc_CellName_Label_Genecount.to_csv('./data/break_SCVI/Pbmc_CellName_Label_GeneCount.csv', index = None, header=True)
Pbmc_Batch = pd.DataFrame({'batchpbmc4k':pbmc_dataset.design.iloc[:,1]})
Pbmc_Batch.to_csv('./data/break_SCVI/Pbmc_batch.csv', index = None, header=True)

# use break_SCVI.R to produce Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv
# and produce artificial dataset with larger batch effect from PBMC
r_source = robjects.r['source']
r_source('./code/break_SCVI.R')
print('r script finished running')
