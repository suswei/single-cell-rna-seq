# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 1. Overview
#   
# ## 1.1 Goal
# Try to find the best hyperparameters for the neural network of mutual information between latent vector z and batch vector s.   
#   
# The hyperparameters refers to: 
#  - n_latent_z: number of nodes in each latent layer for the neural network of mutual information.
#  - n_layers_z: number of layers for the neural network of mutual information.
#  - MineLoss_Scale: the scale parameter for the mutual information.
#
# ## 1.2 Design
# Produce 20 combinations for the 3 hyperparameters: 
#  - n_latent_z: [10, 30].
#  - n_layers_z: [3, 10].
#  - MineLoss_Scale: [1000, 5000, 10000, 50000, 100000]. 
#      
# The reason to choose 10, and 30 for n_latent_z is that the default value for n_latent_z in the VAE_MINE.py is 5, it is found   that n_latent_z=5 has no difference from n_latent_z=1. Therefore, tuning value for n_latent_z starts from 10. For MineLoss_Scale, 1000, 5000, 10000, 50000, 100000 are chosen because reconstruction loss could be several thousands, while mutual information is smaller than 1. For each combination of the three hyperparameter, run scVI, and scVI+MINE on built-in pbmc dataset. The split ratio for training and testing set is 6:4.

# # 2. Get Data
# pbmc dataset is built-in scVI dataset. just use the function PbmcDataset() in scvi.dataset to load the dataset

# # 3. Code
# The code is in code/Tune_Hyperparameter_For_MineNet.py

# # 4. Result and Lab Meeting Discussion
# The raw result of Tuning Hyperparameters for MineNet using 20 combinations is stored in D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/2019-05-01/, and the corresponding summaried result is shown here.
# Only n_latent_z=30, n_layers_z=3, MineLoss_Scale=[1000, 5000, 10000, 50000, 100000] is demonstrated in the summrized result
# because n_latent_z=30, n_layers_z=3 seems produce most typical result after reviewing all the results of 20 combinations.
#
# ## 4.1 Clustering metric results

# + {"slideshow": {"slide_type": "-"}}
import os
# %matplotlib inline

file_paths = []
subfile_titles = []
MineLoss_Scales = [1000,5000,10000,50000,100000]

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\trainset_clustering_metrics_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    subfile_titles = subfile_titles + ['Trainset n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\code\\SummarizeResult.py').read())
SummarizeResult('image',file_paths,subfile_titles,'Fig1: Clustering metrics of train set between scVI and scVI+MINE')

# + {"slideshow": {"slide_type": "-"}}
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\testset_clustering_metrics_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    subfile_titles = subfile_titles + ['Testset n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\code\\SummarizeResult.py').read())
SummarizeResult('image',file_paths,subfile_titles,'Fig2: Clustering metrics of test set between scVI and scVI+MINE')
# -

# From Fig1 and Fig2, it seems that for n_latent_z=30, n_layers_z=3, when MineLoss_Scale=1000, scVI+MINE is better than scVI with higher clustering metrics. However, the problem is that when all the scVI in the five subplots in Fig1 are compared, UCA and ARI in the subplot with MineLoss_Scale to be 5000 are much larger than those in other subplots. However, MineLoss_Scale difference should not influence the clustering results of scVI. 
#
# One of the reasons for the discrepancy could be the training sample for scVI in each subplot is different. However, this reason is ruled out after the original code in scVI is checked out. Check the code file: D:/UMelb/PhD_Projects/Project1_Modify_SCVI/code/2019-05-01_Tune_Hyperparameter_For_MineNet.py, 
# training and testing dataset is splitted by the UnsupervisedTrainer() function, which is defined in: D:/UMelb/PhD_Projects/Project1_Modify_SCVI/scvi/inference/inference.py.
# The train_test() function in inference.py does the splitting, which takes in a seed with default value to be zero. UnsupervisedTrainer inherits trainer class. train_test() function is defined in D:/UMelb/PhD_Projects/Project1_Modify_SCVI/scvi/inference/trainer.py.  Check train_test() function. As the seed remains 0 every time, the training sample and testing sample for scVI each time are the same.
#
# Another explanation is that scVI is not stable even when the training sample is the same. The reason could be the initialization of the weights in the networks, different initialization values could result in different optimization result. Or could be the reparametrization of z in the scVI method, according to the [tutorial_of_scVI](https://arxiv.org/abs/1606.05908). The paper titled [Deep_generative_modeling_for_single-cell_transcriptomics](https://www-nature-com.ezp.lib.unimelb.edu.au/articles/s41592-018-0229-2#Sec43) investigates the variability of scVI due to different initialization of weights in supplementary figure 1.d.
#
# Now that scVI is not stable, scVI+MINE is also not stable, it is not safe to say that scVI+MINE is better than scVI for n_latent_z=30, n_layers_z=3 and MineLoss_Scale=1000 when only one training sample is run. Therefore, 100 monte carlo samples are necessary to get a more convincing conclusion. 

# ## 4.2 Batch and Cell Labels result of train set between scVI and scVI+MINE
# For each row, left is scVI, right is scVI+MINE.

# +
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\trainset_tsne_SCVI_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\trainset_tsne_SCVI+MINE_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]

    subfile_titles = subfile_titles + ['scVI n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]
    subfile_titles = subfile_titles + ['scVI+MINE n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\code\\SummarizeResult.py').read())
SummarizeResult('image',file_paths,subfile_titles,'Fig3: Trainset tsne plot')
# -

# ## 4.3 Batch and Cell Labels result of test set between scVI and scVI+MINE 
# For each row, left is scVI, right is scVI+MINE.

# +
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\testset_tsne_SCVI_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    file_paths = file_paths + ['D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\result\\2019-05-01\\testset_tsne_SCVI+MINE_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]

    subfile_titles = subfile_titles + ['scVI n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]
    subfile_titles = subfile_titles + ['scVI+MINE n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('D:\\UMelb\\PhD_Projects\\Project1_Modify_SCVI\\code\\SummarizeResult.py').read())
SummarizeResult('image',file_paths,subfile_titles,'Fig4: Testset tsne plot')
# -

# From Fig1 and Fig2, when n_latent_z=30, n_layers_z=3, MineLoss_Scale=1000, no matter train set or test set, at least for this run, scVI+MINE is better than scVI according to the clustering metrics. However, scVI+MINE seems worse than scVI according to the tsne plots both in Fig3 for train set and Fig4 for test set. One reason could be tsne plots are not very accurate, after all it only shows two components of latent vector z's variation (check tsne's paper, and say this sentence in a professional way). The other reason could be that the clustering metrics used here are not appropriate. 
# <font color=red>Which one is correct or is there other explanation???????? Maybe know about manifold learning clustering metrics could help?? Not using euclidean distance</font>

# ## 4.4 tsne 
# tsne is applied mainly in computer vision, and is introduced into computation biology due to single cell RNA seq research. <font color=red>susan talked something more here, what is it?</font>

# ## 4.5 
# In reality, usually the batch effect in the dataset is unknown. The independence between batch and latent factor z is a more important problem than the independence between library size and latent factor z. <font color=red>Why???</font>. Heejung explained batch s is not included in z, but l is???. <font color=red>Make sure again.</font>

# ## 4.6 
# Is deep learning only appropriate when dimension of input is at least 70? The answer from susan is no. Actually linear
# regression, logistic regression etc can be also considered as a type of simple deep learning neural network. 
# <font color=red>But how to consider linear regresion or logistic regression as deep learning neural network?Be clear about the details</font>

# # 5.To Dos
# ## 5.1 
# Establish a system so that everything about the project Project1_Modify_SCVI, including the background, the result, the code, the data, the lab meeting content, the references can be versioned controlled and shared, and can be accessed after many years later. Workflowr is a very good package to realize this, but only in R language. Jupyter Notebooks are not easy for github version control, although it can use python. Fortunately, there are some ways to overcome the problem of version control in Jupyter Notebook, please refer to this link: 
# [How_to_Version_Control_Jupyter_Notebooks](https://nextjournal.com/schmudde/how-to-version-control-jupyter)
#
# ## 5.2
# Because SCVI is not stable even when the sample of Pbmc_dataset fed into the neural network of SCVI remains constant, get 100 Monte Carlo samples. For each sample, run both SCVI and SCVI+MINE, get the average clustering metrics for the 100 Monte Carlos for both SCVI and SCVI+MINE, and then compare. In the raw result of tuning hyperparameters for MineNet (stored in D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/2019-05-01/), 20 grid points are run for the three hyperparameters. But here only one combination is chosen, because of the large computation time. n_latent_z=30, n_layers_z=3, MineLoss_Scale=1000 is chosen because it seems to work best among all the other combinations according to result of 2019-05-06(a link should be added here). 
#
# ## 5.3
# Find real data set with obvious batch effect. Because the pbmc dataset used now do not have obvious batch effect. The dataset  ,harmonizing datasets with different composition of cell types,used in the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1) could be a resource. And another thing is to simulate data set with obvious batch effect from ZINB model.


