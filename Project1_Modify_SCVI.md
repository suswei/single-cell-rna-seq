---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
<a id='section1'></a>
<h1><center>Project1_Modify_SCVI</center></h1>
<b><font size="+2">1. Introduction</font></b>
<a href='#section2'><p style="margin-left: 40px"><b><font size="+1">1.1 Deep Learning Network</font></b></p></a>
<a href='#section3'><p style="margin-left: 40px"><b><font size="+1">1.2 Invariance of Deep Learning</font></b></p></a>
<a href='#section4'><p style="margin-left: 40px"><b><font size="+1">1.3 SCVI</font></b></p></a>
<a href='#section5'><p style="margin-left: 40px"><b><font size="+1">1.4 Modified SCVI</font></b></p></a>

<b><font size="+2">2. Have Done and Discussion</font></b>
<a href='#section6'><p style="margin-left: 40px"><b><font size="+1">2.1 Break SCVI</font></b></p></a>
<p style="margin-left: 40px"><b><font size="+1">2.2 Tune Hyperparameter For MineNet</font></b></p>
<a href='#section7'><p style="margin-left: 60px"><b><font size="+1">2.2.1 step 1</font></b></p></a>
<a href='#section8'><p style="margin-left: 60px"><b><font size="+1">2.2.2 step 2</font></b></p></a>
<a href='#section9'><p style="margin-left: 40px"><b><font size="+1">2.3 Compare estimated mutual inforamtion with true mutual inforamtion</font></b></p></a>
<a href='#section9'><p style="margin-left: 60px"><b><font size="+1">2.3.1 gaussian gaussian</font></b></p></a>
<a href='#section10'><p style="margin-left: 60px"><b><font size="+1">2.3.2 gaussian categorical</font></b></p></a>
<a href='#section11'><p style="margin-left: 60px"><b><font size="+1">2.3.3 gaussian lognormal</font></b></p></a>


<a href='#section12'><b><font size="+2">3. To Does</font></b></a>

<a href='#section13'><b><font size="+2">4. Useful Tools and Information</font></b></a>

<br/><br/>
<br/><br/>
<!-- #endregion -->

<a id='section2'></a>
<b><font size="+1">1.1 Deep Learning Neural Network</font></b>

Tensor is a generalization of matrix with one obvious difference: a tensor is a mathematical entity that lives in a structure and interacts with other mathematical entities. If one transforms the other entities in the structure in a regular way, then the tensor must obey a related transformation rule.
Tensorflow vs Pytorch (which one is better for deep learning?)

Generally, to tune deep learning network:To find the best deep learning network, factors to consider are number of nodes in latent layer (width), number of hidden layers (depth), learning rate, optimizer, regularization, activation function according to [the book](https://www.deeplearningbook.org/) and the [deep learning specification](https://www.coursera.org/specializations/deep-learning?) on coursera. Write down more details here. Do we need to consider all the factors to tune the MINE deep learning network? Yes, eventually, we need to consider all the factors, with some clever strategy illustrated in [this paper](https://deepmind.com/blog/population-based-training-neural-networks/) 

<!-- #region -->
<a id='section3'></a>
<b><font size="+1">1.2 Invariance of Deep Learning</font></b>


[Measuring Invariances in Deep Networks](https://ai.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf)

Mutual Information

Rewrite the following, because it is copied from wikipedia:
Mutual information (MI) of two random variables is a measure of the mutual dependence between the two variables. Let <b><i>X</i></b> be a pair of random variables with values over the space $\mathcal{X}$ X $\mathcal{Y}$. If their joint distribution is \textbf{\textit{P}}_{(\textbf{\textit{X,Y}})}} and the marginal distributions are {\displaystyle P_{X}} P_X and {\displaystyle P_{Y}} {\displaystyle P_{Y}}, the mutual information is defined as
MI determines how similar the joint distribution of the pair <b>(<i>X,Y</i>)</b> is to the product of the marginal distributions of <b><i>X</i></b> and <b><i>Y</i></b>, and is more general than the correlation coefficient which only determines linear dependency. Mutual information contains information about all dependence—linear and nonlinear—and not just linear dependence as the correlation coefficient measures. 

According to the paper [mutual information neural estimation](https://arxiv.org/pdf/1801.04062.pdf), both deep neural network architecture and the sample size of input dataset are important to shrink the difference between the estimator and true value of the mutual information.

  Sample size: There is a method of calculating the required sample size recommended in the paper, understand the equation better.   
  
  To tune MINE, apart from the general factors for general deep learning network, in the paper, it is talked about that a naive application of stochastic gradient estimation leads to a biased estimate of the full batch gradient, the bias can be reduced by replacing the estimate in the demonimator by an exponential moving average, which will imporove all-around performance of MINE. Get more details about it.

The mutual information neural estimation is only for continuous variables? The mutual information neural estimation is compared with the true value and k-NN-based non-parametric estimator in the paper for multivariate gaussian random variabls.

  Between two continuous variable: we can compare the neural network estimator with the true value of mutual inforamtion for multivariate gaussian random variables. How to produce a positive definitive covariance matrix for the joint distribution of two multivariate gaussian. Why the method used in the original paper works.
  
  Between one continuous and one discrete variable: compare the neural network estimator with the nearest neighbour estimator [Mutual information between discrete and continuous data sets](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable).


<!-- #endregion -->

<a id='section4'></a>
<b><font size="+1">1.3 SCVI</font></b>

Single Cell RNA Seq

In single cell RNA seq, batch effect could come from when the sample is processed and when sequencing takes place, sequence platform, which individual the samples come from, how the sample is treated. Sometimes, biological difference can also be considered as batch effect based on different research aims. For example, if T cells from blood samples and T cells from other samples are combined together, and the research aim is to investigate the difference between norm T cells and abnormal T cells, we do not want the origin of the T cells to influence the result, here the origins of the T cells although are biological factors, it is considered as batch effect.

For a given dataset, how can we know there is batch difference at the beginning. The most easiest way is to do some exploratory analysis, like draw the tsne plot using SCVI, if the batch difference is big like from two different sequencing platform, it will be very easy to see. Or we can 


<a id='section5'></a>
<b><font size="+1">1.4 Modified SCVI</font></b>




<a id='section6'></a>
<b><font size="+1">2.1 Break SCVI</font></b>

<b><font size="3">2.1.1 Goal</font></b> 

<b><font size="3">2.1.2 Design</font></b>

<b><font size="3">2.1.3 Get Data</font></b>

<b><font size="3">2.1.4 Code</font></b>

<b><font size="3">2.1.5 Result and Lab Meeting Discussion</font></b>

<b><font size="3">2.1.5.1 Change Library Size</font></b>

<b><font size="3">2.1.5.2 Change number of genes with expression</font></b>

<b><font size="3">2.1.5.3 Change gene's expression proportion</font></b>

Because only 9 genes are expressed in all cells, genes that are expressed in 80% cells (58 genes), 60% cells (117 genes) and 50% cells (168 genes) are selected respectively. All these 58, 117 and 168 genes have median expression in the first 200 places. All the other unselected genes will remain unchanged. Among the 58 (117/168) genes, arrange their expression levels in descending order, increase the last 1/4 of the genes' expression proportion by a certain times in a specific batch so that the expression ratio for a certain gene in batch0:batch1 is 0.2, 0.5, 2, 5, and rescale the rest of the genes among the 58 (117/168) genes. For example, in order to make batch0:batch1 be 0.2, increase the last 1/4 genes' expression proportion by 5 times in batch 1. in order to make batch0:batch1 be 5, increase the last 1/4 genes' expression proportion by 5 times in batch 0.

The result is as follows:

```python
import os
%matplotlib inline
exec(open('code\\SummarizeResult.py').read())

file_paths = []
subfile_titles = []
thresholds = [1,2,3]

for i,threshold in enumerate(thresholds):
    file_paths = file_paths + ['result\\Break_SCVI\\2019-05-01\\trainset_clustering_metrics_for_PropThreshold%s.png'%(threshold)]
    subfile_titles = subfile_titles + ['Trainset clustering metrics changing expression proportions %s'%(threshold)]

SummarizeResult('image',file_paths,subfile_titles,'Fig1: Clustering metrics of train set by changing gene expression proportions')
```

Why when the ratio is 5, batch mixing entropy is even higher than that when the ratio is 2? Maybe it is because scvi is not stable, run different iterations.

<!-- #region -->
<a id='section7'></a>
<b><font size="+1">2.2.1 Tune Hyperparameter For MineNet: step1</font></b>

<b><font size="3">2.2.1.1 Goal</font></b> 

Try to find the best hyperparameters for the neural network of mutual information between latent vector z and batch vector s.

The hyperparameters refers to:

    -n_latent_z: number of nodes in each latent layer for the neural network of mutual information.

    -n_layers_z: number of layers for the neural network of mutual information.

    -MineLoss_Scale: the scale parameter for the mutual information.

<b><font size="3">2.2.1.2 Design</font></b>

Produce 20 combinations for the 3 hyperparameters:

    n_latent_z: [10, 30].

    n_layers_z: [3, 10].

    MineLoss_Scale: [1000, 5000, 10000, 50000, 100000].

The reason to choose 10, and 30 for n_latent_z is that the default value for n_latent_z in the VAE_MINE.py is 5, it is found that n_latent_z=5 has no difference from n_latent_z=1. Therefore, tuning value for n_latent_z starts from 10. For MineLoss_Scale, 1000, 5000, 10000, 50000, 100000 are chosen because reconstruction loss could be several thousands, while mutual information is smaller than 1. For each combination of the three hyperparameter, run scVI, and scVI+MINE on built-in pbmc dataset. The split ratio for training and testing set is 6:4.


<b><font size="3">2.2.1.3 Get Data</font></b>

pbmc dataset is built-in scVI dataset. just use the function PbmcDataset() in scvi.dataset to load the dataset

<b><font size="3">2.2.1.4 Code</font></b>

The code is in code/Tune_Hyperparameter_For_MineNet.py

<b><font size="3">2.2.1.5 Result and Lab Meeting Discussion</font></b>

The raw result of Tuning Hyperparameters for MineNet using 20 combinations is stored in ./result/Tune_Hyperparameter_For_MineNet/2019-05-01/, and the corresponding summaried result is shown here. Only n_latent_z=30, n_layers_z=3, MineLoss_Scale=[1000, 5000, 10000, 50000, 100000] is demonstrated in the summarized result because n_latent_z=30, n_layers_z=3 seems produce most typical result after reviewing all the results of 20 combinations.

<b><font size="3">2.2.1.5.1 Clustering metric result</font></b>
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
file_paths = []
subfile_titles = []
MineLoss_Scales = [1000,5000,10000,50000,100000]

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\trainset_clustering_metrics_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    subfile_titles = subfile_titles + ['Trainset n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

SummarizeResult('image',file_paths,subfile_titles,'Fig1: Clustering metrics of train set between scVI and scVI+MINE')
```

```python slideshow={"slide_type": "-"}
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\testset_clustering_metrics_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    subfile_titles = subfile_titles + ['Testset n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

SummarizeResult('image',file_paths,subfile_titles,'Fig2: Clustering metrics of test set between scVI and scVI+MINE')
```

From Fig1 and Fig2, it seems that for n_latent_z=30, n_layers_z=3, when MineLoss_Scale=1000, scVI+MINE is better than scVI with higher clustering metrics. However, the problem is that when all the scVI in the five subplots in Fig1 are compared, UCA and ARI in the subplot with MineLoss_Scale to be 5000 are much larger than those in other subplots. However, MineLoss_Scale difference should not influence the clustering results of scVI. 

One of the reasons for the discrepancy could be the training sample for scVI in each subplot is different. However, this reason is ruled out after the original code in scVI is checked out. Check the code file: D:/UMelb/PhD_Projects/Project1_Modify_SCVI/code/2019-05-01_Tune_Hyperparameter_For_MineNet.py, 
training and testing dataset is splitted by the UnsupervisedTrainer() function, which is defined in: D:/UMelb/PhD_Projects/Project1_Modify_SCVI/scvi/inference/inference.py.
The train_test() function in inference.py does the splitting, which takes in a seed with default value to be zero. UnsupervisedTrainer inherits trainer class. train_test() function is defined in D:/UMelb/PhD_Projects/Project1_Modify_SCVI/scvi/inference/trainer.py.  Check train_test() function. As the seed remains 0 every time, the training sample and testing sample for scVI each time are the same.

Another explanation is that scVI is not stable even when the training sample is the same. The reason could be the initialization of the weights in the networks, different initialization values could result in different optimization result. Or the minibatch of the training process, Or could be the reparametrization of z in the scVI method, according to the [tutorial_of_scVI](https://arxiv.org/abs/1606.05908). The paper titled [Deep_generative_modeling_for_single-cell_transcriptomics](https://www-nature-com.ezp.lib.unimelb.edu.au/articles/s41592-018-0229-2#Sec43) investigates the variability of scVI due to different initialization of weights in supplementary figure 1.d.

Now that scVI is not stable, scVI+MINE is also not stable, it is not safe to say that scVI+MINE is better than scVI for n_latent_z=30, n_layers_z=3 and MineLoss_Scale=1000 when only one training sample is run. Therefore, 100 monte carlo samples are necessary to get a more convincing conclusion. 


<b><font size="3">2.2.1.5.2 Batch and Cell Labels result of train set between scVI and scVI+MINE</font></b> 

For each row, left is scVI, right is scVI+MINE.

```python
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\trainset_tsne_SCVI_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\trainset_tsne_SCVI+MINE_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]

    subfile_titles = subfile_titles + ['scVI n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]
    subfile_titles = subfile_titles + ['scVI+MINE n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

SummarizeResult('image',file_paths,subfile_titles,'Fig4: Trainset tsne plot')
```

<b><font size="3">2.2.1.5.3 Batch and Cell Labels result of test set between scVI and scVI+MINE</font></b> 

For each row, left is scVI, right is scVI+MINE.

```python
file_paths = []
subfile_titles = []

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\testset_tsne_SCVI_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\testset_tsne_SCVI+MINE_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]

    subfile_titles = subfile_titles + ['scVI n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]
    subfile_titles = subfile_titles + ['scVI+MINE n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('code\\SummarizeResult.py').read())
SummarizeResult('image',file_paths,subfile_titles,'Fig5: Testset tsne plot')
```

From Fig1 and Fig2, when n_latent_z=30, n_layers_z=3, MineLoss_Scale=1000, no matter train set or test set, at least for this run, scVI+MINE is better than scVI according to the clustering metrics. However, scVI+MINE seems worse than scVI according to the tsne plots both in Fig4 for train set and Fig5 for test set. One reason could be tsne plots are not very accurate, after all it only shows two components of latent vector z's variation (check tsne's paper, and say this sentence in a professional way). The other reason could be that the clustering metrics used here are not appropriate. 
<font color=red>Which one is correct or is there other explanation???????? Maybe know about manifold learning clustering metrics could help?? Not using euclidean distance</font>


<b><font size="3">2.2.1.6 other topics discussed</font></b> 

tsne is applied mainly in computer vision, and is introduced into computation biology due to single cell RNA seq research. <font color=red>susan talked something more here, what is it?</font>

In reality, usually the batch effect in the dataset is unknown. The independence between batch and latent factor z is a more important problem than the independence between library size and latent factor z. <font color=red>Why???</font>. Heejung explained batch s is not included in z, but l is???. <font color=red>Make sure again.</font>

Is deep learning only appropriate when dimension of input is at least 70? The answer from susan is no. Actually linear
regression, logistic regression etc can be also considered as a type of simple deep learning neural network. 
<font color=red>But how to consider linear regresion or logistic regression as deep learning neural network?Be clear about the details</font>


<a id='section8'></a>
<b><font size="+1">2.2.2 Tune Hyperparameter For MineNet: step2</font></b>

<b><font size="3">2.2.2.1 Goal</font></b> 

Compare SCVI and SCVI+MINE from 100 monte carlo samples for 20 hyperparameter configurations on pbmc dataset, and retina dataset 
  
The hyperparameters refers to: 
 - n_latent_z: number of nodes in each latent layer for the neural network of mutual information.
 - n_layers_z: number of layers for the neural network of mutual information.
 - MineLoss_Scale: the scale parameter for the mutual information.

<b><font size="3">2.2.2.3 Design</font></b>
 
 - n_latent_z: [10, 30].
 - n_layers_z: [3, 10].
 - MineLoss_Scale: [1000, 5000, 10000, 50000, 100000]. 
     
Each time, pbmc(retina) dataset is randomly splitted into training set and test set at 6:4 ratio. For every random training set, apply SCVI once and SCVI+MINE with the 20 different hyperparameter configurations. Repeat the process 100 times. Get the averaged clustering metrics from the 100 iterations for SCVI and SCVI+MINE with the 20 different hyperparameter configurations, and compare which is better based on the averaged clustering metrics.

<b><font size="3">2.2.2.3 Get Data</font></b>

pbmc dataset is scVI built-in dataset. just use the function PbmcDataset() in scvi.dataset to load the dataset.
retina dataset is also scVI built-in dataset, just use the function RetinaDataset() in scvi.dataset to load the dataset.

<b><font size="3">2.2.2.4 Code</font></b>

The code is code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh. code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py. code/SummarizeResult.py The .sh file is for job submission in spartan system with the command: sh Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh.

<b><font size="3">2.2.2.5 Result and Lab Meeting Discussion</font></b>

The raw result is stored in ./result/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo/2019-05-26/ for pbmc, and in ./result/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo/2019-05-28/ for retina, and the corresponding summaried result is shown here.

```python
import itertools
Average_ClusteringMetric_Barplot(dataset_name = "Pbmc",results_dict = "result/Tune_Hyperparameter_For_MineNet/2019-05-26/", n_sample = 100, hyperparameter_config = {'n_hidden_z':[10,30],'n_layers_z':[3,10],'MineLoss_Scale':[1000,5000,10000,50000,100000]})

```

```python
hyperparameter_config = {
        'n_hidden_z': [10,30],
        'n_layers_z': [3,10],
        'MineLoss_Scale': [1000,5000,10000,50000,100000]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []
    
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    n_hidden_z = value[0]
    n_layers_z = value[1]
    MineLoss_Scale = value[2]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-26\\trainset_mean_clustering_metrics_Pbmc_Hidden%s_layers%s_MineLossScale%s_100samples.png'%(n_hidden_z,n_layers_z, MineLoss_Scale)]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-26\\testset_mean_clustering_metrics_Pbmc_Hidden%s_layers%s_MineLossScale%s_100samples.png'%(n_hidden_z,n_layers_z, MineLoss_Scale)]

SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig3: Pbmc Mean Clustering metrics from 100 samples")
```

Among the 20 hyperparameter configurations, n_hidden_z = 10, n_layers_z = 10, MineLoss_Scale = 1000 seems work best with smaller standard deviation.

ASW here is no longer lower than asw result in step 1 for Tune Hyperparameter For MineNet. In the last version of the step 2 result, only one configuration was run. Among the 100 samples for that configuration, there is running error for task id = 31, 39, 46, 47, 49, 55, 59,63,67,70,74,75. However, the main() function in the code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py file runs successfully on my own computer when I tried for taskid = [31], and taskid=[39]. At first, we guess it is because of the randomness from the stochastic gradient decent process which is due to the initialization of the network parameter, or the minibatch of the training process etc. Moreover, when I check the asw for taskid=[31] and [39] which are run on my own computer, it is much higher than the other 88 tasks run on the server. If I check the slurm*.out file for the 88 tasks, there is a warning message: DeprecationWarning: The linear_assignment function is deprecated in 0.21 and will be removed from 0.23. Use scipy.optimize.linear_sum_assignment instead. (linear_assignment is used in posterior.py) Maybe it could be the reason to produce errors for the 12 task ids not run, and lower asw for the left 88 taskids. 

To  solve the problem of inconsistency of versions of packages on my own computer and on the server, I create a virtualenv in the working directory on the server and installed the required packages of the same version on my computer listed in Project1_Modify_SCVI-requirements.txt file. Everytime when a task is run, the virtualenv will be activated. Check the code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh file. As for how to create a virtualenv, ref to the links [virtualenv](https://docs.python-guide.org/dev/virtualenvs/), [requirements](https://pip.readthedocs.io/en/1.1/requirements.html).

```python
import itertools
Average_ClusteringMetric_Barplot(dataset_name = "Retina",results_dict = "result/Tune_Hyperparameter_For_MineNet/2019-05-28/", n_sample = 100, hyperparameter_config = {'n_hidden_z':[10,30],'n_layers_z':[3,10],'MineLoss_Scale':[1000,5000,10000,50000,100000]})

```

```python
hyperparameter_config = {
        'n_hidden_z': [10,30],
        'n_layers_z': [3,10],
        'MineLoss_Scale': [1000,5000,10000,50000,100000]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []
    
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    n_hidden_z = value[0]
    n_layers_z = value[1]
    MineLoss_Scale = value[2]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-28\\trainset_mean_clustering_metrics_Retina_Hidden%s_layers%s_MineLossScale%s_100samples.png'%(n_hidden_z,n_layers_z, MineLoss_Scale)]
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-28\\testset_mean_clustering_metrics_Retina_Hidden%s_layers%s_MineLossScale%s_100samples.png'%(n_hidden_z,n_layers_z, MineLoss_Scale)]

SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig4: Retina Mean Clustering metrics from 100 samples")
```

According to Fig4, no matter what the configuration is, batch mixing entropy for SCVI+MINE is slightly lower than batch mixing entropy for SCVI, about 0.01-0.05 smaller. What does 0.01-0.05 smaller mean for batch mixing entropy. In order to answer this question, I need to check the range (from minimum to maximum value) of batch mixing entropy.


<a id='section9'></a>
<b><font size="+1">2.3 Compare estimated mutual information with true mutual inforamtion</font></b>

<b><font size="+1">2.3.1 gaussian and gaussian </font></b>

<b><font size="3">2.3.1.1 Goal</font></b> 

Try to reproduce the result in Figure 1 in the papar [Mutual Information Neural Estimator](https://arxiv.org/pdf/1801.04062.pdf), which compares the MineNet mutual inforamtion estimator with the true mutual information between two multivariate gaussian random variables.

<b><font size="3">2.3.1.2 Design</font></b>

Try four MineNet architectures which are Mine_Net, Mine_Net2, Mine_Net3, Mine_Net4, coded in ./scvi/models/modules.py, and compare estimator from each MineNet architecture with true mutual information. 

The main difference between Mine_Net3, Mine_Net4 and Mine_Net, Mine_Net2 is that MineNet, MineNet2 first transforms two random variables into two real number, and do linear combination and activation for the two real numbers. While, MineNet3, MineNet4 transforms two random variables into two vectors, then do linear combination and activation for the two vectors, finally convert the combined vector into a real number. The main difference between Mine_Net3 and Mine_Net4 is that Mine_Net3 transforms the two random variables into two vectors separately, while Mine_Net4 transforms the matrix containing the two random variables together to a combined vector.

The hyperparameter for Mine_Net:

    -n_hidden_z: [10],

    -n_layers_z: [10, 30, 50],

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameter for Mine_Net2:

    -n_hidden_z: [10],

    -n_layers_z: [10,

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameter for Mine_Net3:

    -H : [10] (H means the dimension of the vectors)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameter for Mine_Net4:

    -layers = [32, 16] (the node number of the two hidden layers which converts the random variables matrix into a vector)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


<b><font size="3">2.3.1.3 Code</font></b>

code/compare_gaussian_gaussian_estimatedMI_with_trueMI.py produces result to compare estimator from Mine_Net with true mutual inforamtion. code/compare_gaussian_gaussian_estimatedMI_with_trueMI2.py produces result to compare estimator from Mine_Net2, Mine_Net3, Mine_Net4 with true mutual inforamtion. code/SummarizeResult.py to summarize the final result.

<b><font size="3">2.3.1.4 Result and Lab Meeting Discussion</font></b>

The raw result is stored in ./result/compare_estimatedMI_with_trueMI/gaussian_gaussian/, and the corresponding summarized result is shown here. Figure5 is for Mine_Net, Figure6 is for Mine_Net2, Figure7 is for Mine_Net3, Figure8 is for Mine_Net4. For Mine_Net4, when Gaussian_Dimension=20, rho = -0.99, -0.9, 0.9, 0.99, the estimated mutual inforamtion returned by Mine_Net4 is NaN. Why?: it is because the learning rate is too large for the situation when Gaussian_Dimension=20, rho= -0.99, -0.9, 0.9, 0.99, the loss for mini batches becomes NaN. when learning rate is decreased from 0.0005 to 0.00005, no NaN value is returned. torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation measurements.

```python
import os
%matplotlib inline
exec(open('code\\SummarizeResult.py').read())
Summarize_Compare_NNEstimator_TrueMI(results_dict='./result/compare_estimatedMI_with_trueMI/gaussian_gaussian/', MineNet_Info = {'model':['Mine_Net'], 'Hyperparameter':{'n_hidden_z': [10], 'n_layers_z':[10,30,50], 'Gaussian_Dimension': [2, 20], 'sample_size': [14388], 'train_size': [0.5]}})
Summarize_Compare_NNEstimator_TrueMI(results_dict='./result/compare_estimatedMI_with_trueMI/gaussian_gaussian/', MineNet_Info = {'model':['Mine_Net2'], 'Hyperparameter':{'Gaussian_Dimension': [2, 20], 'sample_size': [14388], 'train_size': [0.5]}})
Summarize_Compare_NNEstimator_TrueMI(results_dict='./result/compare_estimatedMI_with_trueMI/gaussian_gaussian/', MineNet_Info = {'model':['Mine_Net3'], 'Hyperparameter':{'Gaussian_Dimension': [2, 20], 'sample_size': [14388], 'train_size': [0.5]}})
Summarize_Compare_NNEstimator_TrueMI(results_dict='./result/compare_estimatedMI_with_trueMI/gaussian_gaussian/', MineNet_Info = {'model':['Mine_Net4'], 'Hyperparameter':{'Gaussian_Dimension': [2, 20], 'sample_size': [14388], 'train_size': [0.5]}})

```

```python
import math
hyperparameter_config = {
        'n_hidden_z': [10],
        'n_layers_z': [10, 30, 50],
        'Gaussian_Dimension': [2, 20],
        'sample_size': [14388],
        'train_size': [0.5]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []
    
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    n_hidden_z = value[0]
    n_layers_z = value[1]
    Gaussian_Dimension = value[2]
    sample_size = value[3]
    train_size = value[4]
    file_paths = file_paths + ['result\\compare_estimatedMI_with_trueMI\\gaussian_gaussian\\Mine_Net_n_hidden_z%s_n_layers_z%s_variable_dimension%s_sample_size%s_training.png'%(n_hidden_z,n_layers_z, Gaussian_Dimension, math.floor(sample_size*train_size))]
    file_paths = file_paths + ['result\\compare_estimatedMI_with_trueMI\\gaussian_gaussian\\Mine_Net_n_hidden_z%s_n_layers_z%s_variable_dimension%s_sample_size%s_testing.png'%(n_hidden_z,n_layers_z, Gaussian_Dimension, math.floor(sample_size*(1-train_size)))]

SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig5: Compare Mine_Net estimator with true MI")
```

```python
hyperparameter_config = {
        'Net_Name' : ['Mine_Net2', 'Mine_Net3', 'Mine_Net4'],
        'Gaussian_Dimension' : [2, 20],
        'sample_size' : [14388],
        'train_size': [0.5]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []
    
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    Net_Name = value[0]
    Gaussian_Dimension = value[1]
    sample_size = value[2]
    train_size = value[3]
    file_paths = file_paths + ['result\\compare_estimatedMI_with_trueMI\\gaussian_gaussian\\%s_variable_dimension%s_sample_size%s_training.png'%(Net_Name, Gaussian_Dimension, math.floor(sample_size*train_size))]
    file_paths = file_paths + ['result\\compare_estimatedMI_with_trueMI\\gaussian_gaussian\\%s_variable_dimension%s_sample_size%s_testing.png'%(Net_Name, Gaussian_Dimension, math.floor(sample_size*(1-train_size)))]

SummarizeResult(result_type='image', file_paths=file_paths[0:4], figtitle="Fig6: Compare Mine_Net2 estimated MI with true MI for gaussian and gaussian")
SummarizeResult(result_type='image', file_paths=file_paths[4:8], figtitle="Fig7: Compare Mine_Net3 estimated MI with true MI for gaussian and gaussian")
SummarizeResult(result_type='image', file_paths=file_paths[8:], figtitle="Fig8: Compare Mine_Net4 estimated MI with true MI for gaussian and gaussian")
```

<a id='section10'></a>
<b><font size="+1">2.3.2 gaussian and categorical </font></b>

<b><font size="3">2.3.2.1 Goal</font></b> 

Use the nearest neighbor method in the papar [Mutual Information between Discrete and Continuous
Data Sets](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable) to estimate mutual inforamtion between continuous gaussian variable and categorical variable. 

<b><font size="3">2.3.2.2 Design</font></b>

One dimension categorical variable with 3 categories, and one dimension gaussian variables.

One dimension categorical variable with 3 categories, and 4 dimension gaussian variables. (The reason to choose dimension 4 for gaussian variable is because the multivariate dimension integration in matlab is time-demanding when dimension is larger than 4)

For both cases above, choose 13 different combinations of categorical variable and gaussian variable by determining the probability distribution for the categorical variable, and the gaussian distribution conditional on the categorical variable. For each combination, calculate the true mutual information according to the definition of mutual information. Then get the estimated mutual information. In order to get estimated mutual information for each combination, get 10000 samples from the joint distribution of the categorical and gaussian variable, use nearest neighbor method to estimate mutual information, repeat this process 100 times, get the average of the 100 estimated mutual information as the final estimated mutual inforamtion for this combination, and compare it with its true mutual information. Repeat the whole process for the left 12 combinations, and plot the true mutual information, averaged estimated mutual information and standard deviation of the estimated mutual information.

<b><font size="3">2.3.2.3 Code</font></b>

code/compare_gaussian_categorical_estimatedMI_with_trueMI.m, code/discrete_continuous_info.m, code/discrete_continuous_info_fast.m, code/integralN.m, produce the results. code/SummarizeResult.py displays the results.

<b><font size="3">2.3.2.4 Result and Lab Meeting Discussion</font></b>

The result is stored in ./result/compare_estimatedMI_with_trueMI/gaussian_categorical/.

```python
exec(open('code\\SummarizeResult.py').read())
hyperparameter_config = {
        'iterations' : [100],
        'samples' : [10000],
        'discrete_dim' : [1],
        'gaussian_dim': [1,4]
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []
    
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    iteration = value[0]
    sample_size = value[1]
    discrete_dim = value[2]
    gaussian_dimension = value[3]
    file_paths = file_paths + ['result\\compare_estimatedMI_with_trueMI\\gaussian_categorical\\%siterations_%ssamples_discrete%s_gaussian%s.png'%(iteration, sample_size, discrete_dim, gaussian_dimension)]
    
SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig9: Compare estimated MI with true MI for gaussian and categorical")
```

<a id='section12'></a>
<b><font size="+2">3. To Does</font></b>

<b><font size="+1">3.1 Tune Hyperparameter For MineNet</font></b>

Carry out the monte carlo experiment for all the 20 configurations, after that, we can think of improving the architecture of the neural network for MineNet including the activation function etc.

Try to apply the adversarial network architecture between SCVI and MINE

<b><font size="+1">3.2 Difference between the estimated lower bound and true mutual information</font></b>

Deep neural network is used to estimate the lower bound for mutual information between two continuous random variables. But the question is whether the deep neural network can really estimate the lower bound precisely and how close is the lower bound to the true mutual information. We can minimize the mutual information by minimizing the lower bound only when the lower bound is very close to the true mutual inforamtion.

Also check how the mutual inforamtion network estimator works between a categorical random variable and continuous random variable.

<b><font size="+1">3.3 Datasets</font></b>

Find real data set with obvious batch effect. Because the pbmc dataset used now do not have obvious batch effect. The dataset  ,harmonizing datasets with different composition of cell types,used in the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1) could be a resource. And another thing is to simulate data set with obvious batch effect from ZINB model.


<a id='section13'></a>
<b><font size="+2">4. Useful Tools and Information</font></b>

<b><font size="+1">4.1 Version control for jupyter notebook</font></b>

[how-to-version-control-jupyter](https://nextjournal.com/schmudde/how-to-version-control-jupyter)

[jupytext](https://github.com/mwouts/jupytext/blob/master/README.md)

```python

```
