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

<a id='section1'></a>
<h1><center>Project1_Modify_SCVI</center></h1>
<b><font size="+2">1. Introduction</font></b>
<a href='#section2'><p style="margin-left: 40px"><b><font size="+1">1.1 Single Cell RNA Seq</font></b></p></a>
<a href='#section3'><p style="margin-left: 40px"><b><font size="+1">1.2 Deep Learning Neural Network</font></b></p></a>
<a href='#section4'><p style="margin-left: 40px"><b><font size="+1">1.3 SCVI</font></b></p></a>
<a href='#section5'><p style="margin-left: 40px"><b><font size="+1">1.4 Problems</font></b></p></a>
<b><font size="+2">2. Have Done and Discussion</font></b>
<a href='#section6'><p style="margin-left: 40px"><b><font size="+1">2.1 Break SCVI</font></b></p></a>
<p style="margin-left: 40px"><b><font size="+1">2.2 Tune Hyperparameter For MineNet</font></b></p>
<a href='#section7'><p style="margin-left: 60px"><b><font size="+1">2.2.1 step 1</font></b></p></a>
<a href='#section8'><p style="margin-left: 60px"><b><font size="+1">2.2.2 step 2</font></b></p></a>

<a href='#section9'><b><font size="+2">3. To Does</font></b></a>
<br/><br/>
<br/><br/>


<a id='section2'></a>


<a id='section3'></a>


<a id='section4'></a>


<a id='section5'></a>


<a id='section6'></a>
<b><font size="+1">2.1 Break SCVI</font></b>

<b><font size="3">2.1.1 Goal</font></b> 

<b><font size="3">2.1.2 Design</font></b>

<b><font size="3">2.1.3 Get Data</font></b>

<b><font size="3">2.1.4 Code</font></b>

<b><font size="3">2.1.5 Result and Lab Meeting Discussion</font></b>

<b><font size="3">2.1.5.1 Change Library Size</font></b>


<a id='section7'></a>
<b><font size="+1">2.2.1 Tune Hyperparameter For MineNet: step1</font></b>

<b><font size="3">2.2.1.1 Goal</font></b> 

Try to find the best hyperparameters for the neural network of mutual information between latent vector z and batch vector s.   
  
The hyperparameters refers to: 
 - n_latent_z: number of nodes in each latent layer for the neural network of mutual information.
 - n_layers_z: number of layers for the neural network of mutual information.
 - MineLoss_Scale: the scale parameter for the mutual information.

<b><font size="3">2.2.1.2 Design</font></b>

Produce 20 combinations for the 3 hyperparameters: 
 - n_latent_z: [10, 30].
 - n_layers_z: [3, 10].
 - MineLoss_Scale: [1000, 5000, 10000, 50000, 100000]. 
     
The reason to choose 10, and 30 for n_latent_z is that the default value for n_latent_z in the VAE_MINE.py is 5, it is found   that n_latent_z=5 has no difference from n_latent_z=1. Therefore, tuning value for n_latent_z starts from 10. For MineLoss_Scale, 1000, 5000, 10000, 50000, 100000 are chosen because reconstruction loss could be several thousands, while mutual information is smaller than 1. For each combination of the three hyperparameter, run scVI, and scVI+MINE on built-in pbmc dataset. The split ratio for training and testing set is 6:4.

<b><font size="3">2.2.1.3 Get Data</font></b>

pbmc dataset is built-in scVI dataset. just use the function PbmcDataset() in scvi.dataset to load the dataset

<b><font size="3">2.2.1.4 Code</font></b>

The code is in code/Tune_Hyperparameter_For_MineNet.py

<b><font size="3">2.2.1.5 Result and Lab Meeting Discussion</font></b>

The raw result of Tuning Hyperparameters for MineNet using 20 combinations is stored in ./result/Tune_Hyperparameter_For_MineNet/2019-05-01/, and the corresponding summaried result is shown here.
Only n_latent_z=30, n_layers_z=3, MineLoss_Scale=[1000, 5000, 10000, 50000, 100000] is demonstrated in the summarized result
because n_latent_z=30, n_layers_z=3 seems produce most typical result after reviewing all the results of 20 combinations.

<b><font size="3">2.2.1.5.1 Clustering metric result</font></b>

```python slideshow={"slide_type": "-"}
import os
%matplotlib inline

file_paths = []
subfile_titles = []
MineLoss_Scales = [1000,5000,10000,50000,100000]

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-01\\trainset_clustering_metrics_pbmc_Hidden30_layers3_MineLossScale%s.png'%(MineLoss_Scale)]
    subfile_titles = subfile_titles + ['Trainset n_latent_z:30 n_layers_z:3 MineLoss_Scale:%s'%(MineLoss_Scales[i])]

exec(open('code\\SummarizeResult.py').read())
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

Compare SCVI and SCVI+MINE from 100 monte carlo samples for one hyperparameter configuration: n_latent_z:30, n_layers_z:3, MineLoss_Scale: 1000.   
  
The hyperparameters refers to: 
 - n_latent_z: number of nodes in each latent layer for the neural network of mutual information.
 - n_layers_z: number of layers for the neural network of mutual information.
 - MineLoss_Scale: the scale parameter for the mutual information.

<b><font size="3">2.2.2.3 Design</font></b>
 
 - n_latent_z: 30.
 - n_layers_z: 3.
 - MineLoss_Scale: 1000. 
     
The reason to choose only one configuration among the 20 is just that computing facility is slow and limited. Before faster and more abundant computing resources are available, only one configuration is carried out for the monte carlo simulation. n_latent_z:30, n_layers_z:3, MineLoss_Scale:1000 seems to be the configuration at which SCVI+MINE works better than SCVI more obviously based on raw results in step 1 of Tune Hyperparameter For MineNet. Therefore, it is chosen for the monte carlo experiment. Each time, pbmc dataset is randomly splitted into training set and test set at 6:4 ratio. For every random training set, use both SCVI and SCVI+MINE. Repeat the process 100 times. Get the averaged clustering metrics from the 100 iterations for both SCVI and SCVI+MINE, and compare which is better based on the averaged clustering metrics.

<b><font size="3">2.2.2.3 Get Data</font></b>

pbmc dataset is built-in scVI dataset. just use the function PbmcDataset() in scvi.dataset to load the dataset

<b><font size="3">2.2.2.4 Code</font></b>

The code is in code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py, and code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh. The .sh file is for job submission in spartan system with the command: sbatch Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh

<b><font size="3">2.2.2.5 Result and Lab Meeting Discussion</font></b>

The raw result is stored in ./result/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo/2019-05-08/, and the corresponding summaried result is shown here.

```python
Average_ClusteringMetric_Barplot(dataset_name = "Pbmc", file_number = 100, n_hidden_z = 30, n_layers_z =3, MineLossScale = 1000)

file_paths = []
for i,set in enumerate(['trainset','testset']):
    file_paths = file_paths + ['result\\Tune_Hyperparameter_For_MineNet\\2019-05-08\\%s_mean_clustering_metrics_Pbmc_Hidden30_layers3_MineLossScale1000_100samples.png'%(set)]

SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig3: Mean Clustering metrics from 100 samples")
```

There is not much difference for the 4 clustering metrics. One phenomenon is that the ASW is much lower than asw result in step 1 for Tune Hyperparameter For MineNet. At first, it was guessed that it is because of large standard error. However, later, after standard error is added, the large standard error explanation can be ruled out. <font color=red>But is there other explanation???</font>.

Although in the code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.sh file, I set the task id from 0-99 which means 100 monte carlos, however, when the final task is finished, there is running error for task id = 31, 39, 46, 47, 49, 55, 59,63,67,70,74,75. The error is that there are NaN values for penalty:tensor([nan, nan,....,nan]). This means the upper Fig3 only shows the average cluster metrics of 88 monte carlos. In order to figure out the reason for the errors, I run the main() function in the code/Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py file on my own computer for taskid = [31], and taskid=[39]. However, it runs successfully without any error. Therefore, we guess it is because of the randomness from the stochastic gradient decent process which is due to the initialization of the network parameter, or the minibatch of the training process etc, and the optimization process is not finished successfully. <font color=red>Find supporting information online???</font>. 


<a id='section9'></a>
<b><font size="+2">3. To Does</font></b>

<b><font size="+1">3.1 Tune Hyperparameter For MineNet</font></b>

The ASW for the n_latent_z=30, n_layers_z=3, MineLossScale=1000 in step 2 result for Tune Hyperparameter For MineNet is much lower than ASW in step 1 result. <font color=red>why???</font>.  Choose n_latent_z=30, n_layers_z=3, MineLossScale=1000 based on result in step 1 may be is not very reliable, and we can hardly make any conclusion from result in step 2. Therefore, we decide to run 100 monte carlos for the 20 configurations.

After the monte carlo experiment for the 20 configurations, we can think of improving the architecture of the neural network for MineNet including the activation function etc. 

<b><font size="+1">3.2 Datasets</font></b>

Find real data set with obvious batch effect. Because the pbmc dataset used now do not have obvious batch effect. The dataset  ,harmonizing datasets with different composition of cell types,used in the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1) could be a resource. And another thing is to simulate data set with obvious batch effect from ZINB model.

```python

```
