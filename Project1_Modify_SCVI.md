---
jupyter:
  jupytext:
    formats: md,ipynb
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
<a href='#section2'><p style="margin-left: 40px"><b><font size="+1">1.1 The Invariance Problem in single-cell RNA seq analysis</font></b></p></a>

<p style="margin-left: 60px">Batch effects and library size effects both included?</p>

<a href='#section3'><p style="margin-left: 40px"><b><font size="+1">1.2 Existing methods to deal with the invariance problem in single-cell RNA seq problem</font></b></p></a>

<p style="margin-left: 60px">Generalized linear model methods and other neural network</p>

<p style="margin-left: 60px">scVI is superior than previous methods. Mention that at the same time of our research, another network named saucie was proposed, which is better than scVI. But our MineNet can also be combined with saucie</p>

<a href='#section4'><p style="margin-left: 40px"><b><font size="+1">1.3 scVI</font></b></p></a>

<p style="margin-left: 60px">Introduction of scVI: from autoencoder, to variational autoencoder, to scVI</p>

<p style="margin-left: 60px">Although superior, there is still remaining problem of scVI in dealing with the invariance problem in single-cell RNA seq analysis.</p>

<a href='#section5'><p style="margin-left: 40px"><b><font size="+1">1.4 Use dependency between latent vectors and batch as the loss penalty to extend scVI in dealing with the batch effects</font></b></p></a>

<p style="margin-left: 60px">Different ways to get the dependency measures: explain MMD and mutual information</p>

<p style="margin-left: 60px">MMD has been used together with scVI, HISC (scVI penalized by MMD), saucie network penalized by MMD. The problem of using MMD as the penalty is the high dimensional problem for non-parametric kernel estimator.</p>

<a href='#section6'><p style="margin-left: 40px"><b><font size="+1">1.5 Use Mutual information as the penalty</font></b></p></a>

<p style="margin-left: 60px">Explain ways to calculate mutual information, Explain the theory and algorithms of MineNet to estimate mutual information</p>

<p style="margin-left: 60px">Explain how to calculate MI when combined with scVI: not using latent vector and batch as two variables</p>

<a href='#section7'><p style="margin-left: 40px"><b><font size="+1">1.6 How to build MineNet and train MI_penalized scVI</font></b></p></a>

<p style="margin-left: 60px">Explain basics for deep learning: why updating weights in the direction of gradient? how to establish the architecture and choose hyperparameters for MineNet: including the lr, initialization, activation, drop out, batch_normalization, optimizer. loss function for MI_penalized scVI</p>

<p style="margin-left: 60px">Multi-task optimization and Pareto front (convergence issues for pareto front? check papers)</p>

<b><font size="+2">2. Have Done</font></b>
<a href='#section8'><p style="margin-left: 40px"><b><font size="+1">2.1 Batch effects remained partially after scVI treatment of real datasets, and artificial datasets with a large batch effects to demonstrate that there is room to extend scVI for batch effect removal.</font></b></p></a>

<a href='#section9'><p style="margin-left: 40px"><b><font size="+1">2.2 Compare estimated mutual inforamtion with true mutual inforamtion to demonstrate that MineNet can get an appropriate estimator for mutual inforamtion. This result is not necessary to be presented as they already exist in the original research paper.</font></b></p></a>

<p style="margin-left: 60px">Do I also need to compare MMD estimator with MineNet estimator as the penalty? Maybe later</p>

<a href='#section10'><p style="margin-left: 40px"><b><font size="+1">2.3 scVI+MI_Penalty</font></b></p></a>

<a href='#section11'><p style="margin-left: 40px"><b><font size="+1">2.4 automate the process for finding the hyperparameters of MI_penalized SCVI for different datasets </font></b></p></a>

<a href='#section12'><p style="margin-left: 40px"><b><font size="+1">2.5 scVI and scVI+MI_Penalty comparison for downstream analysis</font></b></p></a>

<a href='#section13'><b><font size="+2">3. Discussion</font></b></a>

<a href='#section14'><b><font size="+2">4. Reference</font></b></a>

<a href='#section15'><b><font size="+2">5. To Does</font></b></a>

<b><font size="+2">6. Useful Tools, Information, and Knowledge</font></b>
<a href='#section16'><p style="margin-left: 40px"><b><font size="+1">4.1 How to version control jupyter notebook</font></b></p></a>
<a href='#section16'><p style="margin-left: 40px"><b><font size="+1">4.2 Deep Learning</font></b></p></a>

<br/><br/>


<a id='section2'></a>
<b><font size="+1">1.1 The Invariance Problem in single-cell RNA seq analysis</font></b>

Single cell RNA sequence (scRNA-seq) data analysis has gained enormous attention recently and is contributing significantly to diverse research areas such as cancer(Patel et al., [2014](https://science-sciencemag-org.ezp.lib.unimelb.edu.au/content/344/6190/1396)), development (Semrau et al., [2017](https://www.nature.com/articles/s41467-017-01076-4)), autoimmunity (Gaublomme et al., [2015](https://www-sciencedirect-com.ezp.lib.unimelb.edu.au/science/article/pii/S0092867415014890)). One major interest in scRNA-seq data analysis is to represent the data in a way invariant to diverse confounding factors including transcriptional noise (Wagner et al., [2016](https://www-nature-com.ezp.lib.unimelb.edu.au/articles/nbt.3711), capture efficiency and sequnencing depth (Vallejos et al., [2017](https://www-nature-com.ezp.lib.unimelb.edu.au/articles/nmeth.4292)), amplication bias, and batch effects (Shaham et al., [2017](https://academic-oup-com.ezp.lib.unimelb.edu.au/bioinformatics/article/33/16/2539/3611270)), such that only true biological variance is left for downstream analysises including imputation of missing data in the scRNA-seq datasets with highly abundant 'drop-out' events, visualization and clustering, differential gene expression analysis. 

<!-- #region -->
<a id='section3'></a>
<b><font size="+1">1.2 Existing methods to deal with the invariance problem in single-cell RNA seq problem</font></b>

Although deep learning, especially generative deep network for unsupervised learning (<font color=red>(are there other neural network used in single cell rna seq for unsupervised learning?)</font>), has been used recently to analyze single-cell RNA seq data in several researches (Ding, et al. [2018](https://www.nature.com/articles/s41467-018-04368-5), Wang, et al. [2017](https://www.sciencedirect.com/science/article/pii/S167202291830439X), Eraslan, et al. [2018](https://www.nature.com/articles/s41467-018-07931-2), Grønbech, et al. [2019](https://www.biorxiv.org/content/10.1101/318295v3), Lopez et al. [2018](https://www.nature.com/articles/s41592-018-0229-2)), some of them only focus on dimension reduction (Ding, et al. [2018](https://www.nature.com/articles/s41467-018-04368-5),Wang, et al. [2017](https://www.sciencedirect.com/science/article/pii/S167202291830439X)) and clustering cell types (Grønbech, et al. [2019](https://www.biorxiv.org/content/10.1101/318295v3)). Among those that do investigate about removing the technical variations from the data (Deep Count Autoencoder network (DCA) developed by Eraslan, et al [2018](https://www.nature.com/articles/s41467-018-07931-2), 


Single-cell Variational Inference (scVI) developed by Lopez et al. [2018](https://www.nature.com/articles/s41592-018-0229-2)), is superior than other methods, and its superiority will be demonstrated in more details after the introduction of scVI.
<!-- #endregion -->

<!-- #region -->
<a id='section4'></a>
<b><font size="+1">1.3 scVI</font></b>

Explain from autoencoder, to variational autoencoder, to scVI (Valentine, [2019](https://www.biorxiv.org/content/biorxiv/early/2019/08/16/737601.full.pdf))

scVI is essentially a variational autoencoder, one of the most popular approaches to unsupervised learning of complicated distributions (Doersch, [2016](https://arxiv.org/abs/1606.05908)). Therefore, variational autoencoder will be discussed first. Suppose $x$ is a data point in a dataset with an unknown possibility distribution $p(x)$, $z$ is the representation called latent vector for the data point $x$, defined over a high-, but lower-dimensional space $Z$ than the original datapoint $x$. For each $x$, $z$ is a random variable with probability density $p(z)$. Suppose we have a family of derministic functions $f(z;\theta)$, parameterized by a vector $\theta$ in some space $\Theta$, where $f: Z \times \Theta \rightarrow X$, the objective is to optimize $\theta$ such that the probability of each $x$ in the training dataset can be maximized, according to the definition: $p(x) = \int_{Z} p(x|z;\theta)p(z)dz$. In most cases, p(z) is set to be $N(0,I)$, where $I$ is the identity matrix. $p(x)$ can be approximated by sampling a large number of z values $\{z_{1},...,z_{n}\}$ and averaging $p(x|z_{i}), i\in \{1,...,n\}$. In practice, for most $z$ sampled from $p(z)$, $p(x|z)$ will be almost zero, contributing nothing to our estimator for $p(x)$. The solution to circumvent this problem is to narrow down the sampling space of $z$ such that $z$ are more likely to have produced $x$, thus giving a meaningful value of $p(x|z)$. The narrowed space of $z$ is determined by the posteror probability distribution $p(z|x)$, which, however, is usually difficult to compute analytically. Suppose $q(z|x)$ is a function approximator for $p(z|x)$, then:

\begin{equation}
logp(x) - D[q(z|x)||p(z|x)] = E_{z \sim q(z|x)} [logp(x|z)] - D[q(z|x)||p(z)], 
\end{equation}

where $D[q(z|x)||p(z|x)]$ is the Kullback-Leibler divergence (KL divergence or D) between $q(z|x)$ and $p(z|x)$, and $D[q(z|x)||p(z)]$ is KL divergence between $q(z|x)$ and $p(z)$. As $D[q(z|x)||p(z|x)]$ is non-negative, then naturally:

\begin{equation}
logp(x) \geqslant E_{z \sim q(z|x)} [logp(x|z)] - D[q(z|x)||p(z)].
\end{equation}

The better $q(z|x)$ approximates $p(z|x)$, the smaller $D[q(z|x)||p(z|x)]$ is, the closer $E_{z \sim q(z|x)} [logp(x|z)] - D[q(z|x)||p(z)]$ is to $logp(x)$. Therefore, maximization of $logp(x)$ can be achieved by maximizing the lower bound $E_{z \sim q(z|x)} [logp(x|z)] - D[q(z|x)||p(z)]$. 

Most often, both $p(x|z)$ and q(z|x) are parameterized probability distributions, with the parameters estimated from determinstic functions specified by neural network. The neural network for $q(z|x)$ is called encoder, and the neural network for $p(x|z)$ is called decoder. While $p(x|z)$ is problem-specific, for $q(z|x)$, the usual choice is $N(z|\mu(x; \theta), \Sigma(x; \theta))$, where $\mu$ and $\Sigma$ are deterministic functions with parameters $\theta$ learned from data $x$ via neural networks, and $\Sigma$ is set to be diagnal matrix for computational convenience, as $D[q(z|x)||p(z)]$ can be computed easily:

\begin{equation}
D[N(\mu_{0},\Sigma_{0})||N(\mu_{1},\Sigma_{1})] =  \frac{1}{2}\Big( tr \big( \Sigma_{1}^{-1}\Sigma_{0} \big) + (\mu_{1}- \mu_{0})^{T}\Sigma_{1}{-1}(\mu_{1}-\mu_{0}) -K + log\big( \frac{det\Sigma_{1}}{\Sigma_{0}} \big) \Big)
\end{equation}

When $p(z)=N(0,I)$, $\mu_{0}=\mu(x,\theta)$, and $\Sigma_{0}=\Sigma(x,\theta)$,

\begin{equation}
D[N(\mu(x,\theta),\Sigma(x,\theta))||N(0,I)] =  \frac{1}{2}\Big( tr \big( \Sigma(x,\theta) \big) + (\mu(x, \theta))^{T}(\mu(x, \theta)) - K + log det(\Sigma(x,\theta)) \Big)
\end{equation}

To get $E_{z \sim q(z|x)} [logp(x|z)]$, instead of sampling many $z$ from $q(z|x)$, which is computation-demanding, just take one sample of $z$, and treat $logp(x|z)$ for that $z$ as an approximation of $E_{z \sim q(z|x)} [logp(x|z)]$. This strategy is valid because stochastic gradient descent will be carried out over different values of x sampled from the training dataset. Therefore, the process to calculate the stochastic gradient for optimization is to sample a single value of $x$, then sample a single value of $z$ from $q(z|x)$, compute the gradient of $logp(x|z)-D[q(z|x)||p(z)]$, average the gradient over many samples of $x$ and $z$. However, there is a major problem to get the gradient of $logp(x|z)-D[q(z|x)||p(z)]$ because $z$ is randomly sampled from $q(z|x)$, and the backpropogation can not continue down to the encoder network parameters from $z$. The solution is called 'reparameterization trick'. Instead of sampling $z$ directly from $q(z|x)$, sample $\epsilon$ from $N(0,I)$, where $I$ is a identity matrix, then compute $z = \mu(x) + \Sigma_{1/2}(x)*\epsilon$, $\mu(x)$ and $\Sigma(x)$ are the mean and covariance of $q(z|x)$. Because $\epsilon$ has no relationship with parameters in the encoder network, the backprogation process can continue down to the parameters of the encoder network from decoder network through $\mu(x)$ and $\Sigma(x)$.

SCVI, as a variational autoencoder, is applied to input datasets of $N \times G$-matrix with each element $x_{ng}$ being the number of transcripts for gene $g$ in cell $n$. Meanwhile, each cell may also have a batch annotation $s_{n}$. There are two latent random variables, the library size $l_{n}$ which represents nuisance variation due to capture efficiency and sequency depth, and $z_{n}$ which represents the remaining variability including biological difference between cells. The prior distribution of $z_{n}$ is standard gaussian, however, the prior distribution of $l_{n}$ is lognormal. Besides, scvi has the specialty in that the posterior distribution for the generated data from the decoder is a parametrised zero-inflated negative binomial (ZINB) distribution, which is shown below:

\begin{equation}
\begin{aligned}
& z_{n} \sim Normal(0,I) \\
& l_{n} \sim LogNormal(l_{\mu}, l_{\sigma}^{2}) \\
& \rho_{n} = f_{w}(z_{n}, s_{n}) \\
& w_{ng} \sim Gamma(\rho_{n}^{g}, \theta) \\
& y_{ng} \sim Poisson(l_{n}w_{ng}) \\
& h_{ng} \sim Bernoulli(f_{h}^{g}(z_{n},s_{n})) \\
& x_{ng} = \Bigg\{ \begin{matrix}
    y_{ng} & if \hspace{0.1cm} h_{ng} = 0 \\ 0 & otherwise,
  \end{matrix}
\end{aligned}
\end{equation}

where:

if $B$ demotes the number of batches, calculate the empirical mean and variance of the log-library size per batch to get $l_{\mu}$, $l_{\sigma} \in \mathbb{R}_{+}^{B}$. Note that $l_{n}$ itself is not in the log scale,

the function $f_{w}$ stands for the neural network that encodes the mean proportion of transcripts $\rho_{n}$ expressed across all genes in cell $n$, and the function $f_{h}$ stands for the neural network that encodes the probability $h_{n}$ across all genes in cell $n$ of dropping out due to technical effect. Both $f_{w}$ and $f_{h}$ map the latent space $z_{n}$ and batch annotation $s_{n}$ back to the full dimention of all genes: $\mathbb{R}^{d} \times \{0,1\}^{B} \rightarrow \mathbb{R}^{G}$. The last layer of $f_{w}$ uses a softmax activation function.<font color=red>(check the whole network structure for $f_{w}$ and $f_{h}$ here?)</font> The annotations $g$ and $n$ (e.g., $f_{h}^{g}(z_{n}, s_{n})$) stand for a specific gene $g$ in a specific cell $n$. For each cell $n$, the sum of $f_{w}^{g}(z_{n}, s_{n})$ is 1. 

the parameter $\theta \in \mathbb{R}_{+}^{G}$ denotes a gene-specific inverse dispersion and it is estimated via variational Bayesian inference. <font color=red>(check how to estimate $\theta$ here?)</font>


<font color=red>(how to calculate logp(x|z,l,s) and backpropogate?)</font>

There are many traditional statistical methods, compared with deep learning, to infer the biological variation in single-cell data with nuisance factors. scVI is claimed to be superior than the traditional statistical methods in three aspects (Lopez et al., [2018](https://people.eecs.berkeley.edu/~jregier/publications/lopez2018deep.pdf)). First, although both SCVI and other existing methods use a parametric statistical model with a 'expressiion' component (e.g., ZINB-WaVE uses a negative binomial as the expression componnet (Risso, et al., [2018](https://www.nature.com/articles/s41467-017-02554-5)), BISCUIT uses a log normal distribution as the expression component (Prabhakaran, et al., [2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6004614/))) and a zero compnent (usually a bernoulli distribution <font color=red>(is it true?)</font>) to fit the distribution of the random variable which is the count of the transcripts for each gene in each cell, and although both SCVI and other methods are first trying to learn a low-dimensional manifold from the single-cell RNA count data <font color=red>(what are the other low-dimension manifold learning methods?)</font> as well additional covariates provided as metadata (e.g., batch (Risso, et al., [2018](https://www.nature.com/articles/s41467-017-02554-5) (Lopez et al., [2018](https://people.eecs.berkeley.edu/~jregier/publications/lopez2018deep.pdf))), cell quality (Risso, et al., [2018](https://www.nature.com/articles/s41467-017-02554-5))) and map the low-dimensional representation to the parameter of the model, the neural network in scvi frees the mapping function from the constraint of the generalized linear assumption in other methods, which is difficult to justify. <font color=red>(check at least one method e.g. ZINB-WaVE to check what is the generalized linear assumption in this method?)</font>). Second, scvi can be used to analyze the single-cell RNA seq data for all the various downstream tasks to ensure consistency, compared with other methods usually designed specially only for different subsets of all tasks (e.g., imputation and clustering, but not differential expression (Prabhakaran, et al., [2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6004614/))). Third, scvi provides a higher computational scalability, desirable for the ever-growing size of recent datasets consisting of hundreads of thousands of cells (commercial 10x genomics, [2017](https://support.10xgenomics.com/single-cell-gene-expression/datasets/), or envisioned by consortia like the Human Cell Atlas (Regev, et al., [2017](https://elifesciences.org/articles/27041))), compared with other methods applicable only to tens of thousands of cells <font color=red>(why scvi has higher computational scalability? Refer to (Valentine, [2019](https://www.biorxiv.org/content/biorxiv/early/2019/08/16/737601.full.pdf))) and (Matthew, [2019](https://www.biorxiv.org/content/10.1101/237065v4))</font>). 

As mentioned above, scVI is also superior to other deep learning methods (e.g. DCA, Eraslan, et al [2018](https://www.nature.com/articles/s41467-018-07931-2)) for removing the techinical variation, in a way that apart from removing technical variations from all resources, scVI explicitly models two nuisance factors, library size and batch labels. However, there is still much variation from nuisance factors (e.g. batch) remained in the dataset after scVI implementation when those nuisance factors have large confounding effects initially(Xu, et al. [2019](https://www.biorxiv.org/content/10.1101/532895v1)). 
Lopez et al. [2018](https://arxiv.org/pdf/1805.08672.pdf) analyzed that the different dimensions of the latent vectors are hardly independent because the approximateed posterior distribution poorly matches the real posterior and the data we have are always finite, and they also observed empirically that the the independence properties encoded by the generative model wil often not be respected by the approximate posterior. 
<!-- #endregion -->

<a id='section5'></a>
<b><font size="+1">1.4 Existing Penalized scVI to extend scVI in dealing with the invariance problem in single-cell RNA seq analysis</font></b>

In order to impose invariance to nuisance factors in scVI, Lopez et al. [2018](https://arxiv.org/pdf/1805.08672.pdf) proposed a a framework named Hilbert-Schmidt Independence Criterion (HSIC)-constrained variational autoencoder (VAE), which enforces independence between the learnt latent representations and arbitrary nuisance factors through maximizing the traditional variational lower bound of VAE with a penalty for desired independence requirement calibrated by a kernel-based non-parametric method called HSIC. <font color=red>(Describe the HSIC method here in details!!!!!!!, the code for the paper is [here](https://github.com/romain-lopez/HCV/blob/master/scVI/scVIgenqc.py))</font>. 

However, the main drawback for HSIC to measure dependence is that it is not applicable to high-dimensional random variables, which is the inherent shortcoming for non-parametric kernel estimator. 



<!-- #region -->
<a id='section6'></a>
<b><font size="+1">1.5 Use Mutual information as the penalty</font></b>

Belghazi et al. [2018](https://arxiv.org/pdf/1801.04062.pdf) proposed a neural network estimator named MINE for mutual information between high dimensional continuous random variables, where mutual information quantifies both linear and non-linear dependence of two random variables. And when given two random variables $X$ and $Z$, it is defined as:
\begin{equation}
\begin{aligned}
 I(X;Z) = \int_{\mathcal{X} \times \mathcal{Z}} log \frac{d\mathbb{P}_{XZ}}{d\mathbb{P}_{Z} \otimes d\mathbb{P}_{Z}}d\mathbb{P}_{XZ},
\end{aligned}
\end{equation}

where $\mathbb{P}_{XZ}$ is the joint probability distribution of $X$ and $Z$, $\mathbb{P}_{X} = \int_{\mathcal{Z}}d\mathbb{P}_{XZ}$ and $\mathbb{P}_{Z} = \int_{\mathcal{X}}d\mathbb{P}_{XZ}$ are the marginal probability distribution for $X$ and $Z$, respectively. Essentially, the mutual information is equivalent to the Kullback-Leibler (KL-) divergence between the joint $\mathbb{P}_{XZ}$ and the product of the marginals $\mathbb{P} \otimes \mathbb{Q}$: $I(X, Z) = D_{KL}(\mathbb{P}_{XZ}||\mathbb{P}_{X} \otimes \mathbb{P}_{Z})$, where $D_{KL}(\mathbb{P}||\mathbb{Q})=\mathbb{E}_{\mathbb{P}} \Big[log \frac{d\mathbb{P}}{d\mathbb{Q}}\Big]$. Put simply, the mutual information of two random variable is the KL-divergence between the true joint distribution and joint distribution when $X$ and $Z$ are assumed to be independent. According to the Donsker-Varadhan representation, $D_{KL}(\mathbb{P}||\mathbb{Q}) = \underset{T: \Omega \rightarrow \mathbb{R}}{sup} E_{\mathbb{P}}[T] - log(\mathbb{E}_{\mathbb{Q}}[e^{T}])$, where T is all functions under which the two expectations are finite. Consequently, $D_{KL}(\mathbb{P}||\mathbb{Q}) \geq \underset{T \in \mathcal{F}}{sup} E_{\mathbb{P}}[T] - log(\mathbb{E}_{\mathbb{Q}}[e^{T}])$, where $\mathcal{F}$ is any class of functions $T: \Omega \rightarrow \mathbb{R}$. The larger number of functions $\mathcal{F}$ contains, the tighter the lower bound for $D_{KL}(\mathbb{P}||\mathbb{Q})$ is. $\mathcal{F}$ is chosen to be a family of function $T_{\theta}: \mathcal{X} X \mathcal{Z} \rightarrow \mathbb{R}$ parametrized with parameters $\theta \in \Theta$ in the MINE neural network. Therefore, $I(X;Z) \geq I_{\Theta}(X, Z) = \underset{\theta \in \Theta}{sup} E_{\mathbb{P}_{XZ}}[T_{\theta}] - log(\mathbb{E}_{\mathbb{P}_{X} \otimes \mathbb{P}_{Z}}[e^{T_{\theta}}])$


Let x be data points, z be latent variable, s be batch, then the aggregated posterior of $z$ conditional on batch $s$ is:
\begin{equation}
\begin{aligned}
 \hat{q}_{\phi}(z|s) &= \mathbb{E}_{p_{data}(x|s)}[q_{\phi}(z|x|s)] \\
                     &\approx \frac{1}{n}\Sigma_{i=1}^{n}\hat{q}_{\phi}(z|x|s) \\
\end{aligned}
\end{equation}

Let $\mathbb{P}$ and $\mathbb{Q}$ be the aggregated posterior of $z$ conditional on batch $s=0$ and $s=1$ respectively, then we have:
\begin{equation}
\begin{aligned}
                    \mathbb{P} &= \hat{q}_{\phi}(z|s=0) \\
                    \mathbb{Q} &= \hat{q}_{\phi}(z|s=1) \\
\end{aligned}
\end{equation}

According to Belghazi, et al. [2018](https://arxiv.org/pdf/1801.04062.pdf),   KL-divergence between $\mathbb{P}$ and $\mathbb{Q}$ is:

\begin{equation}
\begin{aligned}
D_{KL}(\mathbb{P}||\mathbb{Q}) &= \underset{T: Z \rightarrow \mathbb{R}}{sup} E_{\mathbb{P}}[T] - log(\mathbb{E}_{\mathbb{Q}}[e^{T}]) \\
                               &= \underset{T: Z \rightarrow \mathbb{R}}{sup}\int_{Z|s=0}T(z)\hat{q}_{\phi}(z|s=0)dz - log(\int_{Z|s=1}e^{T(z)}\hat{q}_{\phi}(z|s=1)dz), \\
\end{aligned}
\end{equation}

where $T: Z \rightarrow \mathbb{R}$ means all functions that makes the two expectations finite.

Let $\mathbb{F}$ be any class of functions $T: Z \rightarrow \mathbb{R}$, then the lower bound for $D_{KL}(\mathbb{P}||\mathbb{Q})$ is:

\begin{equation}
\begin{aligned}
D_{KL}(\mathbb{P}||\mathbb{Q}) &\geq \underset{T\in \mathbb{F}}{sup} E_{\mathbb{P}}[T] - log(\mathbb{E}_{\mathbb{Q}}[e^{T}]) \\
                               &= \underset{T\in \mathcal{F}}{sup}\int_{Z|s=0}T(z)\hat{q}_{\phi}(z|s=0)dz - log(\int_{Z|s=1}e^{T(z)}\hat{q}_{\phi}(z|s=1)dz) \\
                               &\approx \underset{T\in \mathcal{F}}{sup}\frac{1}{N_{0}}\Sigma_{i=1}^{N_{0}}T(z_{i}|s=0) - log(\frac{1}{N_{1}}\Sigma_{i=1}^{N_{1}}e^{T(z_{i}|s=1)})\\
\end{aligned}
\end{equation}
<!-- #endregion -->

<a id='section7'></a>
<b><font size="+1">1.6 How to build MineNet and train MI_penalized scVI</font></b>

<b><font size="+1">1.6.1 Explaining the hyperparameters for neural network, architecture of neural network, adversarial training</font></b>

Explain basics for deep learning: why updating weights in the direction of gradient? how to establish the architecture and choose hyperparameters for MineNet: including the lr, initialization, activation, drop out, batch_normalization, optimizer. loss function for MI_penalized scVI.

For the pretraining in adversarial training, how to save weights of the pretrained neural network and use the saved weights to initialize the new neural network with the same structure but different loss function? check this blog to see the technichs: 
[save and reload weights](https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de).

<b><font size="+1">1.6.2 multi-objective optimization and parento front</font></b>

There are two ways to minimize both two objectives: one way is that $\theta^{\lambda} = \underset{\theta}{argmin}[(1-\lambda)*L_{1}(\theta) + \lambda * L_{2}(\theta)], \{\theta^{\lambda}: \lambda \in [0,1]\}$. The second way is called shebyshev, which is $\theta^{\lambda} = \underset{\theta}{argmin} \hspace{0.2cm} max\{(1-\lambda)*L_{1}(\theta), \lambda * L_{2}(\theta)\}, \{\theta^{\lambda}: \lambda \in [0,1]\}$. 

If we draw the true $\left[\begin{array}{c} L_{1}(\theta) \\ L_{2}(\theta) \end{array}\right] $, when $\theta = \underset{\theta}{argmin} \left[\begin{array}{c} L_{1}(\theta) \\ L_{2}(\theta) \end{array}\right]$, the line is called the parento front. Chebyshev is a better approximation of the parento front. When we use the first way, which is $S_{loss} = [(1-\lambda)*L_{1}(\theta) + \lambda * L_{2}(\theta)]$, we could get $L_{2}{\theta} = \frac{S_{loss}-(1-\lambda)*L_{1}(\theta)}{\lambda}$, all the points on the convex hole can be approximated, but points on the concave hole can not be approximated. In our case, $L_{1}(\theta)$ is the std_reconstloss, $L_{2}(\theta)$ is the std_MI penalty, I want to minimimize both. Standardize both reconstloss and MI penalty is to make sure that they are on the same scale, from 0 to 1.


<a id='section8'></a>
<b><font size="+1">2.1 Batch effects remained partially after scVI treatment of artificial datasets and real datasets with a large batch effects to demonstrate that there is room to extend scVI for batch effect removal.</font></b>

Artificially datasets

<b><font size="3">2.1.1 Goal</font></b> 

<b><font size="3">2.1.1 Design</font></b>

<b><font size="3">2.1.1 Get Data</font></b>

<b><font size="3">2.1.1 Code</font></b>

<b><font size="3">2.1.1 Result and Lab Meeting Discussion</font></b>

<b><font size="3">2.1.1.1 Change Library Size</font></b>

<b><font size="3">2.1.1.2 Change number of genes with expression</font></b>

<b><font size="3">2.1.1.3 Change gene's expression proportion</font></b>

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
Real datasets

<b><font size="3">2.1.2 Goal</font></b> 

Compare SCVI and SCVI with penalty of mutual information(MI) between nuisance factor and latent factor, using cluster metrics, and tsne plots as the comparison criteria.  

<b><font size="3">2.1.2 Design</font></b>

Apply SCVI, SCVI with penalty of mutual information to the datasets: MouseMarrow, pbmc, retina.
When the nuisance factor is 'batch', as batch is a categorical variable, use nearest neighbor method to estimate the mutual information between batch and the latent factor. When the nuisance factor is 'library size', use Mine_Net4 to estimate the mutual information between library size and the latent factor, as library size is a lognormal variable.

According to the paper [Deep Generative Modeling for Single-cell Transcriptomics](https://people.eecs.berkeley.edu/~jregier/publications/lopez2018deep.pdf), the four clustering metrics are :

<b>ASW(Average Silhouette width)</b>:

Silhouette width for each sample i in a latent space or a similarity matrix is: $s(i)=\frac{b(i)-a(i)}{max\{a(i), b(i)\}}$, where $a(i)$ is the average distance of sample i to all other data points in the same cluster $c_{i}$, and $b(i)$ is the minimal average distance of sample i to all other data points in the same cluster $c_{j}$ among all clusters c. In this paper, the cells are clustered by pre-annotated cell type labels, then use latent space vectors to calculate the average silhouette distance. ASW is the average of silhouette width for all samples. Therefore, ASW, the higher the better, on the assumption that the preannoted cell type labels are largely correct. However, if the cells is pre-clustered by batches, the lower, the better

<b>ARI(Adjusted Rand Index)</b>:

ARI is defined as $ARI=\frac{\Sigma_{ij}\binom{n_{ij}}{2} - [\Sigma_{i}\binom{a_{i}}{2}\Sigma_{j}\binom{{bj}}{2}]/2}{\frac{1}{2}[\Sigma_{i}\binom{a_{i}}{2} + \Sigma_{j}\binom{{bj}}{2}]-[\Sigma_{i}\binom{a_{i}}{2} + \Sigma_{j}\binom{{bj}}{2}]/\binom{{n}}{2}}$, where $n_{ij}$, $a_{i}$, $b_{j}$ are values from a contigency table. In this study, the contigency table is generated by comparing the clustering similarity between K-means clustering to the latent vector of cells, and the clustering by preannotated cell type labels. $n_{ij}$ is the number of same cells in cluster i by K-means and cluster j by preannoted cell type labels. For ARI, the higher the better on the assumption that the preannoted cell type labels are largely correct.

<b>NMI(Normalized Mutual Information)</b>:

NMI is defined as $NMI=\frac{I(P;T)}{\sqrt{H(P)H(T)}}$, where $P$, $T$ are the empirical categorical distributions for the predicted and real clustering. $I$ is the mutual entropy and $H$ is the Shannon entropy. In this study, the predicted clustering is the K-means clustering applied to latent vectors. The real clustering is the clustering by preannoted cell type labels. For NMI, the higher the better on the assumption that the preannoted cell type labels are largely correct.

<b>BE</b>:

Calculate a similarity matrix for the cells and make U be a uniform random variable on the population of cells. Take $B_{U}$ as the empirical frequencies for the 50 nearest neighbors of cell $U$ being a in batch b. Get the entropy of this categorical variable and average over $T=100$ values of $U$. In this study, for BE, the higher the better.

Compare neural network (SAUCI), PCA, Diffusion maps, tSNE, PHATE in visualizing single-cell RNA data in exploratory analysis. (Matthew, [2019](https://www.biorxiv.org/content/10.1101/237065v4))

<b><font size="3">a. MouseMarrow</font></b>

Mouse Marrow dataset is generated as the following steps: 1. git clone from the github webpage https://github.com/YosefLab/scVI-data.git to get the mouse_gene_len.txt file. Store mouse_gene_len.txt to the directory where MouseTM-10x and MouseTM-ss2 will be downloaded into. 2. Download both MouseTM-10x and MouseTM-ss2 using TabulaMuris() in ./scvi/dataset/muris_tabula.py file. Then subsample MouseTM-10x and MouseTM-ss2, and concatenate them by row to form the Mouse Marrow dataset. The specific code is in ./code/tune_hyperparameter_for_MineNet_MonteCarlo.py. Note: mouse_gene_len.txt should exist in the directory first before line33 to line37 in ./code/tune_hyperparameter_for_MineNet_MonteCarlo.py can generate Mouse Marrow dataset. We choose this dataset because the batch effect in this combined dataset is expected to be relatively large according to the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1).

When the nuisance factor is batch, for each iteration, apply SCVI once to MouseMarrow Dataset. Then apply SCVI+MI_penalty 6 times to MouseMarrow Dataset, each time of which will have a different scale multiplied to the MI estimator. The hyperparameters for SCVI and SCVI+MI_penalty is as follows:
   
    -n_layers_encoder: [2] (the number of hidden layers for encoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_decoder: [2] (the number of hidden layers for decoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_encoder (for l_encoder): [1] (the default value from the original SCVI)
    
    -n_hidden: [128] (the number of nodes in each hidden layer in both encoder and decoder in SCVI, and SCVI+MI_penalty)
    
    -n_latent: [10] (the dimension of the latent vector Z)
    
    -dropout_rate: [0.1]
    
    -reconstruction_loss: ['zinb']
    
    -use_batches: [True]
    
    -use_cuda: [False]
    
    -Scale: [200, 500, 800, 1000, 2000, 5000, 10000, 100000, 1000000] (the scale multiplied to MI estimator)
    
    -train_size: [0.8] (the ratio to split the dataset into training and testing dataset)
    
    -lr: [0.001](learning rate)
    
    -n_epochs: [250] 
    
 
Repeat the iteration 100 times as SCVI will produce different results on the same dataset multiple times. The reason could be the initialization of the weights in the scvi deep network, different initialization values could result in different optimization result. Or the minibatch of the training process, Or could be the reparametrization of z in the scVI method, according to the [tutorial_of_scVI](https://arxiv.org/abs/1606.05908). The paper titled [Deep_generative_modeling_for_single-cell_transcriptomics](https://www-nature-com.ezp.lib.unimelb.edu.au/articles/s41592-018-0229-2#Sec43) investigates the variability of scVI due to different initialization of weights in supplementary figure 1.d.

The values for the following hyperparameters: n_layers, n_hidden, n_latent, dropout_rate, lr, n_epochs, reconstruction_loss are directly borrowed from the code of the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1).

As the invariance increases as network becomes deeper, I also tried n_layers_encoder = 10, while all other parameters remain the same:
 
    -n_layers_encoder: [10] (the number of hidden layers for encoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_decoder: [2] (the number of hidden layers for decoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_encoder (for l_encoder): [10] (equals the n_layers_encoder for z_encoder)
    
    -n_hidden: [128] (the number of nodes in each hidden layer in both encoder and decoder in SCVI, and SCVI+MI_penalty)
    
    -n_latent: [10] (the dimension of the latent vector Z)
    
    -dropout_rate: [0.1]
    
    -reconstruction_loss: ['zinb']
    
    -use_batches: [True]
    
    -use_cuda: [False]
    
    -Scale: [200, 500, 800, 1000, 2000, 5000, 10000, 100000, 1000000] (the scale multiplied to MI estimator)
    
    -train_size: [0.8] (the ratio to split the dataset into training and testing dataset)
    
    -lr: [0.001](learning rate)
    
    -n_epochs: [250] 


When the nuisance factor is library size, .......

<b><font size="3">b. Pbmc</font></b>

pbmc dataset is built-in scVI dataset. just use the function PbmcDataset() in scvi/dataset/pbmc.py to load the dataset

When the nuisance factor is batch, for each iteration, apply SCVI once to Pbmc Dataset. Then apply SCVI+MI_penalty 7 times to Pbmc Dataset, each time of which will have a different scale multiplied to the MI estimator. Repeat the iteration 100 times. The hyperparameters for SCVI and SCVI+MI_penalty for Pbmc is as follows:
   
    -n_layers: [1] (the number of hidden layers in both encoder and decoder in SCVI, and SCVI+MI_penalty)
    
    -n_hidden: [256] (the number of nodes in each hidden layer in both encoder and decoder in SCVI, and SCVI+MI_penalty)
    
    -n_latent: [14] (the dimension of the latent vector Z)
    
    -dropout_rate: [0.5]   
    
    -reconstruction_loss: ['zinb']
    
    -use_batches: [True]
    
    -use_cuda: [False]
    
    -Scale: [200, 500, 800, 1000, 2000, 5000, 10000,1000000] (the scale multiplied to MI estimator)
    
    -train_size: [0.8] (the ratio to split the dataset into training and testing dataset)
    
    -lr: [0.01]
    
    -n_epochs: [170] (used in the Harmonization and Annotation of single-cell)
    
    
The values for the following hyperparameters: n_layers, n_hidden, n_latent, dropout_rate, lr, n_epochs, reconstruction_loss are directly borrowed from Table3 on the website [hyperparameter search for SCVI](https://yoseflab.github.io/2019/07/05/Hyperoptimization/)

<b><font size="3">c. Retina</font></b>



<b><font size="3">2.1.2 Code</font></b>

In the shell,  git clone the Hui_Li branch of the github page https://github.com/susanwe/single-cell-rna-seq.git. Then in the working directory, create a virtual environment named venv, activate venv, and install all required packages with specific versions listed in Project1_Modify_SCVI-requirements.txt in the venv environment. Use command: cp code/tune_hyperparameter_for_MineNet4_MonteCarlo.sh tune_hyperparameter_for_MineNet4_MonteCarlo.sh to copy the tune_hyperparameter_for_MineNet4_MonteCarlo.sh file to the working directory. Use command: cp code/tune_hyperparameter_for_MineNet4_MonteCarlo.py tune_hyperparameter_for_MineNet4_MonteCarlo.py to copy the tune_hyperparameter_for_MineNet4_MonteCarlo.py to the working directory. Then in the working directory use command: sh tune_hyperparameter_for_MineNet4_MonteCarlo.sh to run tune_hyperparameter_for_MineNet4_MonteCarlo.py 100 times.

After all jobs finish on spartan, git clone the Hui_Li branch of the github page https://github.com/susanwe/single-cell-rna-seq.git to local computer. Use the command: scp myusername@spartan.hpc.unimelb.edu.au:/data/projects/myproject/remote.dat local.dat to copy the results on spartan to local computer.

<b><font size="3">2.1.2 Result</font></b>

<b><font size="3">a MouseMarrow</font></b>

The result for MouseMarrow when n_layers_encoder = 2 is stored in ./result/tune_hyperparameter_for_MineNet/muris_tabula/, and is summarized in Fig4. The tsne plot is shown in Fig6. 

The result for MouseMarrow when n_layers_encoder = 10 is stored in ./result/tune_hyperparameter_for_MineNet/muris_tabula_encoder10layers/, and is summarized in Fig5. The tsne plot is shown in Fig6. 


<!-- #endregion -->

```python slideshow={"slide_type": "-"}
import os
%matplotlib inline
import itertools
exec(open('code\\SummarizeResult.py').read())
Average_ClusteringMetric_Barplot(dataset_name = "muris_tabula",nuisance_variable='batch',results_dict = "result/tune_hyperparameter_for_SCVI_MI/muris_tabula/", n_sample = 100, hyperparameter_config = {'MIScale':[200, 500, 800, 1000, 2000, 5000, 10000,100000,1000000]})

file_paths = []
subfile_titles = []
MIScales = [200,500,800, 1000, 2000,5000,10000, 100000, 1000000]

for i,MIScale in enumerate(MIScales):
    file_paths = file_paths + ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\trainset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch',  MIScale, 100)]
    file_paths = file_paths + ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\testset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
    subfile_titles = subfile_titles + ['trainset_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
    subfile_titles = subfile_titles + ['testset_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','training')]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','testing')]
subfile_titles += ['%s, %s, %s, clusteringmetrics' % ('muris_tabula', 'batch','training')]
subfile_titles += ['%s, %s, %s, clusteringmetrics' % ('muris_tabula', 'batch','testing')]
SummarizeResult('image',file_paths,subfile_titles,'Fig4: Clustering metrics comparison between scVI and scVI+MI_Penalty on muris tabula dataset, batch nuisance factor')
```

Try to adjust hyperparameters:

1. When the hyperparameters are as follows:
   
    -n_layers_encoder: [2] (the number of hidden layers for encoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_decoder: [2] (the number of hidden layers for decoder network in SCVI, and SCVI+MI_penalty)
    
    -n_layers_encoder (for l_encoder): [1] (the default value from the original SCVI)
    
    -n_hidden: [128] (the number of nodes in each hidden layer in both encoder and decoder in SCVI, and SCVI+MI_penalty)
    
    -n_latent: [10] (the dimension of the latent vector Z)
    
    -dropout_rate: [0.1]
    
    -reconstruction_loss: ['zinb']
    
    -use_batches: [True]
    
    -use_cuda: [False]
    
    -Scale: [200, 500, 800, 1000, 2000, 5000, 10000, 100000, 1000000] (the scale multiplied to MI estimator)
    
    -train_size: [0.8] (the ratio to split the dataset into training and testing dataset)
    
    -lr: [0.001](learning rate)
    
    -n_epochs: [250] 
 batch_mixing_entropy is about 0.05 for vae, and 0.07 for vae_MI when scale=100000
 
2. From case 1, when only change the number of hidden layers in l_encoder to n_layers_encoder (which is 2 in the case), batch_mixing_entropy is about 0.12 for vae, and 0.14 for vae_MI (scale=100000) (this is the one time result)

3. From case 1, when maintain the hyperparameter in 1, but change MI=max(MI_l_s, MI_z_s), the batch_mixing_entropy is about 0.04 for vae, 0.1 for vae_MI (scale=100000)(this is the one time result)

4. From case 1, when change n_epochs=500, lr=0.0005, MI=max(MI_l_s, MI_z_s), the batch_mixing_entropy is about 0.11 for vae, 0.12 for vae_MI (scale=100000)(this is the one time result)

5. From case 1, when change change the number of hidden layers in l_encoder to n_layers_encoder (which is 2 in the case), MI=max(MI_l_s, MI_z_s), batch_mix_entropy for vae is 0.12, vae_MI(scale: 100000) is 0.13, vae_MI(scale:10000000) is  0.11 (this is the one time result)


```python
import os
%matplotlib inline
import itertools
exec(open('code\\SummarizeResult.py').read())
Average_ClusteringMetric_Barplot(dataset_name = "muris_tabula",nuisance_variable='batch',results_dict = "result/tune_hyperparameter_for_SCVI_MI/muris_tabula_encoder10layers/", n_sample = 100, hyperparameter_config = {'MIScale':[200, 500, 800, 1000, 2000, 5000, 10000,100000,1000000]})

file_paths = []
subfile_titles = []
MIScales = [200,500,800, 1000, 2000,5000,10000,100000,1000000]

for i,MIScale in enumerate(MIScales):
    file_paths = file_paths + ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\trainset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch',  MIScale, 100)]
    file_paths = file_paths + ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\testset_mean_clustering_metrics_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
    subfile_titles = subfile_titles + ['trainset_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
    subfile_titles = subfile_titles + ['testset_%s_%s_MIScale%s_%ssamples.png'%('muris_tabula', 'batch', MIScale, 100)]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','training')]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','testing')]
subfile_titles += ['%s, %s, %s, encoder10layer,clusteringmetrics' % ('muris_tabula', 'batch','training')]
subfile_titles += ['%s, %s, %s, encoder10layer,clusteringmetrics' % ('muris_tabula', 'batch','testing')]

file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','training')]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\%s_%s_%s_clusteringmetrics.png' % ('muris_tabula', 'batch','testing')]
subfile_titles += ['%s, %s, %s, encoder2layer,clusteringmetrics' % ('muris_tabula', 'batch','training')]
subfile_titles += ['%s, %s, %s, encoder2layer,clusteringmetrics' % ('muris_tabula', 'batch','testing')]
SummarizeResult('image',file_paths,subfile_titles,'Fig5: Clustering metrics comparison between scVI and scVI+MI_Penalty on muris tabula dataset, batch nuisance factor')
```

```python
file_paths = ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\trainset_tsne_SCVI_%s_%s_sample%s.png'%('muris_tabula', 'batch', 0)]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\testset_tsne_SCVI_%s_%s_sample%s.png'%('muris_tabula', 'batch', 0)]

subfile_titles = ['scVI_%s_%s_encoder2layers_trainset'%('muris_tabula', 'batch')]
subfile_titles += ['scVI_%s_%s_encoder2layers_testset'%('muris_tabula', 'batch')]
MIScales = [100000]

for i,MIScale in enumerate(MIScales):
        file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\trainset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s.png'%('muris_tabula', 'batch', MIScale, 0)]
        file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\testset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s.png'%('muris_tabula', 'batch', MIScale, 0)]
        subfile_titles += ['scVI+MI_penalty %s %s MIScale%s trainset'%('muris_tabula', 'batch', MIScale)]
        subfile_titles += ['scVI+MI_penalty %s %s MIScale%s testset'%('muris_tabula', 'batch', MIScale)] 

file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\trainset_tsne_SCVI_%s_%s_sample%s.png'%('muris_tabula', 'batch', 0)]
file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\testset_tsne_SCVI_%s_%s_sample%s.png'%('muris_tabula', 'batch', 0)]

subfile_titles += ['scVI_%s_%s_encoder10layers_trainset'%('muris_tabula', 'batch')]
subfile_titles += ['scVI_%s_%s_encoder10layers_testset'%('muris_tabula', 'batch')]

for i,MIScale in enumerate(MIScales):
        file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\trainset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s.png'%('muris_tabula', 'batch', MIScale, 0)]
        file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula_encoder10layers\\testset_tsne_SCVI+MI_%s_%s_MIScale%s_sample%s.png'%('muris_tabula', 'batch', MIScale, 0)]
        subfile_titles += ['scVI+MI_penalty %s %s MIScale%s trainset'%('muris_tabula', 'batch', MIScale)]
        subfile_titles += ['scVI+MI_penalty %s %s MIScale%s testset'%('muris_tabula', 'batch', MIScale)] 

SummarizeResult('image',file_paths,subfile_titles,'Fig6: tsne plot of scVI and scVI+MI_penalty for muris tabula dataset, batch nuisance factor')
```

```python
import os
%matplotlib inline
import itertools
exec(open('code\\SummarizeResult.py').read())
Average_ClusteringMetric_Barplot(dataset_name = "Pbmc",nuisance_variable='batch',results_dict = "result/tune_hyperparameter_for_MineNet/pbmc/", n_sample = 100, scvi_n_layers=1, hyperparameter_config = {'MineLoss_Scale':[200, 500, 800, 1000, 2000, 5000, 10000,100000]})

file_paths = []
subfile_titles = []
MineLoss_Scales = [200,500,800, 1000, 2000,5000,10000,100000]
scvi_n_layers = 1 

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
    file_paths = file_paths + ['result\\tune_hyperparameter_for_MineNet\\pbmc\\trainset_mean_clustering_metrics_%s_%s_n_layers%s_MineLossScale%s_%ssamples.png'%('Pbmc', 'batch', scvi_n_layers, MineLoss_Scale, 100)]
    file_paths = file_paths + ['result\\tune_hyperparameter_for_MineNet\\pbmc\\testset_mean_clustering_metrics_%s_%s_n_layers%s_MineLossScale%s_%ssamples.png'%('Pbmc', 'batch', scvi_n_layers, MineLoss_Scale, 100)]
    subfile_titles = subfile_titles + ['trainset_%s_%s_MineLossScale%s_%ssamples.png'%('Pbmc', 'batch', MineLoss_Scale, 100)]
    subfile_titles = subfile_titles + ['testset_%s_%s_MineLossScale%s_%ssamples.png'%('Pbmc', 'batch', MineLoss_Scale, 100)]
file_paths += ['result\\tune_hyperparameter_for_MineNet\\pbmc\\%s_%s_n_layers%s_%s_clusteringmetrics.png' % ('Pbmc', 'batch',scvi_n_layers,'training')]
file_paths += ['result\\tune_hyperparameter_for_MineNet\\pbmc\\%s_%s_n_layers%s_%s_clusteringmetrics.png' % ('Pbmc', 'batch',scvi_n_layers,'testing')]
subfile_titles += ['%s, %s, %s, clusteringmetrics' % ('Pbmc', 'batch','training')]
subfile_titles += ['%s, %s, %s, clusteringmetrics' % ('Pbmc', 'batch','testing')]
SummarizeResult('image',file_paths,subfile_titles,'Fig8: Clustering metrics comparison between scVI and scVI+MI_Penalty on Pbmc Dataset, batch nuisance factor')
```

```python
scvi_n_layers = 1
file_paths = ['result\\tune_hyperparameter_for_MineNet\\pbmc\\trainset_tsne_SCVI_%s_%s_n_layers%s_sample%s.png'%('Pbmc', 'batch', scvi_n_layers, 0)]
file_paths += ['result\\tune_hyperparameter_for_MineNet\\pbmc\\testset_tsne_SCVI_%s_%s_n_layers%s_sample%s.png'%('Pbmc', 'batch', scvi_n_layers, 0)]

subfile_titles = ['scVI_%s_%s_trainset'%('Pbmc', 'batch')]
subfile_titles += ['scVI_%s_%s_testset'%('Pbmc', 'batch')]
MineLoss_Scales = [800, 2000]

for i,MineLoss_Scale in enumerate(MineLoss_Scales):
        file_paths += ['result\\tune_hyperparameter_for_MineNet\\pbmc\\trainset_tsne_SCVI+MINE_%s_%s_n_layers%s_sample%s_MineLossScale%s.png'%('Pbmc', 'batch',scvi_n_layers, 0, MineLoss_Scale)]
        file_paths += ['result\\tune_hyperparameter_for_MineNet\\pbmc\\testset_tsne_SCVI+MINE_%s_%s_n_layers%s_sample%s_MineLossScale%s.png'%('Pbmc', 'batch',scvi_n_layers, 0, MineLoss_Scale)]
        subfile_titles += ['scVI+MI_penalty %s %s MineLoss_Scale%s trainset'%('Pbmc', 'batch', MineLoss_Scale)]
        subfile_titles += ['scVI+MI_penalty %s %s MineLoss_Scale%s testset'%('Pbmc', 'batch', MineLoss_Scale)] 
SummarizeResult('image',file_paths,subfile_titles,'Fig9: tsne plot of scVI and scVI+MI_penalty for pbmc dataset, batch nuisance factor')
```

<a id='section9'></a>
<b><font size="+1">2.2 Compare estimated mutual information with true mutual inforamtion</font></b>

<b><font size="+1">2.2.1 Goal </font></b>

Compare estimated mutual information with true mutual information for three cases, namely, mutual information between a gaussian random variable and a gaussian random variable, between a lognorm random variable and a gaussian random variable, between a categorical random variable and a gaussian random variable

<b><font size="3">2.2.2 Design</font></b>

<b><font size="3">a. between a gaussian random variable and a gaussian random variable</font></b>

Estimated mutual information is compared with true mutual information when the dimension of the two gaussian random variables  is 2 and 20. Take dimension 2 for example, suppose the two gaussian random variables are $X$ and $Y$. According to the paper [Mutual Information Neural Estimator](https://arxiv.org/pdf/1801.04062.pdf), set $P\left(X,Y\right)$ to be a gaussian distribution of dimension 4, with the componentwise correlation $corr(X_{i},Y_{j})=\delta_{ij}\rho$, where $\rho\in\left(-1,1\right)$ and $\delta_{ij}$ is Kronecker’s delta. And both $X$ and $Y$ have the standard normal gaussian distribution with dimension 2. 

To calculate the true mutual information between $X$ and $Y$, use the formula $I(X,Y) = H(X) + H(Y) - H(X,Y)$ , where $I(X,Y)$ is the mutual information between $X$ and $Y$, $H(X)$ and $H(Y)$ are the marginal differential entropy for $X$ and $Y$ repectively, $H(X,Y)$ is the joint differential entropy for $X$ and $Y$. $H(X)$, $H(Y)$ and $H(X,Y)$ are calculated by built-in function in python, which is scipy.stats.multivariate_normal.entropy().

To estimate the mutual information between $X$ and $Y$, four neural network architectures, which are Mine_Net, Mine_Net2, Mine_Net3, Mine_Net4 are tried. All the 4 network architectures uses the algorithm 1 in the paper [Mutual Information Neural Estimator](https://arxiv.org/pdf/1801.04062.pdf) as the fundamental guidline. However, the main difference between Mine_Net3, Mine_Net4 and Mine_Net, Mine_Net2 is that MineNet, MineNet2 first transforms two random variables into two real numbers, and then linearly transform and activate the two real numbers. While, MineNet3, MineNet4 transforms two random variables into two vectors, then linearly transform and activate the two vectors, finally convert the combined vector into a real number. The main difference between Mine_Net3 and Mine_Net4 is that Mine_Net3 transforms the two random variables into two vectors separately, while Mine_Net4 transforms the matrix containing the two random variables together to a combined vector. The following shows the hyperparameter for the four network architectures.

The hyperparameters for Mine_Net:

    -n_hidden_z: [10], (the node number of each hidden layer)

    -n_layers_z: [10, 30, 50], (the number of layers)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],
    
    -train_size: [0.5]

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameters for Mine_Net2:

    -n_hidden_z: [10],

    -n_layers_z: [10],

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],
    
    -train_size: [0.5]

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameters for Mine_Net3:

    -H : [10] (H means the dimension of the vectors)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],
    
    -train_size: [0.5]

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


The hyperparameters for Mine_Net4:

    -n_latents = [32, 16] (the node number of the two hidden layers which converts the concatenated matrix of the two random variables into a vector)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],
    
    -train_size: [0.5]

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    
<b><font size="3">b. between a lognormal random variable and a gaussian random variable</font></b>

The dimension for lognormal random variable is 1, and 2 dimensions for the gaussian random variable. Suppose $X$ is the lognormal random variable and $Y$ is the gaussian random variable. Let $Z=ln(X)$. Similar as the gaussian and gaussian case, set $P\left(Z,Y\right)$ to be a gaussian distribution of dimension 3, with the componentwise correlation $corr(Z_{i},Y_{j})=\delta_{ij}\rho$, , $\rho\in\left(-1,1\right)$, and $\delta_{ij}$ is Kronecker’s delta. In this case, the covariance matrix for $P(Z,Y)$ will be \begin{pmatrix}1 & \rho & 0 \\ \rho & 1 & 0 \\ 0 & 0 & 1\end{pmatrix}. Then by simple derivation, the joint distribution $P(X,Y) = P(X|Y)P(Y) = P(Z|Y)*|\frac{d_{Z}}{d_{X}}|*P(Y) = \frac{1}{X}P(Z,Y) = \frac{1}{X}P(ln(X),Y)$. $P(ln(X)) \sim N(0,1)$ and Y is a standard gaussian variable with dimension 2. 

To calculate the true mutual information between $X$ and $Y$, use the formula $I(X,Y) = H(X) + H(Y) - H(X,Y)$ , where $I(X,Y)$ is the mutual information between $X$ and $Y$, $H(X)$ and $H(Y)$ are the marginal differential entropy for $X$ and $Y$ repectively, $H(X,Y)$ is the joint differential entropy for $X$ and $Y$. $H(X)$ and $H(Y)$ are calculated by built-in function in python, which are scipy.stats.lognorm.entropy(), scipy.stats.multivariate_normal.entropy(). $H(X,Y)$ is calculated from the integration of $-P(X,Y)logP(X,Y)$.

To estimate the mutual information between $X$ and $Y$, Mine_Net4 is tried. The hyperparameters for MineNet4 architecture are:

    -n_latents = [32, 16] (the node number of the two hidden layers which converts the concatenated matrix of the two random variables into a vector)

    -Gaussian_Dimension: [2, 20],

    -sample_size: [14388],
    
    -train_size: [0.5]

    -rho: [-0.99, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

<b><font size="3">c. between a categorical random variable and a gaussian mixture random variable</font></b>

The dimension for the categorical random variable is 1, 10 categories, and 2 dimensions for the gaussian mixture random variable. Suppose $X$ is the categorical random variable, $Y$ is the gaussian mixture random variable. Set $P(Y|X=x) = N(\mu_{x}, \sigma^2_{x})$. Mutual information between $X$ and $Y$ is introduced by correlating $\mu_{x}$ with $X$. The first case is when all the 10 categories of $X$ have the same probability, and $P(Y|X) are all standard normal with dimension 2$. The true mutual information between $X$ and $Y$ are 0. Case 2 is when all the 10 categories of $X$ have the same probability, and $P(Y|X)$ are normal with dimension 2, with mean of each dimension increasing 2 from 0 when category index increase, however, covariance matrix is as the same as in case 1. The way to change mean of $P(Y|X)$ could also be increasing 20, or be increasing and then decreasing as category index increases. And the probability of each category could be different. In total, 7 cases have been designed, each of which has a different mutual information. For each case, by simple derivation, the marginal distribution $P(Y) =\sum_{x=1}^{10}P(Y|X=x)P(X=x)$.

To calculate the true mutual information between $X$ and $Y$, use the formula $I(X,Y) = H(Y) - H(Y|X)$ , where $I(X,Y)$ is the mutual information between $X$ and $Y$, $H(Y)$ is the marginal differential entropy for $Y$, and $H(Y|X)$ is the conditional differential entropy $Y$ given $X$. $H(Y)$ is calculated from the integration of $-P(Y)logP(Y)$, $H(Y|X)$ is calculated by built-in function in python, which is scipy.stats.multivariate_normal.entropy().

To estimate the mutual information between $X$ and $Y$, Mine_Net and [nearest neighbor method](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable) are tried. Note: the categorical variable is transformed into dummy variable in Mine_Net method. The hyperparameters for the two methods are:

The hyperparameters for Mine_Net:

    -n_hidden_z: [10],

    -n_layers_z: [10],

    -Gaussian_Dimension: [2],

    -sample_size: [14388],
    
    -train_size: [0.5]

The parameters for nearest neighbors:

    - repos: 100 (the estimator from the nearest neighbor is the mean of 100 estimators, each of which is for a data set of sample size 128)
    
    - sample_size: 128 (because the minibatch size in scvi training is 128)
    
    - k: 3 (the number of neighbors)
    
    - base: 2 (a parameter with default value 2 in the nearest neighbor method)

<b><font size="3">2.2.3 Code</font></b>

code/compare_estimatedMI_with_trueMI_gaussian_continuous.py produces estimatedMI_with_trueMI.csv in ./result/compare_estimatedMI_with_trueMI/continuous_gaussian/, use this csv file as read-in dataset for the Summarize_EstimatedMI_with_TrueMI() function in code/SummarizeResult.py file to produce plots in Fig5 and Fig6.

code/compare_estimatedMI_with_trueMI_gaussian_categorical.py produces estimatedMI_with_trueMI.csv in ./result/compare_estimatedMI_with_trueMI/gaussian_categorical/, use this csv file as read-in dataset for the Summarize_EstimatedMI_with_TrueMI() function in code/SummarizeResult.py file to produce plots in Fig7.

<b><font size="3">2.2.4 Result</font></b>

<b><font size="3">a. between a gaussian random variable and a gaussian random variable</font></b>

MineNet works bad for both dimension 2 and dimension 20 with all hyperparameter combinations which have been tried, whenever apply the trained network to the training dataset and testing dataset. The estimator is always near zero even though the true mutual information is large, like 30. Refer to Fig5 in the jupyternotebook in last committ ahead of current on the github repo.

MineNet2 works bad for both dimension 2 and dimension 20 with the hyperparameter combination which has been tried, whenever apply the trained network to the training dataset and testing dataset. The estimator is always near zero even though the true mutual information is large, like 30. Refer to Fig6 in the jupyternotebook in last committ ahead of current on the github repo.

MineNet3 works well for dimension 2 with the hyperparameter combination which has been tried when apply the trained network to the training dataset, but bad when apply to the testing dataset. And MineNet3 works bad for dimension 20 with the hyperparameter combination for both training and testing dataset. Refer to Fig7 in the jupyternotebook in last committ ahead of current on the github repo.

MineNet4 works well for dimension 2 with the hyperparameter combination which has been tried when applied to both training dataset and testing dataset, but works bad for dimension 20 with the hyperparameter combination when applied to both training and testing dataset. Please refer to Fig5.

<b><font size="3">b. between a lognormal random variable and a gaussian random variable</font></b>

MineNet4 works well when gaussian dimension is 2 using the hyperparameter combination for both training and testing datasets. Please refer to Fig6.

<b><font size="3">c. between a categorical random variable and a gaussian mixture random variable</font></b>

MineNet works bad for gaussian mixture dimension 2 with the hyperparameter combination tried, while nearest neighbor estimator works better for gaussian mixture dimension 2. Please refer to Fig7. Nearest neighbor estimator could be negative values when true mutual information is very small.

```python
import os
import math
import itertools
%matplotlib inline
exec(open('code\\SummarizeResult.py').read())
```

```python
Summarize_EstimatedMI_with_TrueMI(file_path='.\\result\\compare_estimatedMI_with_trueMI\\continuous_gaussian\\estimatedMI_with_trueMI.csv', method='Mine_Net4', distribution='gaussian', gaussian_dimensions=[2,20])
hyperparameter_config = {
        'method': ['Mine_Net4'],
        'distribution': ['gaussian'],
        'gaussian_dimension': [2, 20],
        'type': ['training','testing']
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
file_paths = []

image_dict = '.\\result\\compare_estimatedMI_with_trueMI\\continuous_gaussian'
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    method = value[0]
    distribution = value[1]
    gaussian_dimension = value[2]
    type = value[3]
    file_paths = file_paths + [image_dict + '\\%s_%s_gaussian_dim%s_%s.png'%(method,distribution, gaussian_dimension,type)]
    
SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig5: Compare estimated and true MI between gaussian and gaussian")
```

```python
Summarize_EstimatedMI_with_TrueMI(file_path='.\\result\\compare_estimatedMI_with_trueMI\\continuous_gaussian\\estimatedMI_with_trueMI.csv', method='Mine_Net4', distribution='lognormal', gaussian_dimensions=[2])
hyperparameter_config = {
        'method': ['Mine_Net4'],
        'distribution': ['lognormal'],
        'gaussian_dimension': [2],
        'type': ['training','testing']
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

file_paths = []
image_dict = '.\\result\\compare_estimatedMI_with_trueMI\\continuous_gaussian'
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    method = value[0]
    distribution = value[1]
    gaussian_dimension = value[2]
    type = value[3]
    file_paths = file_paths + [image_dict + '\\%s_%s_gaussian_dim%s_%s.png'%(method,distribution, gaussian_dimension,type)]
    
SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig6: Compare estimated and true MI between lognormal and gaussian")
```

```python
Summarize_EstimatedMI_with_TrueMI(file_path='.\\result\\compare_estimatedMI_with_trueMI\\gaussian_categorical\\estimatedMI_with_trueMI.csv', method='Mine_Net', distribution='categorical', gaussian_dimensions=[2])
Summarize_EstimatedMI_with_TrueMI(file_path='.\\result\\compare_estimatedMI_with_trueMI\\gaussian_categorical\\estimatedMI_with_trueMI.csv', method='nearest_neighbor', distribution='categorical', gaussian_dimensions=[2])
Summarize_EstimatedMI_with_TrueMI(file_path='.\\result\\compare_estimatedMI_with_trueMI\\gaussian_categorical\\estimatedMI_with_trueMI.csv', method='Mine_Net4', distribution='categorical', gaussian_dimensions=[2])

hyperparameter_config = {
        'method': ['Mine_Net','Mine_Net4'],
        'distribution': ['categorical'],
        'gaussian_dimension': [2],
        'type': ['training','testing']
    }
keys, values = zip(*hyperparameter_config.items())
hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

file_paths = []
image_dict = '.\\result\\compare_estimatedMI_with_trueMI\\gaussian_categorical'
for i in range(len(hyperparameter_experiments)):
    key, value = zip(*hyperparameter_experiments[i].items())
    method = value[0]
    distribution = value[1]
    gaussian_dimension = value[2]
    type = value[3]
    file_paths = file_paths + [image_dict + '\\%s_%s_gaussian_dim%s_%s.png'%(method,distribution, gaussian_dimension,type)]

file_paths += [image_dict + '\\nearest_neighbor_categorical_gaussian_dim2.png']
    
SummarizeResult(result_type='image', file_paths=file_paths, figtitle="Fig7: Compare estimated and true MI between categorical and gaussian mixture")
```

<b><font size="3">2.2.5 Questions</font></b> 

When clustering metrics demonstrate that SCVI+MINE works better than SCVI, however, tsne plot of scVI+MINE seems worse than that of scVI. One reason could be tsne plots are not very accurate, after all it only shows two components of latent vector z's variation (check tsne's paper, and say this sentence in a professional way). The other reason could be that the clustering metrics used here are not appropriate. <font color=red>Which one is correct or is there other explanation???????? Maybe know about manifold learning clustering metrics could help?? Not using euclidean distance</font>

tsne is applied mainly in computer vision, and is introduced into computation biology due to single cell RNA seq research. <font color=red>susan talked something more here, what is it?</font>

In reality, usually the batch effect in the dataset is unknown. The independence between batch and latent factor z is a more important problem than the independence between library size and latent factor z. <font color=red>Why???</font>. Heejung explained batch s is not included in z, but l is???. <font color=red>Make sure again.</font>

Is deep learning only appropriate when dimension of input is at least 70? The answer from susan is no. Actually linear
regression, logistic regression etc can be also considered as a type of simple deep learning neural network. 
<font color=red>But how to consider linear regresion or logistic regression as deep learning neural network?Be clear about the details</font>


<a id='section10'></a>
<b><font size="+1">2.3 scVI+MI_Penalty</font></b>

<b><font size="3">2.3.1 Goal</font></b> 

Compare SCVI and SCVI with penalty of mutual information(MI) between nuisance factor and latent factor, using cluster metrics, and tsne plots as the comparison criteria.  

<b><font size="3">2.3.2 Design</font></b>

Apply SCVI, SCVI with penalty of mutual information(MI) to the datasets: MouseMarrow, pbmc, retina.
Use Mine_Net architecture to estimate the mutual information no matter whether the nuisance factor is batch or library size.
Although Nearest_neighbor method can get a good estimator for MI when nuisance factor is batch, the MI estimator can not be back-propogated because of the sorting and counting steps in the nearest-neighbor method. This means although the nearest-neighbor estimator of MI is added to the reconstruction loss, it can not have any influence to update the weights of the vae network. Fortunately, we can use Mine_Net architecture to estimate the MI even when the nuisance factor is batch, a categorical variable. The rational is that when batch is independent with latent vector, then the K-L divergence between $P(z|s=0)$ and $p(z|s=1)$ is zero. The more dependent batch and latent vector is, the larger the K-L divergence between $P(z|s=0)$ and $p(z|s=1)$ is. Therefore, we can use the K-L divergence between $P(z|s=0)$ and $p(z|s=1)$ as the penalty. (Refer to the introductory section for the mathematical equations)

Refer to Section 2.3 for the four clustering metrics.

<b><font size="3">a. MouseMarrow</font></b>

Refer to Section 2.3 for how to get the MouseMarrow dataset

Based on all the former experience (check ./error_part1_Tune_Hyperparameter_for_SCVI_MI.ipynb and the last commit version of Project1_Modify_SCVI.ipynb), I plan to use std_reconstloss as loss for vae, and std_MI for MI penalty, as the scale for reconstloss and MI are quite different, millions vs 1/10~1/1000. reconstloss = -ELBO

I answer the following questions:

Question1, what is the best hyperparameters (esp: lr, n_epochs, n_layer_encoder, n_layer_decoder) for scvi when std_reconstloss is the loss? And what is relationship between cluster metrics and reconstloss. Giving the best clustering metrics and lowest reconstloss is the standard to choose the best hyperparameters. Plots with clustering metrics vs reconstloss for every 10 epochs are provided, reconstloss is on the whole trainset or testset instead of a minibatch. The hyperparameter is as:

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
            'std': [True, False] # whether use standardized ELBO or not
Here the minibatch size for vae is 128.The result is stored in clustermetrics_vs_ELBO, the plot is in figure20. It can be seen that when n_layers_encoder=10, n_layers_decoder=10 or 2, lr=1e-3, and std=False, after several tens of epochs, all 4 clustering metrics and BE increase as reconstloss decrease. But there is not such relationship when std=True for the same architecture of deep neural network and learning rate. The reason could be when std=True, the learning process becomes slower, which can be seen from the minimal reconstloss reached when std=True, compared with when std=False, therefore what I can do next is try to increase the learning rate or n_epoches to see whether the minimal reconstloss and the clustering metrics for std=False can reach the same value as when std=True for the same other hyperparameters. I try the following hyperparameters:
     
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
Here the minibatch size for vae is 128. The result is stored in clustermetric_vs_ELBO2, the plot is in figure21. When lr=1, and 1e-1, ELBO will become nan, therefore produce invalid result. (Note: later on, I also try lr=5e-2 for 10 layer encoder, 2 layer decoder, ELBO also turns into nan). From figure11, higher learning rate(1e-2) is better than lower learning rate (1e-3), larger n_epochs (800) is better than smaller n_epochs (350), with the standard of smaller reconstloss and better clustering metrics. I want to see whether the ELBO and clustering metrics can be improved further, as there is not much space to increase the learning rate, I increase the n_epochs to 1200, and try the following hyperparameters:

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
The result is stored in clustermetric_vs_ELBO3, the plot is in figure22. Yes, n_epochs=1200 will produce better clustering metrics and lower reconstloss than n_epochs=800. But considering the computation time and the result, n_epochs = 800 is enough for later on analysis. 
            
Besides, combined the results in figure20, 21, 22, the first conclusion is that n_layer_decoder=10 will produce higher batch entropy and lower clustering metrics than n_layer_decoder=2, no matter n_layer_encoder=2 or 10 when lr=1e-2. This phenomenon is only obvious when n_epochs=1200 (Need to repeat to see it is not because of variation). Second, n_layer_encoder=10 will produce higher BE and lower clustering metrics than n_layer_encoder=2, no matter n_layer_decoder=10 or 2. It seems that when the network is deeper, no matter in encoder or decoder, BE will be higher and clustering metrics will be lower. I don't know whether it is because when network is deeper, the optimal is reached slower than when network is shallower. If that is the case, when we elongate the training process futher, the more deeper network may produce higher BE and higher clustering metrics. 

For later on analysis, I'll choose the hyperparameters for vae, because our goal is to increase BE with some penalty in clustering metrics, these combinations of hyperparameters produces lower BE and higher clustering metrics within limited n_epochs, especially when n_layers_encoder=2 and n_layers_decoder=2, 
 
            'n_layers_encoder': [2,10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [1e-2],
            'n_epochs': [800],
            'std': [True]


Question2, what is the parento front of std_MI and std_reconstloss using chebyshev optimization? Whether when the scale for MI ,denoted as $\lambda$, increases, std_reconstloss increases and std_MI decreases. What does the std_reconstloss increase and std_MI decrease mean in clustering metrics, does it mean better BE and worse clustering metrics.

First, I need to get the hyperparameter for MI_net (esp: adv_lr), and the minimum and maximum value of MI for MI standardization. The loss is torch.max((1-MI_Scale)*std_reconstloss, MI_Scale*MI). Here MI is not standardized, because I need first to figure out MI's minimum and maximum value based on the MI estimator of the minibatches over the training process. I use the following hyperparameter combinations: 

            'n_layers_encoder': [2,10],
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
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [350],
            'adv_epochs': [3],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [False],
            'taskid': [0, 10, 50, 80]
Here the minibatch size for vae is 128, the minibatch size for MINE net is 256, because when sample size is larger, MINE net will give a better estimator, and when sample size is larger, there is a lower chance to get a minibatch which is extremely uneven between the two batches. For MIScale, I try 0, and 0.9, because theorectically, when MIScale=0, it produces the maximum value of MI, when MIScale=0.9, it produces the minimum value of MI.

The result is stored in adv_lr_min_max_MI, and plot in figure 25. Note that the horizontal line in the plot means the configuration returns NaN. From figure 25, although when adv_lr=0.01 can give higher MI estimator, but it is rather unstable, and has convergent issues. I choose adv_lr-0.005. I choose 0.3, -0.3 as the max and min for MI, no matter the n_layer_encoder=2 or 10.

Next I try the following hyperparameter configurations to draw the std_MI vs std_reconstloss, here the loss is torch.max((1-MI_Scale)*std_reconstloss, MI_Scale*std_MI), here std_MI and std_reconstloss is on the whole trainset and testset instead of minibatch:
            
            'n_layers_encoder': [2,10],
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
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [350],
            'adv_epochs': [3],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True],
            'taskid': [0, 10, 50, 80]            
The minibatch size for vae is 128, the minibatch size for MINE net is 256. For each MIScale, I repeat 4 times. Therefore, get the std_MI vs std_reconstloss plot 4 times, to get a preliminary idea that whether I should choose the MIScale evenly from 0 to 1, or unevenly from 0 to 1. Here for standardize reconstloss, min and max value I use is 10000 and 30000. To standardize mutual information, min and max value I use is -0.3 and 0.3. 

The result is stored in stdMI_stdreconstloss, plot in Figure26. Unfortunately, I can not observe the trend that when MIScale increase, stdMI decrease and stdreconstloss increase. It could be because of convergence issue. I can try another learning rate for minenet. 

Remember that previously, there is a hyperparameter combination which gives me the trend that when MIScale increase, stdreconstloss increase, stdMI decrease. I repeat it again.

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-3],
            'adv_lr': [5e-4],
            'pre_n_epochs': [50],
            'n_epochs': [350],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [5],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'],
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]

The minibatch size for vae is 128, the minibatch size for MINE net is 256. For each MIScale, I repeat 10 times. Here to standardize reconstloss, min and max value I use is 10000 and 30000. To standardize mutual information, min and max value I use is -0.02 and 0.03. The result is stored in stdMI_stdreconstloss2, plot in Figure27. So we can observe the trend that when MIScale increase, stdMI decrease and stdreconstloss increase. But the problem for this result is although the trend is OK, when MIScale=0, the loss is just stdreconstloss, this gives bad clustering metrics. Considering about practical usage, I want to start from a good clustering metrics when MIScale=0.

Pay attention, for figure26, I use a new subset method for minibatch when calculating mutual information. Because when calculating mutual information, I need to get latent vector z from batch 0 and from batch 1 separately. Formerly and for figure 27, I used 
      #z_batch0_tensor = z[[i for i in range(len(batch_index_list)) if batch_index_list[i] == [0]], :]
      #z_batch1_tensor = z[[i for i in range(len(batch_index_list)) if batch_index_list[i] == [1]], :]
      #l_batch0_tensor = library[[i for i in range(len(batch_index_list)) if batch_index_list[i] == [0]], :]
      #l_batch1_tensor = library[[i for i in range(len(batch_index_list)) if batch_index_list[i] == [1]], :]
Now for figure26 and figure 28, I use the code:
      z_batch0_tensor = z[(Variable(torch.LongTensor([1])) - batch_index).squeeze(1).byte()]
      z_batch1_tensor = z[batch_index.squeeze(1).byte()]
      l_batch0_tensor = library[(Variable(torch.LongTensor([1])) - batch_index).squeeze(1).byte()]
      l_batch1_tensor = library[batch_index.squeeze(1).byte()]
      l_z_batch0_tensor = torch.cat((l_batch0_tensor, z_batch0_tensor), dim=1)
      l_z_batch1_tensor = torch.cat((l_batch1_tensor, z_batch1_tensor), dim=1)
In order to make sure that this code is OK, for figure28, I repeat the experiment in figure27 using the same hyperparameters, but use the new code in the inference.py and trainer.py for the adversarial training process, the result is stored in stdMI_stdreconstloss3, plot in Figure28. Compared with Figure27, it seems that the code is OK, I can still observe the trend that when MIScale increases, stdreconstloss increases, stdMI decreases. Therefore in figure26, when I can not observe the trend, maybe it is because of the hyperparameters. 

From now on, I'll use the code used for figure28 to subset latent vector z for batch 0 and batch 1. 

In order to optimize hyperparameters either for result in figure 26 and figure 27, I try to use a different standardization scheme or/and different learning rate for minenet.

For example, in order to optimize the result in figure26, I try the same hyperparameters as in figure26, but a different standardization scheme for both vae and Minenet, the hyperparameters are:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [350],
            'adv_epochs': [3],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The min and max to standardize reconstloss is 12000 and 18000 respectively. The min and max to standardize MI is -0.2 and 0.2 respectively. The result is stored in stdMI_stdreconstloss4, plot in figure29. Still the stdreconstloss changes dramatically, and when MIScale becomes large, stdMI even increases. Some jobs among the 100 jobs don't finish running on clusters, but I cancel them because the finished jobs have already demonstrated that I need to adjust the hyperparameters to make the trend of stdMI vs stdreconstloss to be better.

Moreover, to optimize the result in figure26, I also try to decrease the learning rate for minenet, the hyperparameters are:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [1e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [350],
            'adv_epochs': [3],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The learning rate for minenet has decreased from 5e-3 to 1e-3, the min and max to standardize reconstloss are 12000 and 18000, respectively. However, as lr for minenet decreases, the min and max to standardize MI has become -0.06 and 0.06. The result is stored in stdMI_stdreconstloss5, plot in figure30. Still the stdreconstloss changes dramatically, and when MIScale becomes large, stdMI even increases. Some jobs among the 100 jobs don't finish running on clusters, but I cancel them because the finished jobs have already demonstrated that I need to adjust the hyperparameters to make the trend of stdMI vs stdreconstloss to be better. When I checked the MI value for MIScale=0 in stdMI_stdreconstloss5, min of MI is about -0.03, max of MI is about 0.03 (in most cases).       

For figure29, to standardize MI, I use -0.2 and 0.2, actually when I checked the real MI value when MIScale=0 in stdMI_stdreconstloss4 directory, min of MI is -0.1, max of MI is 0.1. Therefore, I try to use the same hyperparameters as in figure29, but use a different min and max of MI:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [350],
            'adv_epochs': [3],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The min and max to standardize reconstloss is 12000 and 18000 respectively. The min and max to standardize MI used here is -0.07 and 0.07 respectively. The result is stored in stdMI_stdreconstloss6, plot in figure31. 

Also I try to dampen the minenet using a smaller pre_adv_epochs, and smaller adv_epochs, but adv_lr is still 5e-3 as in figure29:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [150],
            'adv_epochs': [1],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The min and max to standardize reconstloss is 12000 and 18000 respectively. The min and max to standardize MI used here is smaller than figure31, they are -0.03 and 0.03 respectively. The result is stored in stdMI_stdreconstloss7, plot in figure32. 

To my surprise, in check the true MI value in stdMI_stdreconstloss7, max of MI is even a little bigger than that in stdMI_stdreconstloss6. Therefore -0.03 and 0.03 are two narrow to standardize MI. I try the following hyperparameters, but use -0.03 and 0.06 to standardize MI, and the pre_adv_epochs changes to 100.:
           
            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The result is stored in stdMI_stdreconstloss8, plot in figure33. Still have the convergence issue at 0.9.

In order to decrease the stdMI so that stdreconstloss will increase less, I use -0.01 and 0.05 to standardize MI. The hyperparameters are the same as in figure33. They are:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The result is in stdMI_stdreconstloss9, plot in figure34. This time, convergence issue seems better, however, when MIScale=0, stdMI is even smaller than 0, and stdMI at MIScale=0 varies a lot. This may be because our learning rate for adv_lr is a bit high, the MI estimator for MI varies a lot, and the min for MI we use is -0.01, which has a relatively small absolute value, therefore, the stdMI is very sensitive to the variation in MI estimator.

There, I try to increase max to standardize MI, so that stdMI can be decreased, and stdreconstloss will increase less, while the absolute value for min becomes larger so that the stdMI doesn't vary so much, the hyperparameters are the same:
           
            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The min and max to standardize MI are -0.03 and 0.08. The result is stored in stdMI_stdreconstloss10, plot in figure35. Still not solved the former problem, when MIScale=0.8 and/or 0.9, there are convergence issue. 

Try the following hyperparameter:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(10)),
            'MIScale': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100], 
            'n_epochs': [700],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'], 
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]        
The min and max to standardize reconstloss changes to 12000, and 16000. The min and max to standardize MI is -0.02, and 0.06. The result is stored in stdMI_stdreconstloss11, plot in figure36. Figure36 have 6 repetitions out of 10 give a satisfactory image compared to 2 repetitions out out of 10 in figure 35. But in figure36, there are new problems, for repid=8, MIScale=0.5 even gives a smaller stdreconstloss than when MIScale=0, that is because I've decreased the max value used to standardize reconstructloss from 18000 to 16000, then the stdreconstloss is more sensitive to the variability of reconstloss.

I'll use the hyperparameters in figure35 or figure36 for larger scale of repetitions. But before that, I need to decrease the computation times. Now the time is much longer than standard SCVI alone.

I first tried the following hyperparameters to see  when MIScale=0, whether the pretrained-scvi and then trained std_scvi can have the same effects as scvi, the hyperparameters are:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(3)),
            'MIScale': [0],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100, 150, 200],
            'n_epochs': [200,300,400],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'],
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The result is stored in shorten_time directory, plot in figure37. It can be seen that all the configurations can recover the same effect as scvi. Therefore, pre_n_epochs=100, n_epochs=200 is the best one, because it is the shortest. Maybe we can decrease n_epochs even further, perhaps to 150 after checking the clustering metrics during training for config0.

Next I try to see under these pre_n_epochs and n_epochs (try both 150 and 200 for n_epochs), whether the trend that when MIScale increases, stdreconstloss increase, and stdMI decrease remains. The hyperparameters are:

           'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(5)),
            'MIScale': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100],
            'n_epochs': [150, 200],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'],
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
The result is stored in shorten_time1, and plot in figure38. It seems that when n_epochs=200, the result is better than when n_epochs=150. But the phenomenon that when MIScale increase, the stdreconstloss even decrease. This is because of the max value to standardize reconstloss is a little small.

Next, I tried the following hyperparameters (the same hyperparameters as above but n_epochs=200):

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(5)),
            'MIScale': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100],
            'n_epochs': [200],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'],
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
But try both 16500 and 17000 to standardize reconstloss. The min to standardize reconstloss is 12000. The min and max to standardize MI is -0.02 and 0.06. The result is stored in shorten_time2, plot in figure39. But when I checked the MI estimation when MIScale=0, the max is about 0.03, the min is about -0.03. 

Therefore, I tried the following hyperparameters:

            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'taskid': list(range(5)),
            'MIScale': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100],
            'n_epochs': [200],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'],
            'unbiased_loss': [True], 
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True]
Use 12000 and 17000 as min and max to standardize reconstloss. Use 0.045 and -0.015 as min and max to standardize MI. Results are stord in shorten_time3, plot in figure 40. It appears that this is not better than result in figure39 for when min and max to standardize reconstloss is 12000 and 17000, min and max to standardize MI is 0.06 and -0.02, respectively. 

To get the pareto frontier, I used the following hyperparameters:

       hyperparameter_config = {
            'n_layers_encoder': [10],
            'n_layers_decoder': [2],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.8],
            'lr': [1e-2],
            'adv_lr': [5e-3],
            'pre_n_epochs': [100],
            'n_epochs': [200],
            'nsamples_z': [200],
            'adv': [True],
            'Adv_Net_architecture': [[256] * 10],
            'pre_adv_epochs': [100],
            'adv_epochs': [1],
            'activation_fun': ['ELU'],  
            'unbiased_loss': [True],  
            'initial': ['xavier_normal'], 
            'adv_model' : ['MI'],
            'optimiser': ['Adam'],
            'adv_drop_out': [0.2],
            'std': [True],
            'max_reconst': [17000]
        }
To fully train MineNet, lr is 5e-3, epochs number is 400. For each repetition, run 10 MI_scales, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]. Run this for 100 repetitions. The result shows that the learning rate for the MineNet architecture is too high, with very positive and very negative values, and cause failure for nearly 50% repetitions. Because high MIScales like 0.8, 0.9 could have convergence issues, try to sample more MIScales between 0.7 and 0.9. And use the MI estimator not the std MI estimator for the fully trained network.

<b><font size="3">2.3.3 Code</font></b>

copy code/tune_hyperparameter_for_SCVI_MI.sh and code/tune_hyperparameter_for_SCVI_MI.py to the directory containing the code directory. Then in the directory containing code directory, submit tune_hyperparameter_for_SCVI_MI.sh for job submission which will run the tune_hyperparameter_for_SCVI_MI.py 108 times. For muris_tabula dataset and batch nuisance factor, the parameters for the main function in the tune_hyperparameter_for_SCVI_MI.py is taskid=0, dataset_name='muris_tabula', nuisance_variable='batch', MI_estimator='MINE_Net4_3', config_id=a number from range(0, 108). The result of different configurations are downloaded from spartan and stored in local computer.

<b><font size="3">2.3.4 Result</font></b>

Among the 108 configurations for SCVI+Mine_estimated_MI, only 23 configurations complete successfully, among which none gives a better BE clustering metric than SCVI alone with the same hyperparameter for the SCVI part as the SCVI+Mine_estimated_MI here. And the MI_loss of minibatch of size 256 during the whole training process is very small. Refer to figure 9.

Figure 10 shows when the epoch increases to 500 and 750, 44 among 216 configs run successfully. By analyzing figure 10, in the plot of the clustering metrics, there are cyclic phenomena for the right half of the plot, which is three consective configs giving drastically small clustering metrics, two consecutive configs give similar large clustering metrics compared with the three consecutive configs. When comparing their corresponding hyperparameters between the three consective configs and the two consective configs, I find that the main difference is the activation function. For the two consective configs, the activation fuction is either ReLU or leaky_ReLU, while for the three consecutive configs, the activation function is ELU. The MI_loss plots show that when the activation function is ELU, MI_loss of minibatch will decrease fast, however, when activation function is ReLU or leaky_ReLU, the MI_loss of minibatch will either remain almost stable or increases. The conclusion here is that activation function is important ELU vs ReLU/leaky_ReLU, however ReLU and leaky_ReLU are similar. When analyzing the three consective configs, another finding is that MINE_Net with 50 layers are almost similar as MINE_Net with 100 layers.

Then another phenomenon to notice is that the left half of the clustering metric plot doesn't have such cyclic phenomenon, when comparing the single peak in the left half and the two consecutive peaks of clustering metrics in the right half, the main difference is the adv_lr, from left, to right are 5e-6, 1e-8, 1e-10. Therefore, when adv_lr is smaller, the MINE_Net4 is less sensitive to initiation strategies.



```python
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
clustermetric_vs_ELBO(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset','testset'], config_numbers=16, hyperparameter_index=1)

file_paths = []
for config in range(16):
    for k in [0, 7]:
        if config==9 and k==7:
           one_file_path = 'result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\clustermetric_vs_ELBO\\scviconfig%s\\muris_tabula_batch_config%s_trainset_clustervsELBO_epoch100.png'%(config, config)
        else:
           one_file_path = 'result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\clustermetric_vs_ELBO\\scviconfig%s\\muris_tabula_batch_config%s_trainset_clustervsELBO_epoch%s.png'%(config, config, k*10) 
        if os.path.isfile(one_file_path):
            file_paths += [one_file_path]
       
SummarizeResult('image',file_paths,'Fig20: Clustering metrics of trainset vs reconstloss for different configs of scvi')
hyperparameters_dataframe=pd.read_csv('D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO/muris_tabula_batch_configs_hyperparameters.csv')
print(hyperparameters_dataframe)
```

```python
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
clustermetric_vs_ELBO(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO2/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO2/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset','testset'], config_numbers=32, hyperparameter_index=2, starting_epoch=10)
file_paths = []
for config in range(32):
    for k in [10]:
        one_file_path = 'result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\clustermetric_vs_ELBO2\\scviconfig%s\\muris_tabula_batch_config%s_trainset_clustervsELBO_epoch%s.png'%(config, config, k*10) 
        if os.path.isfile(one_file_path):
            file_paths += [one_file_path]
       
SummarizeResult('image',file_paths,'Fig21: Clustering metrics of trainset vs ELBO for different configs of scvi')
hyperparameters_dataframe=pd.read_csv('D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO2/muris_tabula_batch_configs_hyperparameters.csv')
print(hyperparameters_dataframe)
```

```python
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
clustermetric_vs_ELBO(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO3/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO3/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset','testset'], config_numbers=4, hyperparameter_index=3, starting_epoch=10)
file_paths = []
for config in range(4):
    for k in [10]:
        one_file_path = 'result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\clustermetric_vs_ELBO3\\scviconfig%s\\muris_tabula_batch_config%s_trainset_clustervsELBO_epoch%s.png'%(config, config, k*10) 
        if os.path.isfile(one_file_path):
            file_paths += [one_file_path]
       
SummarizeResult('image',file_paths,'Fig22: Clustering metrics of trainset vs ELBO for different configs of scvi')
hyperparameters_dataframe=pd.read_csv('D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/clustermetric_vs_ELBO3/muris_tabula_batch_configs_hyperparameters.csv')
print(hyperparameters_dataframe)
```

```python
#Figure25
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_adv_lr_min_max_MI(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/adv_lr_min_max_MI/',
                  results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/adv_lr_min_max_MI/',
                  dataset_name='muris_tabula', nuisance_variable='batch', n_layer_number=2, adv_lr_number=6, MIScale_number=2, repetition_id=[0,10,50,80])
```

```python
#Figure26
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
#choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], hyperparameter_config_index=1, 
#            adv='MI', n_layer_number=2, repetition_id=[0, 10, 50, 80])
#choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'], hyperparameter_config_index=1, 
#            adv='MI', n_layer_number=2, repetition_id=[0, 10, 50, 80])
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure27
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss2/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss2/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], hyperparameter_config_index=1, 
            adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss2/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss2/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'], hyperparameter_config_index=1, 
            adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss2/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure28
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss3/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss3/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], hyperparameter_config_index=1, 
            adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss3/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss3/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'], hyperparameter_config_index=1, 
            adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss3/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure29
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss4/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss4/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss4/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss4/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'], adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss4/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure30
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss5/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss5/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss5/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss5/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss5/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure31
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss6/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss6/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss6/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss6/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss6/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure32
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss7/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss7/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss7/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss7/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss7/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure33
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss8/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss8/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss8/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss8/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss8/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure34
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss9/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss9/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss9/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss9/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss9/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure35
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss10/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss10/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss10/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss10/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss10/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure36
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss11/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss11/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_layer_number=1, repetition_number=10)
choose_config(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss11/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss11/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_layer_number=1, repetition_number=10)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/stdMI_stdreconstloss11/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python

```

```python
#Figure38
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
#shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time1/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time1/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', n_epochs_number=2, repetition_number=5)
#shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time1/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time1/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', n_epochs_number=2, repetition_number=5)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time1/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure39
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
#shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time2/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time2/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', max_reconstloss_number=2, repetition_number=5)
#shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time2/',
#            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time2/',
#            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', max_reconstloss_number=2, repetition_number=5)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time2/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure40
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time3/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time3/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['trainset'], adv='MI', max_reconstloss_number=1, repetition_number=5)
shorten_time(input_dir_path='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time3/',
            results_dict='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time3/',
            dataset_name='muris_tabula', nuisance_variable='batch', Label_list=['testset'],adv='MI', max_reconstloss_number=1, repetition_number=5)
from IPython.display import Video
Video("./result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/shorten_time3/trainset/trainset_stdMI_stdreconstloss.mp4")
```

```python
#Figure40
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
pareto_front(input_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front/',
            output_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front/',
            rep_number=100, dataset_name='muris_tabula', nuisance_variable='batch', Label='trainset')
pareto_front(input_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front/',
            output_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front/',
            rep_number=100, dataset_name='muris_tabula', nuisance_variable='batch', Label='testset')
file_paths = []
for Label in ['trainset','testset']:
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front\\muris_tabula_batch_%s_pareto_front.png'%(Label)] 
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front\\muris_tabula_batch_%s_std_penalty_full_pareto_front.png'%(Label)]

SummarizeResult('image',file_paths,'Fig40: pareto_front')
```

```python
#Figure41
import os
%matplotlib inline
import itertools
import pandas as pd
exec(open('code\\SummarizeResult.py').read())
pareto_front(input_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
            output_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
            rep_number=100, dataset_name='muris_tabula', nuisance_variable='batch', Label='trainset')
pareto_front(input_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
            output_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
            rep_number=100, dataset_name='muris_tabula', nuisance_variable='batch', Label='testset')
stdMI_clustermetrics(input_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
                  output_dir='D:/UMelb/PhD_Projects/Project1_Modify_SCVI/result/tune_hyperparameter_for_SCVI_MI/muris_tabula/choose_config/pareto_front2/',
                  scales_list=[0, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], rep_number=100, dataset_name='muris_tabula', nuisance_variable='batch')
file_paths = []
for Label in ['trainset','testset']:
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front2\\muris_tabula_batch_%s_pareto_front.png'%(Label)] 
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front2\\muris_tabula_batch_%s_penalty_full_pareto_front.png'%(Label)]
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front2\\muris_tabula_batch_%s_stdMI_vs_clustermetrics.png'%(Label)]
    file_paths += ['result\\tune_hyperparameter_for_SCVI_MI\\muris_tabula\\choose_config\\pareto_front2\\muris_tabula_batch_%s_stdMI_vs_be.png'%(Label)]

SummarizeResult('image',file_paths,'Fig41: pareto_front')
```

<a id='section11'></a>
<b><font size="+1">2.4 automate the process for finding the hyperparameters of MI_penalized SCVI for different datasets</font></b>



<a id='section12'></a>
<b><font size="+1">2.5 scVI and scVI+MI_Penalty comparison for downstream analysis</font></b>




<a id='section13'></a>
<b><font size="+2">3. Discussion</font></b>




<a id='section14'></a>
<b><font size="+2">4. Reference</font></b>



<a id='section15'></a>
<b><font size="+2">3. To Does</font></b>

<b><font size="+1">3.1 Tune Hyperparameter For MineNet</font></b>

Carry out the monte carlo experiment for all the 20 configurations, after that, we can think of improving the architecture of the neural network for MineNet including the activation function etc.

Try to apply the adversarial network architecture between SCVI and MINE

<b><font size="+1">3.2 Difference between the estimated lower bound and true mutual information</font></b>

Deep neural network is used to estimate the lower bound for mutual information between two continuous random variables. But the question is whether the deep neural network can really estimate the lower bound precisely and how close is the lower bound to the true mutual information. We can minimize the mutual information by minimizing the lower bound only when the lower bound is very close to the true mutual inforamtion.

Also check how the mutual inforamtion network estimator works between a categorical random variable and continuous random variable.

<b><font size="+1">3.3 Datasets</font></b>

Find real data set with obvious batch effect. Because the pbmc dataset used now do not have obvious batch effect. The dataset  ,harmonizing datasets with different composition of cell types,used in the paper [Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models](https://www.biorxiv.org/content/10.1101/532895v1) could be a resource. And another thing is to simulate data set with obvious batch effect from ZINB model.


<a id='section16'></a>
<b><font size="+2">4. Useful Tools and Information</font></b>

<b><font size="+1">4.1 Version control for jupyter notebook</font></b>

[how-to-version-control-jupyter](https://nextjournal.com/schmudde/how-to-version-control-jupyter)

[jupytext](https://github.com/mwouts/jupytext/blob/master/README.md)

<b><font size="+1">4.2 Deep Learning</font></b>

Tensor is a generalization of matrix with one obvious difference: a tensor is a mathematical entity that lives in a structure and interacts with other mathematical entities. If one transforms the other entities in the structure in a regular way, then the tensor must obey a related transformation rule.
Tensorflow vs Pytorch (which one is better for deep learning?)

Generally, to tune deep learning network:To find the best deep learning network, factors to consider are number of nodes in latent layer (width), number of hidden layers (depth), learning rate, optimizer, regularization, activation function according to [the book](https://www.deeplearningbook.org/) and the [deep learning specification](https://www.coursera.org/specializations/deep-learning?) on coursera. Write down more details here. Do we need to consider all the factors to tune the MINE deep learning network? Yes, eventually, we need to consider all the factors, with some clever strategy illustrated in [this paper](https://deepmind.com/blog/population-based-training-neural-networks/) 

<b><font size="+1">4.3 Courses on Deep Learning</font></b>

[Stanford University CS231n, Spring 2017](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

<b><font size="+1">4.4 Peek into the black box of deep learning </font></b>

[deep learning in biomedical image processing](http://biomedicalcomputationreview.org/content/deep-learning-and-future-%E2%80%A8biomedical-image-analysis)

Researchers are working on ways of peeking inside the models to understand how they select discriminant features. For example, a group of Stanford graduate students led by Avanti Shrikumar, a PhD candidate in computer science, recently developed an algorithm called DeepLIFT that attempts to determine which features are important by analyzing the activity of a model’s neurons when they are exposed to data. A team of engineers at the Israel-Technion Institute of Technology have devised a method of visualizing the neural activity of a network that resembles what one sees in fMRI of the human brain. And Rubin recently published a paper in which he and his colleagues trained a CNN to distinguish between benign and malignant breast tumors, and then used a visualization algorithm, called Directed Dream, to heighten and exaggerate specific details in order to maximize the images’ scores as either benign or malignant. The resulting “CNN-based hallucinations” effectively show how the CNN 

<b><font size="+1">4.5 Fairness of machine learning</font></b>

[Link to blog about fairness of machine learning](https://blog.godatadriven.com/fairness-in-ml)

<b><font size="+1">4.6 Initialization of weight for deep neural network</font></b>

[Link to weight initialization](https://www.deeplearning.ai/ai-notes/initialization/)

<b><font size="+1">4.7 Use of retain_graph and detach() in pytorch</font></b>

[Link to detach() and retain_graph](https://blog.csdn.net/qxqsunshine/article/details/82973979)





<!-- #region -->
keywords for searching: invari* + deep learning/deep neural network/deep network, high-level feature + learning, representation learning/learning representation, 
No review paper about invariance of deep learning ???

Goodfellow et al.([2009](https://ai.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf)) showed that both stacked autoencoder networks and convolutional deep belief networks (CDBNs) enjoy increasing invariance with depth even to complex 3-D out-of-plane rotation of natural images and natural video sequences, although the effects of depth in the two cases are different. Their observations supports the common view that invariances to minor shifts, rotations and deformations are learned in the lower layers, and combined in the higher layers to form progressively more invariant features. 

And the two aspects can be combined to achieve the best result.

[Measuring Invariances in Deep Networks](https://ai.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf)

Our proposed invariance measure is broadly applicable to evaluating many deep learning algorithms
for many tasks, but the present paper will focus on two different algorithms applied to computer
vision. First, we examine the invariances of <b>stacked autoencoder networks</b> [2]. These networks
were shown by Larochelle et al. [3] to learn useful features for a range of vision tasks; this suggests
that their learned features are significantly invariant to the transformations present in those tasks.
Unlike the artificial data used in [3], however, our work uses natural images and natural video
sequences, and examines more complex variations such as out-of-plane changes in viewing angle.
We find that when trained under these conditions, <b>stacked autoencoders</b> learn increasingly invariant
features with depth, but the effect of depth is small compared to other factors such as <b>regularization</b>.
Next, we show that convolutional deep belief networks (CDBNs) [5], which are hand-designed to be
invariant to certain local image translations, do enjoy dramatically increasing invariance with depth.
This suggests that there is a benefit to using deep architectures, but that mechanisms besides simple
stacking of autoencoders are important for gaining increasing invariance.

Another interesting finding is that by incorporating <b>sparsity</b>, networks can become more invariant. Refer to this paper about [induce sparsity to deep network](https://arxiv.org/pdf/1902.09574.pdf). Pay attention to section 2. How to understand? Difference between sparsity and drop out?

We also document that explicit approaches to achieving invariance such as <b>max-pooling</b> and weightsharing in CDBNs are currently successful strategies for achieving invariance. Refer to [this website](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) for CNN introduction. In it, 
For CNN, adding a Fully-Connected layer after the convolutional and pooling layers is a (usually) cheap way of learning non-linear combinations of the high-level features as represented by the output of the convolutional layer. The Fully-Connected layer is learning a possibly non-linear function in that space. Soft-Max activation is used. Why? Refer to [the link](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f) to compare different activation functions in deep learning. How to understand the problems of each activation function.

Our work also seeks to address the question: why are deep learning algorithms useful? Bengio and
LeCun gave a theoretical answer to this question, in which they showed that a deep architecture is
necessary to <b>represent many functions compactly</b> [1].

<b>restricted Boltzmann machine</b> 

<b>deep network vs deep belief network</b>



[Deep Learning of Invariant Features via Simulated
Fixations in Video](http://papers.nips.cc/paper/4730-deep-learning-of-invariant-features-via-simulated-fixations-in-video.pdf)

As a learning principle, sparsity is essential to understanding the statistics of natural images [2].
However, it remains unclear to what extent <b>sparsity and subspace pooling</b> [3, 4] could produce
invariance exhibited in higher levels of visual systems. Another approach to learning invariance is
temporal slowness [1, 5, 6, 7]. Experimental evidence suggests that high-level visual representations
become slow-changing and tolerant towards non-trivial transformations, by associating low-level
features which appear in a coherent sequence [5]

By stacking learning modules, we are able to learn
features that are increasingly invariant. Using <b>temporal slowness</b>, the first layer units become locally
translational invariant, similar to subspace or spatial pooling; the second layer units can then encode
more complex invariances such as out-of-plane transformations and non-linear warping

We have described an unsupervised learning algorithm for learning invariant features from video
using the temporal slowness principle. The system is improved by using simulated fixations and
smooth pursuit to generate the video sequences provided to the learning algorithm. We illustrate
by virtual of visualization and invariance tests, that the learned features are invariant to a collection
of non-trivial transformations. With concrete recognition experiments, we show that the features
learned from natural videos not only apply to still images, but also give competitive results on a
number of object recognition benchmarks. Since our features can be extracted using a feed-forward
neural network, they are also easy to use and efficient to compute.
<!-- #endregion -->

Deep learning has been widely used in many pattern recognition tasks, like image classification, object detection, and segmentation. During the process of deep learning, invariance could be imposed such that a pattern can still be correctly recognized when there are many confounding properties. For example, a cat in an image can still be classified correctly as a cat even when the image is rotated, enlarged or brightened. There are many researches, especially in computer vision field, about how to make deep learning network selects the complex, high level invariant features of the input, yet robust to irrelevant input transformations, which can be summarized in two aspects. 

One aspect to achieve invariability is to increase the amount of training data. Le et al.([2013](http://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf)) devoted enormous unlabeled image data and computation power to train a large neuron network with billions of parameters. The face feature detector learnt from the neuron network without labeling images as containing a face or not is robust not only to translation but also to scaling and out-of-plane rotation. Small training dataset can be expanded by generating new samples in the way of applying small random deformations to the original training examples, using deformations like rotations, scaling, translation or shearing, which are known not to change the target variables of interest. Ciresan et al. ([2010](https://arxiv.org/pdf/1003.0358.pdf)) applied large deep neural network on deformed MNIST digits and reached 0.35% classification error rate. Simard et al. ([2003](http://cognitivemedium.com/assets/rmnist/Simard.pdf)) trained convolutional neural network on MNIST English digit images with both affine and elastic deformations, and reached a record of 0.32% classification error rate. The other aspect to achieve invariability is to model general (e.g. a hierarchical organization of explanatory factors as the depth of neural network grows) or domain-specific prior information (e.g. the topological 2D structure of image data in computer vision, penalty) into deep learning network in the form of network structures before any data is fed in ([Bengio, Courville, & Vincent, 2012](https://arxiv.org/pdf/1206.5538.pdf)). <font color=red>can we learn nuisance factor effect from house keeping genes, and delete these effects for other genes???</font>

Although imposing invariance to the patterns identified by deep learning is extremely investigated in computer vision, it is also a general, or even more important problem in biology research when data is relatively expensive to obtain and difficult to deform to generate new data.

```python

```
