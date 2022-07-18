============================================================================================================================================
Trade-off between conservation of biological variation and batch effect removal in deep generative modeling for single-cell transcriptomics
============================================================================================================================================
1. The research is built upon single cell Variational Inference (scVI). The inference/trainer.py, inference/inference.py, models/vae.py, models/modules.py in the scvi directory has been modified.
2. Refer to the code directory to reproduce all results in the research.


Quick Start
-----------

1. Install Python 3.6 or later. We use the Anaconda Python distribution.

2. Install PyTorch_. If you have an Nvidia GPU, be sure to install a version of PyTorch that supports it -- scVI runs much faster with a discrete GPU.

.. _PyTorch: http://pytorch.org

3. Install scVI through conda (``conda install scvi -c bioconda``) or through pip (``pip install scvi``). Alternatively, you may download or clone this repository and run ``python setup.py install``.

4. Briefly speaking, there are 3 steps to estimate a Pareto Front for Pareto MTL with MINE:

   a. Pretrain scVI and MINE neural networks separately:
   trainer_vae.pretrain_extreme_regularize_paretoMTL(pre_train=pre_train, pre_epochs=pre_epochs, pre_lr=pre_lr, pre_adv_epochs=pre_adv_epochs, pre_adv_lr=pre_adv_lr, path=save_path)

   b. Obtain the minimum and maximum value for the two objectives for standardization, and the two extreme points of the Pareto Front, :
   minibatch_loss_list = trainer_vae.pretrain_extreme_regularize_paretoMTL(path=save_path, weight=weight, extreme_points=extreme_points, lr=lr, adv_lr=adv_lr, epochs=epochs, adv_epochs=adv_epochs).
   When weight equals 0, we will get minimum and maximum value of the MINE,  and the extreme point which has the smallest MINE value.
   When weight equals 1, we will get minimum and maximum value of the generative loss, and the extreme point which has the smallest generative loss.

   c. Obtain K Pareto optimal points by running Pareto MTL with MINE K times with different preference vector each time:
   trainer_vae.pretrain_extreme_regularize_paretoMTL(path=save_path, lr=lr, adv_lr=adv_lr, paretoMTL=paretoMTL,
   obj1_max=obj1_max, obj1_min=obj1_min, obj2_max=obj2_max, obj2_min=obj2_min, epochs = epochs,
   adv_epochs=adv_epochs, n_tasks = n_tasks, npref = npref, pref_type=pref_type, pref_idx = pref_idx)

References
----------
Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong.
**"Pareto Multi-Task Learning."**
NeurIPS, 2019. `[pdf]`__

.. __: https://proceedings.neurips.cc/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf

Romain Lopez, Jeffrey Regier, Michael Cole, Michael I. Jordan, Nir Yosef.
**"Deep generative modeling for single-cell transcriptomics."**
Nature Methods, 2018. `[pdf]`__

.. __: https://rdcu.be/bdHYQ

