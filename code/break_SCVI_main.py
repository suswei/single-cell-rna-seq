import os
import sys
from scvi.dataset import *
import numpy as np
import pandas as pd
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from scipy import sparse
import ntpath
import itertools


def LibrarySize_NoOfGene(trainer, pbmc_dataset, n_samples_tsne, Pbmc_Info_GeneCount, original_bool, save_path_original):

    full = trainer.create_posterior(trainer.model, pbmc_dataset, indices=None)

    latent, batch_indices, labels = full.get_latent(sample=True)
    latent, idx_t_sne = full.apply_t_sne(latent, n_samples=n_samples_tsne)

    batch_sample = (Pbmc_Info_GeneCount.iloc[:, 6]).values[idx_t_sne].ravel()

    celltypeindex_sample = (Pbmc_Info_GeneCount.iloc[:, 5]).values[idx_t_sne].ravel()

    library_size_sample = (Pbmc_Info_GeneCount.iloc[:, 0]).values[idx_t_sne].ravel()

    experssedgene_number_sample = (Pbmc_Info_GeneCount.iloc[:, 1]).values[idx_t_sne].ravel()

    percentage_sample = (Pbmc_Info_GeneCount.iloc[:, 2]).values[idx_t_sne].ravel()

    n_batch = full.gene_dataset.n_batches
    fig, axes = plt.subplots(3, 2, figsize=(14, 21))

    for i in range(n_batch):
        axes[0,0].scatter(latent[batch_sample == i, 0], latent[batch_sample == i, 1], label=str(i), s=8)
    axes[0,0].set_title("batch")
    axes[0,0].axis("off")
    axes[0,0].legend()

    cell_indices = celltypeindex_sample
    if hasattr(full.gene_dataset, 'cell_types'):
        plt_labels = full.gene_dataset.cell_types
    else:
        plt_labels = [str(i) for i in range(len(np.unique(cell_indices)))]

    for i, cell_type in zip(range(full.gene_dataset.n_labels), plt_labels):
        axes[0,1].scatter(latent[cell_indices == i, 0], latent[cell_indices == i, 1], label=cell_type, s=8)
    axes[0,1].set_title("cell type")
    axes[0,1].axis("off")
    axes[0,1].legend()

    ##use cyclic colormap "hsv" in matplotlib
    axes[1,0].scatter(latent[:, 0], latent[:, 1], c=library_size_sample, s=10, cmap='hsv')
    axes[1,0].set_title("library size")
    axes[1,0].axis("off")

    cax1 = fig.add_axes([0.47, 0.4, 0.015, 0.2])
    norm1 = mpl.colors.Normalize(vmin=library_size_sample.min(), vmax=library_size_sample.max())
    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=cm.hsv, norm=norm1, orientation='vertical')

    axes[1,1].scatter(latent[:, 0], latent[:, 1], c=experssedgene_number_sample, s=10, cmap='hsv')
    axes[1,1].set_title("Number of genes with non-zero count")
    axes[1,1].axis("off")

    cax2 = fig.add_axes([0.9, 0.4, 0.015, 0.2])
    norm2 = mpl.colors.Normalize(vmin=experssedgene_number_sample.min(), vmax=experssedgene_number_sample.max())
    cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cm.hsv, norm=norm2, orientation='vertical')

    axes[2,0].scatter(latent[:, 0], latent[:, 1], c=percentage_sample, s=10, cmap='hsv')
    axes[2,0].set_title("Percentage of the most expressed 100 genes")
    axes[2,0].axis("off")

    cax3 = fig.add_axes([0.47, 0.125, 0.015, 0.2])
    norm3 = mpl.colors.Normalize(vmin=percentage_sample.min(), vmax=percentage_sample.max())
    cb3 = mpl.colorbar.ColorbarBase(cax3, cmap=cm.hsv, norm=norm3, orientation='vertical')

    axes[2,1].axis('off')

    if original_bool == True:
        fig.savefig('./result/break_SCVI/BatchCellTypeLibrarySizeNoofGeneColoring_OriginalPbmc.png')
    else:
        fig.savefig(save_path_original.replace('data','result').replace('Modify', 'BatchCellTypeLibrarySizeNoofGeneColoring_').replace('.csv', '.png'))
    plt.close(fig)


def main(changed_property, jobid):
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

    if changed_property == 'Change_Library_Size':
        hyperparameter_config = {
            'n_layers_encoder': [1],
            'n_layers_decoder': [1],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.75],
            'lr': [1e-3],
            'n_epochs': [400],
            'frequency': [5],
            'n_samples_tsne': [1000],
            'batch': [0,1],
            'ratio': [1/10, 3/10, 3, 10]
        }
    elif changed_property == 'Change_Expressed_Gene_Number':
        hyperparameter_config = {
            'n_layers_encoder': [1],
            'n_layers_decoder': [1],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.75],
            'lr': [1e-3],
            'n_epochs': [400],
            'frequency': [5],
            'n_samples_tsne': [1000],
            'batch': [0,1],
            'ratio': [1/10, 3/10, 4/5]
        }
    elif changed_property == 'Change_Gene_Expression_Proportion':
        hyperparameter_config = {
            'n_layers_encoder': [1],
            'n_layers_decoder': [1],
            'n_hidden': [128],
            'n_latent': [10],
            'dropout_rate': [0.1],
            'reconstruction_loss': ['zinb'],
            'use_batches': [True],
            'use_cuda': [False],
            'train_size': [0.75],
            'lr': [1e-3],
            'n_epochs': [400],
            'frequency': [5],
            'n_samples_tsne': [1000],
            'batch': [0,1],
            'ratio': [2/10, 5/10, 2, 5],
            'proportion': [0.2, 0.4, 0.5]
        }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    data_save_path = './data/break_SCVI/%s' % (changed_property)
    result_save_path = './result/break_SCVI%s' % (changed_property)
    jobid = int(jobid)

    key, value = zip(*hyperparameter_experiments[jobid].items())
    n_layers_encoder = value[0]
    n_layers_decoder = value[1]
    n_hidden = value[2]
    n_latent = value[3]
    dropout_rate = value[4]
    reconstruction_loss = value[5]
    use_batches = value[6]
    use_cuda = value[7]
    train_size = value[8]
    lr = value[9]
    n_epochs = value[10]
    frequency = value[11]
    n_samples_tsne = value[12]
    batch = value[13]
    ratio = value[14]
    if changed_property == 'Change_Gene_Expression_Proportion':
       proportion = value[15]

    if changed_property == 'Change_Library_Size' or changed_property == 'Change_Expressed_Gene_Number':
       input_file_path = data_save_path + '/ModifyBatch%s_ratio%s.csv'%(batch, ratio)
    else:
        input_file_path = data_save_path + '/ModifyProportion%s_Batch%s_ratio%s.csv'%(batch, ratio)
    pbmc_dataset_original = PbmcDataset(save_path=os.path.dirname(input_file_path))

    pbmc_dataset = pbmc_dataset_original
    if jobid==0 and changed_property=='Change_Library_Size':
        Pbmc_Info_GeneCount_Original = pd.read_csv('./data/break_SCVI/Pbmc_CellName_Label_Batch_CellMetric_GeneCount.csv')
    Pbmc_Info_GeneCount = pd.read_csv(input_file_path)
    pbmc_dataset._X = sparse.csr_matrix(Pbmc_Info_GeneCount.iloc[:, 7:].values)

    if jobid==0 and changed_property=='Change_Library_Size':
       vae_original = VAE(pbmc_dataset_original.nb_genes, n_batch=pbmc_dataset_original.n_batches * use_batches)
       trainer_original = UnsupervisedTrainer(vae_original, pbmc_dataset_original, train_size=train_size, seed=3210, use_cuda=use_cuda, frequency=frequency)

    vae = VAE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches)
    trainer = UnsupervisedTrainer(vae, pbmc_dataset, train_size=train_size, seed=3210, use_cuda=use_cuda, frequency=frequency)

    if jobid==0 and changed_property=='Change_Library_Size':
       vae_file_path_original = './data/break_SCVI/original_pbmc.pk1'
    vae_file_path = input_file_path.replace('.csv', '.pk1')

    if jobid==0 and changed_property=='Change_Library_Size':
        if os.path.isfile(vae_file_path_original):
            trainer_original.model.load_state_dict(torch.load(vae_file_path_original))
            trainer_original.model.eval()
        else:
            trainer_original.train(n_epochs=n_epochs, lr=lr)
            torch.save(trainer_original.model.state_dict(), vae_file_path_original)

    if os.path.isfile(vae_file_path):
        trainer.model.load_state_dict(torch.load(vae_file_path))
        trainer.model.eval()
    else:
        trainer.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer.model.state_dict(), vae_file_path)

    if jobid == 0 and changed_property == 'Change_Library_Size':
        ll_train_set = trainer_original.history["ll_train_set"]
        ll_test_set = trainer_original.history["ll_test_set"]
        x = np.linspace(0, 500, (len(ll_train_set)))
        fig = plt.figure(figsize=(14, 7))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        plt.ylim(1150, 1600)
        plt.title("Blue for training error and orange for testing error")
        fig.savefig('./result/break_SCVI/Original_Pbmc_training_testing_error.png')
        plt.close(fig)

    ll_train_set = trainer.history["ll_train_set"]
    ll_test_set = trainer.history["ll_test_set"]
    x = np.linspace(0, 500, (len(ll_train_set)))
    fig = plt.figure(figsize=(14, 7))
    plt.plot(x, ll_train_set)
    plt.plot(x, ll_test_set)
    plt.ylim(1150, 1600)
    plt.title("Blue for training error and orange for testing error")
    fig.savefig(input_file_path.replace('data', 'result').replace('Modify','training_testing_error_').replace('.csv', '.png'))
    plt.close(fig)

    if jobid == 0 and changed_property == 'Change_Library_Size':
       memo_original = 'original_pbmc'
    memo = ntpath.basename(input_file_path).replace('Modify', '').replace('.csv', '')

    if jobid == 0 and changed_property == 'Change_Library_Size':
        be = trainer_original.train_set.entropy_batch_mixing()
        asw, nmi, ari, uca = trainer_original.train_set.clustering_scores()
        clustering_metrics = pd.DataFrame({'memo': [memo_original], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
        clustering_metrics.to_csv('./result/break_SCVI/ClusteringMetric_OriginalPbmc.csv', index=None, header=True)

    be = trainer.train_set.entropy_batch_mixing()
    asw, nmi, ari, uca = trainer.train_set.clustering_scores()
    clustering_metrics = pd.DataFrame({'memo': [memo], 'asw': [asw], 'nmi': [nmi], 'ari': [ari], 'uca': [uca], 'be': [be]})
    clustering_metrics.to_csv(input_file_path.replace('Modify', 'ClusteringMetrics_'),index=None, header=True)

    if jobid == 0 and changed_property == 'Change_Library_Size':
       LibrarySize_NoOfGene(trainer=trainer_original, pbmc_dataset=pbmc_dataset_original, n_samples_tsne=n_samples_tsne, Pbmc_Info_GeneCount=Pbmc_Info_GeneCount_Original, original_bool=True, save_path_original=input_file_path)
    LibrarySize_NoOfGene(trainer=trainer, pbmc_dataset=pbmc_dataset, n_samples_tsne=n_samples_tsne, Pbmc_Info_GeneCount=Pbmc_Info_GeneCount, original_bool=False, save_path_original=input_file_path)

# Run the actual program
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
