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


def Break_SCVI(input_file_paths_prefix: chr = './data/break_SCVI/Change_Library_Size/2019-04-05/Pbmc_CellInfo_GeneCount*',
               use_batches: bool = True, train_size: float = 0.75, use_cuda: bool = False, frequency: int = 5,
               n_epochs: int = 400, lr: float = 1e-3, n_samples_tsne: int = 1000):

    all_header_files = glob2.glob(input_file_paths_prefix+'.csv')
    pbmc_dataset_original = PbmcDataset(save_path=os.path.dirname(input_file_paths_prefix))

    clustering_metrics = pd.DataFrame(columns=['memo', 'asw', 'nmi', 'ari', 'uca', 'be'])

    for index in range(len(all_header_files)+1):
        pbmc_dataset = pbmc_dataset_original
        if index < len(all_header_files):
            Pbmc_Info_GeneCount = pd.read_csv(all_header_files[index])
            pbmc_dataset._X = sparse.csr_matrix(Pbmc_Info_GeneCount.iloc[:, 7:].values)

        vae = VAE(pbmc_dataset.nb_genes, n_batch=pbmc_dataset.n_batches * use_batches)

        trainer = UnsupervisedTrainer(vae, pbmc_dataset, train_size=train_size, use_cuda=use_cuda, frequency=frequency)

        if index < len(all_header_files):
            vae_file_path = all_header_files[index].replace('.csv', '.pk1')
        else:
            vae_file_path = os.path.dirname(input_file_paths_prefix) + '/original_pbmc.pk1'

        if os.path.isfile(vae_file_path):
            trainer.model.load_state_dict(torch.load(vae_file_path))
            trainer.model.eval()
        else:
            trainer.train(n_epochs=n_epochs, lr=lr)
            torch.save(trainer.model.state_dict(), vae_file_path)

        ll_train_set = trainer.history["ll_train_set"]
        ll_test_set = trainer.history["ll_test_set"]
        x = np.linspace(0, 500, (len(ll_train_set)))

        fig = plt.figure(figsize=(14, 7))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        plt.ylim(1150, 1600)
        plt.title("Blue for training error and orange for testing error")
        if index < len(all_header_files):
            fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount','Pbmc_training_testing_error').replace('.csv', '.png'))
        else:
            path1 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_training_testing_error.png'
            fig.savefig(path1.replace('data','result'))
        plt.close(fig)

        full = trainer.create_posterior(trainer.model, pbmc_dataset, indices=None)

        latent, batch_indices, labels = full.get_latent(sample=True)
        latent, idx_t_sne = full.apply_t_sne(latent, n_samples=n_samples_tsne)

        batch_sample = (Pbmc_Info_GeneCount.iloc[:, 6]).values[idx_t_sne].ravel()

        celltypeindex_sample = (Pbmc_Info_GeneCount.iloc[:, 5]).values[idx_t_sne].ravel()

        library_size_sample = (Pbmc_Info_GeneCount.iloc[:, 0]).values[idx_t_sne].ravel()

        experssedgene_number_sample = (Pbmc_Info_GeneCount.iloc[:, 1]).values[idx_t_sne].ravel()

        percentage_sample = (Pbmc_Info_GeneCount.iloc[:, 2]).values[idx_t_sne].ravel()

        n_batch = full.gene_dataset.n_batches
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for i in range(n_batch):
            axes[0].scatter(latent[batch_sample == i, 0], latent[batch_sample == i, 1], label=str(i), s=8)
        axes[0].set_title("batch coloring")
        axes[0].axis("off")
        axes[0].legend()

        cell_indices = celltypeindex_sample
        if hasattr(full.gene_dataset, 'cell_types'):
            plt_labels = full.gene_dataset.cell_types
        else:
            plt_labels = [str(i) for i in range(len(np.unique(cell_indices)))]

        for i, cell_type in zip(range(full.gene_dataset.n_labels), plt_labels):
            axes[1].scatter(latent[cell_indices == i, 0], latent[cell_indices == i, 1], label=cell_type, s=8)
        axes[1].set_title("label coloring")
        axes[1].axis("off")
        axes[1].legend()
        if index < len(all_header_files):
            fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount', 'Pbmc_batch_labels').replace('.csv', '.png'))
        else:
            path2 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_batch_labels.png'
            fig.savefig(path2.replace('data', 'result'))
        plt.close(fig)

        ##use diverging colormap "PiYG" and cyclic colormap "hsv" in matplotlib
        for colormap in ["PiYG","hsv"]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].scatter(latent[:, 0], latent[:, 1], c=library_size_sample, s=10, cmap=colormap)
            axes[0].set_title("library size")

            cax1 = fig.add_axes([0.47, 0.125, 0.015, 0.76])
            norm1 = mpl.colors.Normalize(vmin=library_size_sample.min(), vmax=library_size_sample.max())
            cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=colormap, norm=norm1, orientation='vertical')

            axes[1].scatter(latent[:, 0], latent[:, 1], c=experssedgene_number_sample, s=10, cmap=colormap)
            axes[1].set_title("Number of genes expressed")

            cax2 = fig.add_axes([0.9, 0.125, 0.015, 0.76])
            norm2 = mpl.colors.Normalize(vmin=experssedgene_number_sample.min(), vmax=experssedgene_number_sample.max())
            cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=colormap, norm=norm2, orientation='vertical')
            if colormap == "PiYG":
                if index < len(all_header_files):
                    fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount','Pbmc_librarysize_geneNo_divergent-colormap').replace('.csv', '.png'))
                else:
                    path3 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_librarysize_geneNo_divergent-colormap.png'
                    fig.savefig(path3.replace('data', 'result'))
            else:
                if index < len(all_header_files):
                    fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount', 'Pbmc_librarysize_geneNo_cyclic-colormap').replace('.csv', '.png'))
                else:
                    path4 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_librarysize_geneNo_cyclic-colormap.png'
                    fig.savefig(path4.replace('data', 'result'))
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            axes[0].scatter(latent[:, 0], latent[:, 1], c=percentage_sample, s=10, cmap=colormap)
            axes[0].set_title("Percentage of the most expressed 100 genes")

            cax3 = fig.add_axes([0.47, 0.125, 0.015, 0.76])
            norm3 = mpl.colors.Normalize(vmin=percentage_sample.min(), vmax=percentage_sample.max())
            cb3 = mpl.colorbar.ColorbarBase(cax3, cmap=colormap, norm=norm3, orientation='vertical')

            axes[1].axis('off')

            if colormap == "PiYG":
                if index < len(all_header_files):
                    fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount','Pbmc_percentage_divergent-colormap').replace('.csv', '.png'))
                else:
                    path5 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_percentage_divergent-colormap.png'
                    fig.savefig(path5.replace('data', 'result'))
            else:
                if index < len(all_header_files):
                    fig.savefig(all_header_files[index].replace('data', 'result').replace('Pbmc_CellInfo_GeneCount', 'Pbmc_percentage_cyclic-colormap').replace('.csv', '.png'))
                else:
                    path6 = os.path.dirname(input_file_paths_prefix) + '/original_Pbmc_percentage_cyclic-colormap.png'
                    fig.savefig(path6.replace('data', 'result'))
            plt.close(fig)

        if index < len(all_header_files):
            memo = ntpath.basename(all_header_files[index]).replace('Pbmc_CellInfo_GeneCount_', '').replace('.csv', '')
        else:
            memo = 'original_pbmc'

        asw, nmi, ari, uca = trainer.train_set.clustering_scores()
        be = trainer.train_set.entropy_batch_mixing()
        clustering_metrics = clustering_metrics.append(pd.DataFrame({'memo':[memo],'asw':[asw],'nmi':[nmi],'ari':[ari],'uca':[uca],'be':[be]}.items(), columns=['memo', 'asw', 'nmi', 'ari', 'uca', 'be']))

    clustering_metrics.to_csv(os.path.dirname(input_file_paths_prefix)+'/clustering_metrics_train_set.csv', index = None, header=True)



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


Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Library_Size/2019-04-05/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Library_Size/2019-04-14/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Expressed_Gene_Number/2019-04-14/Pbmc_CellInfo_GeneCount*')
Break_SCVI(input_file_paths_prefix = './data/Break_SCVI/Change_Gene_Expression_Proportion/2019-05-01/Pbmc_CellInfo_GeneCount*')
