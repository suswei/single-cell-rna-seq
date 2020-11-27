from scvi.dataset.utils import get_matrix_from_dir,assign_label
from scvi.dataset.pbmc import PbmcDataset

import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset

dataset1 = PbmcDataset(filter_out_de_genes=False)
dataset1.update_cells(dataset1.batch_indices.ravel()==0)
dataset1.subsample_genes(dataset1.nb_genes)


count, geneid, cellid = get_matrix_from_dir('cite')
count = count.T.tocsr()
seurat = np.genfromtxt('../cite/cite.seurat.labels', dtype='str', delimiter=',')
cellid = np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 0, 1, 2, 3, 4, 5, 6]
labels = seurat[1:, 4]
cell_type = ['CD4 T cells', 'NK cells', 'CD14+ Monocytes', 'B cells','CD8 T cells', 'FCGR3A+ Monocytes', 'Other']
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)
set(dataset2.cell_types).intersection(set(dataset2.cell_types))

dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)


pbmc = PbmcDataset()
de_data  = pbmc.de_metadata
pbmc.update_cells(pbmc.batch_indices.ravel()==0)
# pbmc.labels = pbmc.labels.reshape(len(pbmc),1)

donor = Dataset10X('fresh_68k_pbmc_donor_a')
donor.gene_names = donor.gene_symbols
donor.labels = np.repeat(0,len(donor)).reshape(len(donor),1)
donor.cell_types = ['unlabelled']
all_dataset = GeneExpressionDataset.concat_datasets(pbmc, donor)

# Now resolve the Gene symbols to properly work with the DE
all_gene_symbols = donor.gene_symbols[
    np.array(
        [np.where(donor.gene_names == x)[0][0] for x in list(all_dataset.gene_names)]
    )]


import pandas as pd
import gzip
import zipfile
#get dataset1 from the paper 'An immune-cell signature of bacterial sepsis'
#data is SCP548 on Single Cell Portal-Broad institute website
scp548_meta = pd.read_csv('./data/Pbmc_SCP548/scp_meta.txt', sep='\t')
#subset the 'Control' cohort
scp548_meta_control = scp548_meta[scp548_meta.Cohort.eq('Control')]

scp548_genelist=[]
chunksize = 10 ** 2
for chunk in pd.read_csv('./data/Pbmc_SCP548/scp_gex_matrix.csv.gz', chunksize=chunksize):
    scp548_genelist += chunk.loc[:,'GENE'].values.tolist()

scp256_meta = pd.read_csv('./data/Pbmc_SCP256/alexandria_structured_metadata.txt', sep='\t')
#subset 'Pre-Infection' cells
scp256_meta_subset = scp256_meta[scp256_meta.TimePoint.eq('Pre-Infection')]
scp256_subset_cellid = scp256_meta_subset.loc[:,'CellID'].values.tolist()

scp256_raw_count = pd.read_csv('./data/Pbmc_SCP256/raw_counts_matrix_040220.txt.zip',header=0)
scp256_genelist = []
#get column index for 'Pre-Infection' cells


for record in scp256_raw_count.iloc[:,0].values.tolist():
    scp256_genelist += [record.split('\t')[0]]

#scp256_lognormal_count = pd.read_csv('./data/Pbmc_SCP256/log_normalized_matrix_012020.txt.gz', compression='gzip', header=0, error_bad_lines=False)
