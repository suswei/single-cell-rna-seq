import os
import pandas as pd
from scvi.dataset.dataset10X import Dataset10X
import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset

cell_types = np.array(["cd4_t_helper", "regulatory_t", "naive_t", "memory_t", "cytotoxic_t", "naive_cytotoxic",
                       "b_cells", "cd34", "cd56_nk", "cd14_monocytes"])
cell_type_name = np.array(["CD4 T cells", "CD4 T cells Regulatory", "CD4 T cells Naive", "CD4 Memory T cells", "CD8 T cells", "CD8 T cells Naive",
                       "B cells", "CD34 cells", "NK cells", "CD14+ Monocytes"])

datasets = []
for i,cell_type in enumerate(cell_types):
    dataset = Dataset10X(cell_type, save_path='data/')
    dataset.cell_types = np.array([cell_type_name[i]])
    dataset.labels = dataset.labels.astype('int')
    dataset.subsample_genes(dataset.nb_genes)
    dataset.gene_names = dataset.gene_symbols
    datasets += [dataset]

pure = GeneExpressionDataset.concat_datasets(*datasets, shared_batches=True)

donor = Dataset10X('fresh_68k_pbmc_donor_a')
donor.gene_names = donor.gene_symbols

if not os.path.isfile('data/10X/fresh_68k_pbmc_donor_a/68k_pbmc_barcodes_annotation.tsv'):
    import urllib.request
    annotation_url = 'https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv'
    urllib.request.urlretrieve(annotation_url, 'data/10X/fresh_68k_pbmc_donor_a/68k_pbmc_barcodes_annotation.tsv')

annotation = pd.read_csv('data/10X/fresh_68k_pbmc_donor_a/68k_pbmc_barcodes_annotation.tsv',sep='\t')
cellid1 = donor.barcodes
temp = cellid1.join(annotation)
assert all(temp[0]==temp['barcodes'])

donor.cell_types,donor.labels = np.unique(temp['celltype'],return_inverse=True)

donor.labels = donor.labels.reshape(len(donor.labels),1)
donor.cell_types = np.array([ 'CD14+ Monocytes','B cells','CD34 cells', 'CD4 T cells','CD4 T cells Regulatory',
                             'CD4 T cells Naive','CD4 Memory T cells','NK cells',
                            'CD8 T cells',  'CD8 T cells Naive', 'Dendritic'])
gene_dataset = GeneExpressionDataset.concat_datasets(donor, pure)


#Cells in Lung tissue from Muris_tabula FACS, and MCA DGE data
MCA_meta = pd.read_csv('./data/pareto_front_paretoMTL/TM_MCA_Lung/MCA_CellAssignments.csv')

for id in range(3):
    id += 1
    MCA_Lung_one = pd.DataFrame()
    chunksize = 10 ** 2
    for chunk in pd.read_csv('./data/pareto_front_paretoMTL/TM_MCA_Lung/MCA_500more_dge/Lung{}_dge.txt.gz'.format(id), compression='gzip', sep=' ', header=0, chunksize=chunksize):
        chunk['GENE'] = chunk.index
        first_col = chunk.pop('GENE')
        chunk.insert(0, 'GENE', first_col)
        MCA_Lung_one = pd.concat([MCA_Lung_one, chunk],axis=0)
    if id == 1:
        MCA_Lung = MCA_Lung_one
    else:
        MCA_Lung = MCA_Lung.merge(MCA_Lung_one, how='inner', left_on='GENE', right_on='GENE')

import pandas as pd
import pickle
#get dataset1 from the paper 'An immune-cell signature of bacterial sepsis'
#data is SCP548 on Single Cell Portal-Broad institute website
scp548_meta = pd.read_csv('./data/Pbmc_SCP548/scp_meta.txt', sep='\t')
#subset the 'Control' cohort
scp548_meta_control = scp548_meta[scp548_meta.Cohort.eq('Control')]

'''
scp548_genelist=[]
chunksize = 10 ** 2
for chunk in pd.read_csv('./data/Pbmc_SCP548/scp_gex_matrix.csv.gz', chunksize=chunksize):
    scp548_genelist += chunk.loc[:,'GENE'].values.tolist()
with open('./data/Pbmc_SCP548/scp548_genelist.pkl', 'wb') as f:
        pickle.dump({'gene':scp548_genelist}, f)
'''
with open('./data/Pbmc_SCP548/scp548_genelist.pkl', 'rb') as f:
    scp548_genelist = pickle.load(f)['gene']

scp256_meta = pd.read_csv('./data/Pbmc_SCP256/alexandria_structured_metadata.txt', sep='\t')
#subset 'Pre-Infection' cells
scp256_meta_subset = scp256_meta[scp256_meta.TimePoint.eq('Pre-Infection')]
scp256_subset_cellid = scp256_meta_subset.loc[:,'CellID'].values.tolist()

chunksize = 10 ** 2
scp256_subset_gene_cellid = pd.DataFrame()
for chunk in pd.read_csv('./data/Pbmc_SCP256/log_normalized_matrix_012020.txt.gz', compression='gzip', header=0, sep='\t', chunksize=chunksize):
    chunk_subset = chunk[chunk.GENE.isin(scp548_genelist)].loc[:, ['GENE']+ scp256_subset_cellid]
    scp256_subset_gene_cellid = pd.concat([scp256_subset_gene_cellid, chunk_subset],axis=0)

#change row for a cell, column for a gene for scp256
scp256_subset_X = np.transpose(scp256_subset_gene_cellid.iloc[:,1:].values)
#exp(element)-1 because in scVI, x will be transformed into log(1+x)
scp256_subset_X = np.exp(scp256_subset_X) - 1
scp256_subset_gene = scp256_subset_gene_cellid.loc[:,'GENE'].values.tolist()
scp256_subset_cellid = scp256_subset_gene_cellid.columns[1:].to_list()

#subset scp548
scp548_subset_cellname = scp548_meta_control.loc[:,'NAME'].values.tolist()

chunksize = 10 ** 2
for chunk in pd.read_csv('./data/Pbmc_SCP548/scp_gex_matrix.csv.gz', chunksize=chunksize):
    scp548_cellname = chunk.columns.to_list()
    break
#not all cells in scp548_subset_cellname are included in scp548_cellname
scp548_subset_cellname2 = [k for k in scp548_subset_cellname if k in scp548_cellname]

scp548_subset_gene_cellid = pd.DataFrame()
for chunk in pd.read_csv('./data/Pbmc_SCP548/scp_gex_matrix.csv.gz', chunksize=chunksize):
    chunk_subset = chunk[chunk.GENE.isin(scp256_subset_gene)].loc[:, ['GENE']+scp548_subset_cellname2]
    scp548_subset_gene_cellid = pd.concat([scp548_subset_gene_cellid, chunk_subset],axis=0)

#change row for a cell, column for a gene for scp548
scp548_subset_X = np.transpose(scp548_subset_gene_cellid.iloc[:,1:].values)
#exp(element)-1 because in scVI, x will be transformed into log(1+x)
scp548_subset_X = np.exp(scp548_subset_X) - 1
scp548_subset_gene = scp548_subset_gene_cellid.loc[:,'GENE'].values.tolist()
scp548_subset_cellid = scp548_subset_gene_cellid.columns[1:].to_list()

