import os
from os import listdir
from os.path import isfile, join
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
MCA_meta_lung = MCA_meta[MCA_meta.Tissue.eq('Lung')]

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

MCA_Lung_subset = MCA_Lung.loc[:, ['GENE'] + MCA_meta_lung.loc[:,'Cell.name'].values.tolist()]
MCA_meta_lung_cellannotation = MCA_meta_lung.loc[:, ['Cell.name','Annotation']]
Consensus_celltype_MCA = pd.read_excel('./data/pareto_front_paretoMTL/TM_MCA_Lung/Consensus_celltype_MCA.xlsx', index_col=None, header=0)
MCA_meta_lung_cellannotation2 = MCA_meta_lung_cellannotation.merge(Consensus_celltype_MCA, how='left', left_on='Annotation', right_on='MCA_Type')
count = np.transpose(MCA_Lung_subset.iloc[:,1:].values)
genenames = np.array(MCA_Lung_subset.loc[:,'GENE'].values.tolist())
labels = MCA_meta_lung_cellannotation2.loc[:,'Consensus_Type']
cell_type, labels = np.unique(np.asarray(labels).astype('str'), return_inverse=True)



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


#download all the expression count file and meta file from the study titled
#'Study: Molecular specification of retinal cell types underlying central and peripheral vision in primates' on single cell portal website
#Like in the paper 'Deep learning enables accurate clustering with batch effect removal in single-cell RNA-seq analysis',
#we only focus on the bipolar cells


from os import walk
import pandas as pd

save_path = './data/pareto_front_paretoMTL/macaque_retina/'
retina_meta = pd.read_csv(save_path + 'Macaque_NN_RGC_AC_BC_HC_PR_metadata_3.txt', sep=',')
# bipolar cell types
cell_types_list = ['BB/GB*', 'DB1', 'DB2', 'DB3a', 'DB3b', 'DB4', 'DB5*', 'DB6', 'FMB', 'IMB', 'OFFx', 'RB']
retina_meta_fovea = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Fovea')]
retina_meta_per = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Periphery')]

dirs = []
for (dirpath, dirnames, filenames) in walk(save_path):
    dirs.extend(dirnames)
    break

tissue = 'fovea'
count_total = pd.DataFrame()
for dirname in dirs:
    files = []
    for (dirpath, dirnames, filenames) in walk(save_path + dirname + '/'):
        files.extend(filenames)
        break
    count_onefile = pd.DataFrame()
    if tissue == 'fovea':
        for filename in files:
            if 'fovea' in filename:
                chunksize = 10 ** 2
                for chunk in pd.read_csv(save_path + dirname + '/' + filename, chunksize=chunksize):
                    chunk_columns = chunk.columns.to_list()
                    break
                if any('Fovea4S' in s for s in chunk_columns):
                    chunk_columns = ['Unnamed: 0'] + [s.replace('Fovea4S', 'M4Fovea') for s in chunk_columns[1:]]
                subset_index = [s in ['Unnamed: 0'] + retina_meta_fovea.loc[:, 'NAME'].values.tolist() for s in chunk_columns ]
                for chunk in pd.read_csv(save_path + dirname + '/' + filename, chunksize=chunksize):
                    chunk_subset = chunk.loc[:, subset_index]
                    if chunk_subset.shape[1] == 1:
                        break
                    else:
                        count_onefile = pd.concat([count_onefile, chunk_subset], axis=0)
                if any('Fovea4S' in s for s in count_onefile.columns.to_list()):
                    count_onefile.columns = ['Unnamed: 0'] + [s.replace('Fovea4S', 'M4Fovea') for s in count_onefile.columns.to_list()[1:]]
    elif tissue == 'periphery':
        for filename in files:
            if 'per' in filename:
                chunksize = 10 ** 2
                for chunk in pd.read_csv(save_path + dirname + '/' + filename, chunksize=chunksize):
                    chunk_columns = chunk.columns.to_list()
                    break
                print(chunk_columns[0:3])
                if 'M4perCD73' in filename:
                    chunk_columns = ['Unnamed: 0'] + [s.replace('MacaqueCD73DP2', 'M4PerCD73') for s in chunk_columns[1:]]
                elif 'M5perCD73' in filename:
                    chunk_columns = ['Unnamed: 0'] + [s.replace('PerCd73', 'M1PerCD73') for s in chunk_columns[1:]]
                elif 'M5perPNA' in filename:
                    chunk_columns = ['Unnamed: 0'] + [s.replace('PerCd90PNAS1', 'M1CD90PNA_S1') for s in chunk_columns[1:]]
                elif 'M6perCD73' in filename:
                    chunk_columns = ['Unnamed: 0'] + [s.replace('PerCd73S3', 'M2PerCD73S1') for s in chunk_columns[1:]]
                    chunk_columns = ['Unnamed: 0'] + [s.replace('PerCd73S4', 'M2PerCD73S2') for s in chunk_columns[1:]]
                elif 'M6perMixed' in filename:
                    chunk_columns = ['Unnamed: 0'] + [s.replace('PerMixedS1', 'M2PerMixedS1') for s in chunk_columns[1:]]
                subset_index = [s in ['Unnamed: 0'] + retina_meta_per.loc[:, 'NAME'].values.tolist() for s in chunk_columns]
                for chunk in pd.read_csv(save_path + dirname + '/' + filename, sep=',', chunksize=chunksize):
                    chunk_subset = chunk.loc[:, subset_index]
                    if chunk_subset.shape[1] == 1:
                        break
                    else:
                        count_onefile = pd.concat([count_onefile, chunk_subset], axis=0)
                if count_onefile.shape[0] > 0:
                    chunk_columns_subset = [chunk_columns[k] for k in range(len(chunk_columns)) if subset_index[k]==True]
                    count_onefile.columns = chunk_columns_subset

    if count_total.shape[0] == 0 and count_onefile.shape[0] > 0:
        count_onefile = count_onefile.rename({'Unnamed: 0': 'GENE'}, axis='columns')
        count_total = count_onefile
    elif count_total.shape[0] > 0 and count_onefile.shape[0] > 0:
        count_onefile = count_onefile.rename({'Unnamed: 0': 'GENE'}, axis='columns')
        count_total = count_total.merge(count_onefile, how='inner', left_on='GENE', right_on='GENE')
    #for both fovea and periphery, the total UMI for all cells are larger than 100
    count_total.to_csv(save_path + "processed_{}_count_celltype_gene.csv.gz".format(tissue), index=False, compression="gzip")


from os import walk
import pandas as pd
save_path = './data/pareto_front_paretoMTL/macaque_retina/'
retina_meta = pd.read_csv(save_path + 'Macaque_NN_RGC_AC_BC_HC_PR_metadata_3.txt', sep=',')
# bipolar cell types
cell_types_list = ['BB/GB*', 'DB1', 'DB2', 'DB3a', 'DB3b', 'DB4', 'DB5*', 'DB6', 'FMB', 'IMB', 'OFFx', 'RB']
retina_meta_fovea = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Fovea')]
retina_meta_per = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Periphery')]
dirs = []
for (dirpath, dirnames, filenames) in walk(save_path):
    dirs.extend(dirnames)
    break
tissue = 'periphery'
count_total = pd.DataFrame()
for dirname in dirs:
    print('dir_name: {}'.format(dirname))
    files = []
    for (dirpath, dirnames, filenames) in walk(save_path + dirname + '/'):
        files.extend(filenames)
        break
    count_onefile = pd.DataFrame()
    if tissue == 'periphery':
        for filename in files:
            print('filename: {}'.format(filename))
            if 'per' in filename:
                chunksize = 10 ** 2
                for chunk in pd.read_csv(save_path + dirname + '/' + filename, chunksize=chunksize):
                    print(chunk.iloc[0:2,0:3])
                    chunk_columns = chunk.columns.to_list()
                    break
                chop_column_name = [s.split('_')[0] for s in chunk_columns]
                print(set(chop_column_name))
