import pickle
import pandas as pd
from scvi.dataset.dataset import GeneExpressionDataset
import numpy as np
from scipy.sparse import csr_matrix

class Pbmc_SCP256_SCP548(GeneExpressionDataset):
    def __init__(self, dataname, save_path='./data/pareto_front_paretoMTL/Pbmc_SCP256_SCP548/', tissue='PBMC'):
        self.save_path = save_path
        self.dataname = dataname
        self.tissue = tissue
        '''
        self.urls =  ['https://singlecell.broadinstitute.org/single_cell/data/public/SCP256/integrated-single-cell-analysis-of-multicellular-immune-dynamics-during-hyper-acute-hiv-1-infection?filename=alexandria_structured_metadata.txt',
                'https://singlecell.broadinstitute.org/single_cell/data/public/SCP256/integrated-single-cell-analysis-of-multicellular-immune-dynamics-during-hyper-acute-hiv-1-infection?filename=log_normalized_matrix_012020.txt.gz',
                'https://singlecell.broadinstitute.org/single_cell/data/public/SCP548/an-immune-cell-signature-of-bacterial-sepsis-patient-pbmcs?filename=scp_meta.txt',
                'https://singlecell.broadinstitute.org/single_cell/data/public/SCP548/an-immune-cell-signature-of-bacterial-sepsis-patient-pbmcs?filename=scp_gex_matrix.csv.gz']
        self.download_names = ['alexandria_structured_metadata.txt','log_normalized_matrix_012020.txt.gz','scp_meta.txt','scp_gex_matrix.csv.gz']
        self.download()
        '''
        count, labels, cell_type, gene_names = self.preprocess()
        count = csr_matrix(count)
        super(Pbmc_SCP256_SCP548, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix_muris_tabula(
                count, labels=labels), gene_names=np.char.upper(gene_names), cell_types=cell_type)
    def preprocess(self):
        '''
        scp548_genelist=[]
        chunksize = 10 ** 2
        for chunk in pd.read_csv(self.save_path + '/scp_gex_matrix.csv.gz', chunksize=chunksize):
             scp548_genelist += chunk.loc[:,'GENE'].values.tolist()
        with open(self.save_path + '/scp548_genelist.pkl', 'wb') as f:
                 pickle.dump({'gene':scp548_genelist}, f)

        with open(self.save_path + '/scp548_genelist.pkl', 'rb') as f:
            scp548_genelist = pickle.load(f)['gene']

        if self.dataname =='SCP256':
            scp256_meta = pd.read_csv(self.save_path + '/alexandria_structured_metadata.txt', sep='\t')
            # subset 'Pre-Infection' cells
            scp256_meta_subset = scp256_meta[scp256_meta.TimePoint.eq('Pre-Infection')]
            scp256_subset_cellid = scp256_meta_subset.loc[:, 'CellID'].values.tolist()

            chunksize = 10 ** 2
            scp256_subset_gene_cellid = pd.DataFrame()
            for chunk in pd.read_csv(self.save_path + '/log_normalized_matrix_012020.txt.gz', compression='gzip',
                                     header=0, sep='\t', chunksize=chunksize):
                chunk_subset = chunk[chunk.GENE.isin(scp548_genelist)].loc[:, ['GENE'] + scp256_subset_cellid]
                scp256_subset_gene_cellid = pd.concat([scp256_subset_gene_cellid, chunk_subset], axis=0)

            # change row for a cell, column for a gene for scp256
            count = np.transpose(scp256_subset_gene_cellid.iloc[:, 1:].values)
            # exp(element)-1 because in scVI, x will be transformed into log(1+x)
            count = np.exp(count) - 1
            genenames = scp256_subset_gene_cellid.loc[:, 'GENE'].values.tolist()
            scp256_subset_dataframe = pd.DataFrame(count, columns=genenames)
            labels = scp256_meta_subset.loc[:, 'cell_type__ontology_label'].values.tolist()
            scp256_subset_dataframe = pd.concat([pd.DataFrame.from_dict({'Cell_Type': labels}), scp256_subset_dataframe], axis=1)
            scp256_subset_dataframe.to_csv(self.save_path + "processed_scp256_count_celltype_gene.csv.gz", index=False, compression="gzip")
            '''

        if self.dataname == 'SCP256':
            chunksize = 10 ** 2
            scp256_subset_dataset = pd.DataFrame()
            for chunk in pd.read_csv(self.save_path + '/processed_scp256_count_celltype_gene.csv.gz', compression='gzip',header=0, sep=',', chunksize=chunksize):
                scp256_subset_dataset = pd.concat([scp256_subset_dataset, chunk], axis=0)

            count = scp256_subset_dataset.iloc[:,1:].values
            genenames = np.array(scp256_subset_dataset.columns.to_list()[1:])
            labels = scp256_subset_dataset.loc[:,'Cell_Type']
            cell_type, labels = np.unique(np.asarray(labels).astype('str'), return_inverse=True)

        elif self.dataname == 'SCP548':
            '''
            scp256_genelist=[]
            chunksize = 10 ** 2
            for chunk in pd.read_csv(self.save_path + '/log_normalized_matrix_012020.txt.gz', compression='gzip',
                                     header=0, sep='\t', chunksize=chunksize):
                scp256_genelist += chunk.loc[:,'GENE'].values.tolist()
            with open(self.save_path + '/scp256_genelist.pkl', 'wb') as f:
                pickle.dump({'gene':scp256_genelist}, f)
            
            with open(self.save_path + '/scp256_genelist.pkl', 'rb') as f:
                scp256_genelist = pickle.load(f)['gene']

            scp548_meta = pd.read_csv(self.save_path + '/scp_meta.txt', sep='\t')
            # subset the 'Control' cohort
            scp548_meta_control = scp548_meta[scp548_meta.Cohort.eq('Control')]
            scp548_subset_cellname = scp548_meta_control.loc[:, 'NAME'].values.tolist()

            chunksize = 10 ** 2
            for chunk in pd.read_csv(self.save_path + '/scp_gex_matrix.csv.gz', chunksize=chunksize):
                scp548_cellname = chunk.columns.to_list()
                break
            # not all cells in scp548_subset_cellname are included in scp548_cellname
            scp548_subset_cellname2 = [k for k in scp548_subset_cellname if k in scp548_cellname]

            scp548_subset_dataframe = pd.DataFrame()
            for chunk in pd.read_csv(self.save_path + '/scp_gex_matrix.csv.gz', chunksize=chunksize):
                chunk_subset = chunk[chunk.GENE.isin(scp256_genelist)].loc[:, ['GENE'] + scp548_subset_cellname2]
                scp548_subset_dataframe = pd.concat([scp548_subset_dataframe, chunk_subset], axis=0)

            genenames = scp548_subset_dataframe.loc[:, 'GENE'].values.tolist()
            cellid = scp548_subset_dataframe.columns.to_list()[1:]
            labels = pd.DataFrame.from_dict({'NAME':cellid}).merge(scp548_meta_control, how='left',left_on='NAME', right_on='NAME').loc[:, 'Cell_Type']
            
            Cell_Type_comparison_df = pd.DataFrame.from_dict({'scp548_Cell_Type':['B', 'DC', 'Megakaryocyte', 'Mono', 'NK', 'T'], 'scp256_Cell_Type': ['B cell', 'dendritic cell', 'Megakaryocyte', 'monocyte', 'natural killer cell', 'T cell']})
            new_labels = pd.DataFrame(labels).merge(Cell_Type_comparison_df, how='left', left_on='Cell_Type', right_on='scp548_Cell_Type').loc[:, 'scp256_Cell_Type'].values.tolist()
            
            # change row for a cell, column for a gene for scp548
            scp548_subset_dataframe = np.transpose(scp548_subset_dataframe.iloc[:, 1:].values)
            # exp(element)-1 because in scVI, x will be transformed into log(1+x)
            for i in range(scp548_subset_dataframe.shape[0]):
                scp548_subset_dataframe[i,:] = np.exp(scp548_subset_dataframe[i,:]) - 1
            
            scp548_subset_dataframe = pd.DataFrame(scp548_subset_dataframe, columns=genenames)
            scp548_subset_dataframe = pd.concat([pd.DataFrame.from_dict({'Cell_Type': new_labels}),scp548_subset_dataframe],axis=1)
            scp548_subset_dataframe.to_csv(self.save_path + "processed_scp548_count_celltype_gene.csv.gz", index=False, compression="gzip")
        '''
            chunksize = 10 ** 2
            scp548_subset_dataset = pd.DataFrame()
            for chunk in pd.read_csv(self.save_path + '/processed_scp548_count_celltype_gene.csv.gz',
                                     compression='gzip', header=0, sep=',', chunksize=chunksize):
                scp548_subset_dataset = pd.concat([scp548_subset_dataset, chunk], axis=0)

            count = scp548_subset_dataset.iloc[:, 1:].values
            genenames = np.array(scp548_subset_dataset.columns.to_list()[1:])
            labels = scp548_subset_dataset.loc[:, 'Cell_Type']
            cell_type, labels = np.unique(np.asarray(labels).astype('str'), return_inverse=True)

        return(count, labels, cell_type, genenames)

