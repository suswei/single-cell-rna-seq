from os import walk
import pandas as pd
from scvi.dataset.dataset import GeneExpressionDataset
import numpy as np
import scipy.sparse as sp

class Macaque_Retina(GeneExpressionDataset):
    def __init__(self, dataname, save_path='./data/pareto_front_paretoMTL/macaque_retina/', region='fovea'):
        self.save_path = save_path
        self.dataname = dataname
        self.region = region
        count, labels, cell_type, gene_names = self.preprocess()

        super(Macaque_Retina, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)
    def preprocess(self):
        # As in the paper 'Deep learning enables accurate clustering with batch effect removal in single-cell RNA-seq analysis',
        # we only focus on the bipolar cells.
        # Download the count matrix for both fovea and periphery from GSE118480 on the NCBI GEO website
        # Download the metadata file ('Macaque_NN_RGC_AC_BC_HC_PR_metadata_3.txt') from the study titled
        # 'Study: Molecular specification of retinal cell types underlying central and peripheral vision in primates' on single cell portal website
        '''
        retina_meta = pd.read_csv(self.save_path + 'Macaque_NN_RGC_AC_BC_HC_PR_metadata_3.txt', sep=',')
        # bipolar cell types
        cell_types_list = ['BB/GB*', 'DB1', 'DB2', 'DB3a', 'DB3b', 'DB4', 'DB5*', 'DB6', 'FMB', 'IMB', 'OFFx', 'RB']
        retina_meta_fovea = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Fovea')]
        retina_meta_per = retina_meta[retina_meta.Cluster.isin(cell_types_list) & retina_meta.Subcluster.eq('Periphery')]

        dirs = []
        for (dirpath, dirnames, filenames) in walk(self.save_path):
            dirs.extend(dirnames)
            break

        tissue = 'fovea'
        count_total = pd.DataFrame()
        for dirname in dirs:
            files = []
            for (dirpath, dirnames, filenames) in walk(self.save_path + dirname + '/'):
                files.extend(filenames)
                break
            count_onefile = pd.DataFrame()
            if tissue == 'fovea':
                for filename in files:
                    if 'fovea' in filename:
                        chunksize = 10 ** 2
                        for chunk in pd.read_csv(self.save_path + dirname + '/' + filename, chunksize=chunksize):
                            chunk_columns = chunk.columns.to_list()
                            break
                        if any('Fovea4S' in s for s in chunk_columns):
                            chunk_columns = ['Unnamed: 0'] + [s.replace('Fovea4S', 'M4Fovea') for s in chunk_columns[1:]]
                        subset_index = [s in ['Unnamed: 0'] + retina_meta_fovea.loc[:, 'NAME'].values.tolist() for s in chunk_columns]
                        for chunk in pd.read_csv(self.save_path + dirname + '/' + filename, chunksize=chunksize):
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
                        for chunk in pd.read_csv(self.save_path + dirname + '/' + filename, chunksize=chunksize):
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
                        for chunk in pd.read_csv(self.save_path + dirname + '/' + filename, sep=',', chunksize=chunksize):
                            chunk_subset = chunk.loc[:, subset_index]
                            if chunk_subset.shape[1] == 1:
                                break
                            else:
                                count_onefile = pd.concat([count_onefile, chunk_subset], axis=0)
                        if count_onefile.shape[0] > 0:
                            chunk_columns_subset = [chunk_columns[k] for k in range(len(chunk_columns)) if subset_index[k] == True]
                            count_onefile.columns = chunk_columns_subset

            if count_total.shape[0] == 0 and count_onefile.shape[0] > 0:
                count_onefile = count_onefile.rename({'Unnamed: 0': 'GENE'}, axis='columns')
                count_total = count_onefile
            elif count_total.shape[0] > 0 and count_onefile.shape[0] > 0:
                count_onefile = count_onefile.rename({'Unnamed: 0': 'GENE'}, axis='columns')
                count_total = count_total.merge(count_onefile, how='inner', left_on='GENE', right_on='GENE')
            # for both fovea and periphery, the total UMI for all cells are larger than 100
            count_total.to_csv(self.save_path + "processed_macaque_{}_BC_count.csv.gz".format(tissue), index=False,
                               compression="gzip")
            '''

        chunksize = 10 ** 2
        genenames, cell_names = [], []
        if self.region == 'fovea':
            index = 0
            for chunk in pd.read_csv(self.save_path + 'processed_macaque_fovea_BC_count.csv.gz', compression='gzip', header=0, sep=',', chunksize=chunksize):
                if index == 0:
                    count = sp.csr_matrix(np.transpose(chunk.iloc[:,1:]))
                    genenames += chunk.iloc[:, 0].values.tolist()
                    cell_names = pd.DataFrame.from_dict({'NAME': chunk.columns.to_list()[1:]})
                    index = 1

                genenames += chunk.iloc[:, 0].values.tolist()
                count = sp.hstack((count, sp.csr_matrix(np.transpose(chunk.iloc[:,1:]))))

        elif self.region == 'periphery':
            index = 0
            for chunk in pd.read_csv(self.save_path + 'processed_macaque_periphery_BC_count.csv.gz', compression='gzip', header=0, sep=',', chunksize=chunksize):
                if index == 0:
                    count = sp.csr_matrix(np.transpose(chunk.iloc[:, 1:]))
                    genenames += chunk.iloc[:, 0].values.tolist()
                    cell_names = pd.DataFrame.from_dict({'NAME': chunk.columns.to_list()[1:]})
                    index = 1

                genenames += chunk.iloc[:, 0].values.tolist()
                count = sp.hstack((count, sp.csr_matrix(np.transpose(chunk.iloc[:, 1:]))))

        genenames = np.array(genenames)

        meta = pd.read_csv(self.save_path + 'Macaque_NN_RGC_AC_BC_HC_PR_metadata_3.txt')
        labels = cell_names.merge(meta, how='left', left_on='NAME', right_on='NAME').loc[:, 'Cluster']
        cell_type, labels = np.unique(np.asarray(labels).astype('str'), return_inverse=True)
        return(count.tocsr(), labels, cell_type, genenames)

