import pandas as pd
from code.scvi import GeneExpressionDataset
import numpy as np
from scipy.sparse import csr_matrix

class MCA(GeneExpressionDataset):
    def __init__(self, save_path='./data/pareto_front_scVI_MINE/MCA/', tissue='Lung'):
        self.save_path = save_path
        self.tissue = tissue

        self.download()
        count, labels, cell_type, gene_names = self.preprocess()
        count = csr_matrix(count)
        super(MCA, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                count, labels=labels),
            gene_names=np.char.upper(gene_names), cell_types=cell_type)

    def preprocess(self):
        MCA_meta = pd.read_csv(self.save_path + 'MCA_CellAssignments.csv')
        MCA_meta_lung = MCA_meta[MCA_meta.Tissue.eq('Lung')]

        for id in range(3):
            id += 1
            MCA_Lung_one = pd.DataFrame()
            chunksize = 10 ** 2
            for chunk in pd.read_csv(self.save_path + 'Lung{}_dge.txt.gz'.format(id), compression='gzip', sep=' ', header=0, chunksize=chunksize):
                chunk['GENE'] = chunk.index
                first_col = chunk.pop('GENE')
                chunk.insert(0, 'GENE', first_col)
                MCA_Lung_one = pd.concat([MCA_Lung_one, chunk], axis=0)
            if id == 1:
                MCA_Lung = MCA_Lung_one
            else:
                MCA_Lung = MCA_Lung.merge(MCA_Lung_one, how='inner', left_on='GENE', right_on='GENE')

        MCA_Lung_subset = MCA_Lung.loc[:, ['GENE'] + MCA_meta_lung.loc[:, 'Cell.name'].values.tolist()]
        MCA_meta_lung_cellannotation = MCA_meta_lung.loc[:, ['Cell.name', 'Annotation']]
        Consensus_celltype_MCA = pd.read_excel(self.save_path + 'Consensus_celltype_MCA.xlsx', index_col=None, header=0)
        MCA_meta_lung_cellannotation2 = MCA_meta_lung_cellannotation.merge(Consensus_celltype_MCA, how='left', left_on='Annotation', right_on='MCA_Type')
        count = np.transpose(MCA_Lung_subset.iloc[:, 1:].values)
        genenames = np.array(MCA_Lung_subset.loc[:, 'GENE'].values.tolist())
        labels = MCA_meta_lung_cellannotation2.loc[:, 'Consensus_Type']
        cell_type, labels = np.unique(np.asarray(labels).astype('str'), return_inverse=True)

        return (count, labels, cell_type, genenames)

