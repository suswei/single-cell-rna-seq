from numpy import loadtxt
from scipy.io import mmread
import scipy.sparse as sparse

from scvi.inference.posterior import *
from scvi.dataset import GeneExpressionDataset
from copy import deepcopy

#get_matrix_from_dir and assign_label are copied directly from scvi/harmonization/utils_chenling.py from the code for
#the paper: 'Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models'

def get_matrix_from_dir(dirname,storage='../'):
    geneid = loadtxt(storage + dirname +'/genes.tsv',dtype='str',delimiter="\t")
    cellid = loadtxt(storage + dirname + '/barcodes.tsv',dtype='str',delimiter="\t")
    count = mmread(storage + dirname +'/matrix.mtx')
    return count, geneid, cellid

def assign_label(cellid, geneid, labels_map, count, cell_type, seurat):
    labels = seurat[1:, 4]
    labels = np.int64(np.asarray(labels))
    labels_new = deepcopy(labels)
    for i, j in enumerate(labels_map):
        labels_new[labels == i] = j
    temp = dict(zip(cellid, count))
    new_count = []
    for x in seurat[1:, 5]:
        new_count.append(temp[x])
    new_count = sparse.vstack(new_count)
    dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(new_count, labels=labels_new),
                                    gene_names=geneid, cell_types=cell_type)
    return dataset
