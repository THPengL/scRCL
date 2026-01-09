
import scanpy
import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
from utils import (preprocess,
                   construct_interaction,
                   get_feature,
                   get_edge_index,
                   get_feat_input)



def load_h5_data(dataPath, dataset, hvg, n_neighbors=15, ts=None, metric='cosine'):
    adata = ad.read(dataPath)
    scanpy.pp.filter_cells(adata, min_genes=1)
    scanpy.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    adata.X = adata.X.astype(np.float32)
    scanpy.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]

    rawData = adata.raw[:, adata.raw.var['highly_variable']].X
    adj, r_adj = adata_knn(adata, method='gauss', knn=True,
                           n_neighbors=n_neighbors, metric=metric)
    adj = adj.toarray()
    adj[adj > 0] = int(1)
    edge_index = np.where(adj > 0)
    edge_index = np.concatenate((np.expand_dims(edge_index[0], axis=0), np.expand_dims(edge_index[1], axis=0)), axis=0)
    adj_gene = kneighbors_graph(rawData.T, n_neighbors=5, mode='connectivity', include_self=True)
    if issparse(adj_gene):
        adj_gene = adj_gene.toarray()
    edge_index_g = np.where(adj_gene > 0)
    edge_index_g = np.concatenate((np.expand_dims(edge_index_g[0], axis=0), np.expand_dims(edge_index_g[1], axis=0)), axis=0)

    feature = rawData
    if 'cell_type1' in adata.obs.keys():
        celltype = adata.obs['cell_type1']
        label = celltype.values.codes
        feature = feature.toarray()
    else:
        celltype = adata.obs['celltype']
        label = celltype.values
    features = get_feat_input(feature, adj, ts)
    n_classes =len(np.unique(celltype))

    dataDict = {}
    dataDict['features'] = features
    dataDict['adj'] = adj
    dataDict['label'] = label
    dataDict['n_classes'] = n_classes
    dataDict['edge_index'] = edge_index
    dataDict['edge_index_g'] = edge_index_g

    return dataDict, adata


def load_ST_data(dataPath, dataset, hvg=3000, n_neighbors=15, ts=None):
    if dataset in ["151507", "151508", "151509", "151510", "151669", "151670",
                   "151671", "151672", "151673", "151674", "151675", "151676"]:
        file_fold = f"{dataPath}/DLPFC/{dataset}/"
        adata = scanpy.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
    elif dataset == "Mouse_Embryo_E9.5":
        file_fold = f'{dataPath}/{dataset}/'
        # adata = scanpy.read_h5ad(file_fold + 'E9.5_E1S1.MOSTA.h5ad')
        adata = scanpy.read_h5ad(file_fold + "E9.5_E1S1.MOSTA.h5ad")
    elif dataset == "STARmap_mouse_visual_cortex":
        file_fold = f'{dataPath}/{dataset}/'
        adata = scanpy.read_h5ad(file_fold + 'STARmap_20180505_BY3_1k.h5ad')
    elif dataset == "Mouse_Brain_Anterior":
        file_fold = f"{dataPath}/{dataset}/"
        adata = scanpy.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
    elif dataset == "Human_Breast_Cancer":
        file_fold = f"{dataPath}/{dataset}/"
        adata = scanpy.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
    elif dataset == "Mouse_Olfactory":
        file_fold = f"{dataPath}/{dataset}/"
        adata = scanpy.read(file_fold + 'filtered_feature_bc_matrix.h5ad')
    else:
        raise Exception('Undefined dataset')
    adata.var_names_make_unique()

    if dataset == "Mouse_Embryo_E9.5":
        adata.obs['ground_truth'] = adata.obs['annotation'].values
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        unique_layer = adata.obs['ground_truth'].dropna().unique()
    else:
        meta = pd.read_csv(file_fold + 'metadata.tsv', sep='\t')
        if dataset in ["Mouse_Brain_Anterior", "Human_Breast_Cancer", "Mouse_Olfactory"]:
            meta_layer = meta['ground_truth']
        else:
            meta_layer = meta['layer_guess']
        adata.obs['ground_truth'] = meta_layer.values
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]
        unique_layer = meta_layer.dropna().unique()

    layer2num_dict = {label: i + 1 for i, label in enumerate(sorted(unique_layer))}
    adata.uns['layer2num_dict'] = layer2num_dict

    ground_truth = np.array(adata.obs['ground_truth']).astype('str')
    for i in range(len(ground_truth)):
        ground_truth[i] = layer2num_dict[ground_truth[i]]
    ground_truth = ground_truth.astype('int')
    num_class = len(np.unique(ground_truth))

    preprocess(adata, hvg=hvg)
    construct_interaction(adata, n_neighbors=n_neighbors)
    adata = get_feature(adata)
    adata.uns['edge_index'] = get_edge_index(adata.obsm['adj'])
    features = get_feat_input(adata.obsm['feat'], adata.obsm['adj'], ts)

    adj_gene = kneighbors_graph(adata.obsm['feat'].T, n_neighbors=5, mode='connectivity', include_self=True)
    if issparse(adj_gene):
        adj_gene = adj_gene.toarray()
    edge_index_g = get_edge_index(adj=adj_gene)

    dataDict = {}
    dataDict['features'] = features
    dataDict['adj'] = adata.obsm['adj']
    dataDict['edge_index'] = adata.uns['edge_index']
    dataDict['edge_index_g'] = edge_index_g
    dataDict['label'] = ground_truth
    dataDict['n_classes'] = num_class
    dataDict['layer2num_dict'] = layer2num_dict
    dataDict['position'] = adata.obsm['spatial']

    return dataDict, adata


def adata_knn(adata, method, knn, n_neighbors=2, metric='cosine'):
    if adata.shape[0] >= 10000:
        scanpy.pp.pca(adata, n_comps=50)
        n_pcs = 50
    else:
        n_pcs = 0
    if method == 'umap':
        scanpy.pp.neighbors(adata, method=method, metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        scanpy.pp.neighbors(adata, method='gauss', metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']

    return adj, r_adj



