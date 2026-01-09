import os
import ot
import torch
import random
import umap
import numpy as np
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import scanpy as sc
from matplotlib.colors import ListedColormap
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors



def add_self_loops(A, value=1.0):
    """Set the diagonal for sparse adjacency matrix."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    """Remove self-loops from the sparse adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def euclidean_distance(data1, data2, device=torch.device('cuda:0')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # N*N matrix for pairwise euclidean distance
    dis = dis.sum(dim=-1).squeeze()

    return dis


def cosine_sim(data1, data2, device=torch.device('cuda:0')):
    """
    Calculate the cosine similarity between each row in matrix A and that in matrix B.\n

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_sim (Tensor): (N x N) cosine similarity between each row in matrix data1 and\\
        that in matrix data2.
    """
    A, B = data1.to(device), data2.to(device)

    # unitization eg.[3, 4] -> [3/sqrt(9 + 16), 4/sqrt(9 + 16)] = [3/5, 4/5]
    A_norm = A / torch.norm(A, dim=1, keepdim=True)
    B_norm = B / torch.norm(B, dim=1, keepdim=True)

    cos_sim = torch.mm(A_norm, B_norm.t())

    # OR
    # A_norm = A / A.norm(dim = -1, keepdim = True)
    # B_norm = B / B.norm(dim = -1, keepdim = True)
    # cos_sim = (A_norm * B_norm).sum(dim=-1)

    return cos_sim


def cosine_distance(data1, data2, device=torch.device('cuda:0')):
    """\
    Calculate the cosine distance between each row in matrix A and that in matrix B.

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_distance (Tensor): (N x N) cosine distance between each row in matrix data1 and\\
        that in matrix data2.
    """
    cos_sim = cosine_sim(data1, data2, device)
    cos_distance = 1 - cos_sim

    return cos_distance


def filter_noise(feature, adj, times, renorm=True):
    if times == 0:
        return feature
    else:
        adj = sp.coo_matrix(adj)
        eye_ = sp.eye(adj.shape[0])
        adj_ = adj if not renorm else adj + eye_
        row_sum = np.array(adj_.sum(1))
        D_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
        adj_n = D_inv_sqrt.dot(adj_).dot(D_inv_sqrt).tocoo()
        m_ = eye_ - adj_n
        feat_ = feature
        for i in range(times):
            h_ = eye_ - m_
            feat_ = h_.dot(feat_)
        feat_ = sp.csr_matrix(feat_).toarray()
        return feat_


def get_fusion(z1, z2, fusion_mode=0):
    if fusion_mode == 0:
        fusion = torch.concat((z1, z2), dim=-1)
    elif fusion_mode == 1:
        fusion = (z1 + z2) / 2
    else:
        fusion = z1 + z2

    return fusion


def view_distribution(data, temperature=1.0):
    p_view = torch.exp(data / temperature)
    p_view = p_view / torch.sum(p_view)

    return p_view


def sample_distribution(data, temperature=1.0):
    p_sample = F.softmax(data / temperature, dim=-1)

    return p_sample


def kl_div_matrix(matrix1, matrix2, temperature=1.0, device=torch.device('cuda:0')):
    if matrix1.device != device or matrix2.device != device:
        matrix1, matrix2 = matrix1.to(device), matrix2.to(device)

    matrix1 = F.softmax(matrix1 / temperature, dim=-1)
    matrix2 = F.softmax(matrix2 / temperature, dim=-1)

    n_samples = matrix1.size(0)

    kl_div_matrix = torch.sum(matrix1 * torch.log(matrix1), dim=-1).view(n_samples, 1)
    kl_div_matrix = kl_div_matrix.expand(n_samples, n_samples) - torch.matmul(matrix1, torch.log(matrix2).t())
    
    return kl_div_matrix


def symmetric_kl_divergence_global(x_1, x_2, t=1.0):
    p_x_1_global = view_distribution(x_1, t)
    p_x_2_global = view_distribution(x_2, t)

    skl_dev = torch.sum(p_x_1_global * (torch.log(p_x_1_global) - torch.log(p_x_2_global)))
    skl_dev += torch.sum(p_x_2_global * (torch.log(p_x_2_global) - torch.log(p_x_1_global)))

    return skl_dev


def symmetric_kl_divergence_sample(x_1, x_2, t=1.0):
    p_x_1_sample = sample_distribution(x_1, t)
    p_x_2_sample = sample_distribution(x_2, t)

    skl_dev = torch.sum(p_x_1_sample * (torch.log(p_x_1_sample) - torch.log(p_x_2_sample))) / p_x_1_sample.size(0)
    skl_dev = (skl_dev + torch.sum(p_x_2_sample * (torch.log(p_x_2_sample) - torch.log(p_x_1_sample))) / p_x_2_sample.size(0))/2

    return skl_dev


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def preprocess(adata, hvg=3000):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def get_feature(adata):
    adata = adata[:, adata.var['highly_variable']]

    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        feat = adata.X.toarray()[:, ]
    else:
        feat = adata.X[:, ]
    adata.obsm['feat'] = feat

    return adata


def get_edge_index(adj):
    if isinstance(adj,np.ndarray):
        adj = csr_matrix(adj)
    rows, cols = adj.nonzero()
    edge_index = np.vstack((rows,cols)).astype(np.int64)
    return edge_index


def get_feat_input(feat, adj, ts):
    adj_ = eliminate_self_loops(csr_matrix(adj))
    features = []
    if ts is None:
        ts = [0, 0]
    for t in ts:
        features.append(filter_noise(feat, adj_, t))
    return features


def show_tsne(feature, true_label, random_state=5, save_path=None):
    mpl.rcParams.update({'figure.dpi': 300})
    color_list14 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a','#0687aa']
    all_colors = len(color_list14)
    n_color = len(np.unique(true_label))
    assert n_color <= all_colors, f"The number of classes must be less than or equal to {all_colors}."
    color_list = color_list14[0:n_color]
    cmap = ListedColormap(color_list)

    X_tsne = TSNE(n_components=2, random_state=random_state).fit_transform(feature)
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0],
                X_tsne[:, 1],
                c=true_label,
                label="feature",
                s=45,
                marker='o',
                cmap=cmap)
    # plt.legend(fontsize = 20)
    # plt.colorbar()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(bottom=False, left=False)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def show_umap(feature, true_label, show_legend=False, save_path=None):
    mpl.rcParams.update({'figure.dpi': 300})
    color_list14 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a', '#0687aa']
    all_colors = len(color_list14)
    n_color = len(np.unique(true_label))
    n_sample = len(true_label)
    assert n_color <= all_colors, f"The number of classes must be less than or equal to {all_colors}."

    color_dict = {}
    for idx, label in enumerate(np.unique(true_label)):
        cur_color = color_list14[idx]
        color_dict[label] = cur_color

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(feature)

    plt.figure(figsize=(10, 10))
    for label in set(true_label):
        x_list = [X_umap[i, 0] for i in range(n_sample) if true_label[i] == label]
        y_list = [X_umap[i, 1] for i in range(n_sample) if true_label[i] == label]
        plt.scatter(x_list,
                    y_list,
                    color=color_dict[label],
                    label=label,
                    s=45,
                    marker='o'
                    )
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    spine_width = 3
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['right'].set_linewidth(spine_width)
    ax.spines['top'].set_linewidth(spine_width)
    ax.tick_params(bottom=False, left=False)
    if show_legend:
        plt.legend(fontsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('UMAP 1', fontsize=40)
    plt.ylabel('UMAP 2', fontsize=40)

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
