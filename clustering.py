
import ot
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import cosine_distance, euclidean_distance


def pairwise_distance(data1, data2, device=torch.device('cuda:0')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()

    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda:0')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()

    return cosine_dis


def initialize(X, n_clusters):
    """
    Initialize cluster centers.

    Parameters
    - X: (torch.tensor) matrix
    - n_clusters: (int) number of clusters
    
    Return
    - initial state: (np.array) 
    """
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    initial_state = X[indices]

    return initial_state


def k_means(X, n_clusters, distance='euclidean', tol=1e-4, device=torch.device('cuda:0')):
    """
    Perform k-means algorithm on X.

    Parameters
    - X: torch.tensor. matrix
    - n_clusters: int. number of clusters
    - distance: str. pairwise distance 'euclidean'(default) or 'cosine'
    - tol: float. Threshold 
    - device: torch.device. Running device
    
    Return
    - choice_cluster: torch.tensor. Predicted cluster ids.
    - initial_state: torch.tensor. Predicted cluster centers.
    - dis: minimum pair wise distance.
    """
    if distance == 'euclidean':
        # pairwise_distance_function = pairwise_distance
        pairwise_distance_function = euclidean_distance
    elif distance == 'cosine':
        # pairwise_distance_function = pairwise_cosine
        pairwise_distance_function = cosine_distance
    else:
        raise NotImplementedError(f"Not implemented '{distance}' distance!")

    # convert to float
    X = X.float()
    # transfer to device
    X = X.to(device)

    # initialize
    dis_min = float('inf')
    # initial_state_best = initialize(X, n_clusters)
    initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, n_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()

        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best

    # initial_state = torch.tensor(initial_state, device=device)

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state, dis
    # return choice_cluster.cpu(), initial_state


def clustering(feature, cluster_num, method="kmeans", seed = 10, device = torch.device('cpu')):
    """
    Clustering using feature matrix.

    Parameters
    - feature: feature matrix.
    - cluster_num: number of clusters.
    - seed: random seed.
    - device: torch.device. device where the clustering algorithm will be running on.

    Return
    - label_pred: predicted label.
    """
    label_pred, centers, dis = None, None, None
    if method == "kmeans":
        if device is not torch.device('cpu'):
            label_pred, centers, dis = k_means(X=feature,
                                        n_clusters=cluster_num,
                                        distance="cosine",           # cosine euclidean
                                        device=device)
            label_pred = label_pred.numpy()
        else:
            feature.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=cluster_num,
                            random_state=seed,
                            init='k-means++')
            label_pred = kmeans.fit_predict(feature)

    elif method == "mclust":
        if isinstance(feature, torch.Tensor):
            if feature.device is not torch.device('cpu'):
                feature = feature.cpu()
            feature = feature.numpy()
        label_pred = mclust_R(feature, num_cluster=cluster_num, usepca=False)

    return label_pred, centers, dis


def mclust_R(feature, num_cluster, usepca=False, modelNames='EEE', random_seed=10):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # feature = feature.cpu().detach().numpy()
    if usepca:
        pca = PCA(n_components=20, random_state=random_seed)
        embedding = pca.fit_transform(feature.copy())
    else:
        embedding = feature.copy()
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames, verbose = False)
    mclust_res = np.array(res[-2])

    return mclust_res


def refine_label(label, position, radius=50):
    n_neigh = radius
    new_type = []
    old_type = label

    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = np.array(new_type).astype('int')

    return new_type

