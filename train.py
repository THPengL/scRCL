import os
import numpy as np
import torch
import scanpy as sc
import scipy.io as sio
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from model import Model
from clustering import clustering, refine_label
from evaluation import evaluate
from dataset import load_h5_data, load_ST_data
from utils import (set_random_seed, get_fusion, )


def train(config, seeds):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config['task'] == "SC":
        print(f"Loading scRNA-seq dataset {config['dataset']}...")
        data_path = f"./datasets/scdata/{config['dataset']}.h5ad"
        data, adata = load_h5_data(data_path, config['dataset'], config['hvg'], config['k'], config['tms'])

    elif config['task'] == "ST":
        print(f"Loading spatial transcriptomics dataset {config['dataset']}...")
        data_path = f"./datasets/stdata"
        data, adata = load_ST_data(data_path, config['dataset'], config['hvg'], config['k'], config['tms'])
        position = data['position']
    else:
        raise NotImplementedError("The task should be 'SC' or 'ST'.")

    features = data['features']
    config['n_samples'] = features[0].shape[0]
    config['n_classes'] = data['n_classes']
    label_true = data['label']
    config['in_dim'] = features[0].shape[-1]
    if config['out_dim'] == 0:
        config['out_dim'] = config['n_classes']
    config['gene_dim'] = features[0].shape[0]

    adj = data['adj']
    edge_index = data['edge_index']
    adj = torch.tensor(adj, dtype=torch.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    adj = adj.fill_diagonal_(0)  # set the diagonal of adj as 0, remove self loops
    edge_index_g = data['edge_index_g']
    edge_index_g = torch.tensor(edge_index_g, dtype=torch.int64)
    in_features = [torch.FloatTensor(feat) for feat in features]

    results = {
        'acc': [], 'nmi': [], 'ari': [], 'f1': [],
    }
    embd_dict = {}
    pred_label_dict = {}

    for seed in seeds:
        set_random_seed(seed)
        print(f'===================== SEED {seed} =====================')

        model = Model(in_dim=config['in_dim'],
                      hid_dim=config['hid_dim'],
                      out_dim=config['out_dim'],
                      gene_dim=config['gene_dim'],
                      dropout=config['dropout'],
                      device=device)

        optimizer = optim.Adam(params=model.parameters(),
                               lr=float(config['lr']),
                               weight_decay=float(config['weight_decay']))

        model = model.to(device)
        in_features = [feat.to(device) for feat in in_features]
        adj = adj.to(device)
        edge_index = edge_index.to(device)
        edge_index_g = edge_index_g.to(device)

        record_list = {'loss': [], 'acc': [], 'nmi': [], 'ari': [], 'f1': []}
        acc_best, nmi_best, ari_best, f1_best, best_epoch = 0.0, 0.0, 0.0, 0.0, 0

        for epoch in tqdm(range(config['epochs'])):
            model.train()
            emb_1, emb_2, z_1, z_2 = model(in_features,
                                           edge_index,
                                           edge_index_g)

            loss_hea = model.embedding_distribution_alignment(x_1=emb_1, x_2=emb_2, t=config['t'])
            loss_ndc = model.neighborhood_contrastive_alignment(x_1=emb_1, x_2=emb_2, adj=adj, t=config['t'])
            loss_cvc = model.cross_view_consistency(Z1=z_1, Z2=z_2, adj=adj)

            loss = loss_hea + config['lambda1'] * loss_ndc + config['lambda2'] * loss_cvc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            emb_1, emb_2, z_1, z_2 = model(in_features,
                                           edge_index,
                                           edge_index_g)
            emb_fusion = get_fusion(z1=z_1.detach(),
                                    z2=z_2.detach())
            label_pred, _, _ = clustering(feature=emb_fusion,
                                          cluster_num=config['n_classes'],
                                          device=device)
            acc, nmi, ari, f1 = evaluate(label_true, label_pred)

            record_list['acc'].append(acc)
            record_list['nmi'].append(nmi)
            record_list['ari'].append(ari)
            record_list['f1'].append(f1)
            record_list['loss'].append(loss.item())

            # ACC, NMI or ARI
            if config['task'] == "ST":
                if ari >= ari_best:
                    acc_best = acc
                    nmi_best = nmi
                    ari_best = ari
                    f1_best = f1
                    embeddings = emb_fusion.cpu().numpy()
                    best_pred = label_pred
            else:
                if nmi >= nmi_best:
                    acc_best = acc
                    nmi_best = nmi
                    ari_best = ari
                    f1_best = f1
                    embeddings = emb_fusion.cpu().numpy()
                    best_pred = label_pred

        if config['task'] == "ST" and config['radius'] > 0:
            label_pred_rf = refine_label(best_pred, position, radius=config['radius'])
            acc_best, nmi_best, ari_best, f1_best = evaluate(label_true, label_pred_rf)
        print(f" ACC {acc_best:.4f}, NMI {nmi_best:.4f}, ARI {ari_best:.4f}, F1 {f1_best:.4f}")

        if config['task'] == "ST":
            if config['radius'] > 0:
                adata.obs['label_pred'] = label_pred_rf.astype('int').astype('str')
                pred_labels = label_pred_rf.astype('int')
            else:
                adata.obs['label_pred'] = best_pred.astype('int').astype('str')
                pred_labels = best_pred.astype('int')

            if config['save_res']:
                embd_dict["emb"] = embeddings
                pred_label_dict["pred_labels"] = pred_labels

            if config['showimg']:
                if not os.path.exists(f"{config['fig_path']}/{config['dataset']}"):
                    os.makedirs(f"{config['fig_path']}/{config['dataset']}")

                # spatial domains
                spatial_gt_name = f"{config['fig_path']}/{config['dataset']}/{config['dataset']}_spatial_manual_annotation.png"
                # if not os.path.exists(spatial_gt_name):
                plt.rcParams["figure.figsize"] = (5, 5)
                if config['dataset'] == "Mouse_Embryo_E9.5":
                    adata.obs['x_pos'] = adata.obsm['spatial'][:, 0]
                    adata.obs['y_pos'] = adata.obsm['spatial'][:, 1]
                    ax1 = sc.pl.scatter(adata,
                                        alpha=1,
                                        x="x_pos", y="y_pos",
                                        color="ground_truth",
                                        title="Manual Annotation",
                                        palette=adata.uns['annotation_colors'],
                                        show=False)
                    ax1.set_aspect('equal', 'box')
                    ax1.axes.invert_yaxis()
                    ax1.axis('off')
                    plt.tight_layout()
                    plt.savefig(spatial_gt_name,
                                bbox_inches='tight',
                                dpi=300)
                    plt.show()
                else:
                    sc.pl.spatial(adata,
                                  img_key="hires",
                                  color=["ground_truth"],
                                  title="Manual Annotation",
                                  show=False)
                    plt.gca().set_xlabel("")
                    plt.gca().set_ylabel("")
                    plt.savefig(spatial_gt_name,
                                dpi=300,
                                bbox_inches='tight')
                    plt.show()

                plt.rcParams["figure.figsize"] = (5, 5)
                spatial_name = f"{config['fig_path']}/{config['dataset']}/{config['dataset']}_spatial.png"
                if config['dataset'] == "Mouse_Embryo_E9.5":
                    adata.obs['x_pos'] = adata.obsm['spatial'][:, 0]
                    adata.obs['y_pos'] = adata.obsm['spatial'][:, 1]
                    ax = sc.pl.scatter(adata,
                                       alpha=1,
                                       x="x_pos", y="y_pos",
                                       color="label_pred",
                                       title=f"ARI: {ari_best:.2f}",
                                       palette=adata.uns['annotation_colors'],
                                       show=False)
                    ax.set_aspect('equal', 'box')
                    ax.axes.invert_yaxis()
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(spatial_name, bbox_inches='tight', dpi=300)
                else:
                    sc.pl.spatial(adata,
                                  color=["label_pred"],
                                  title=[f"ARI: {ari_best:.2f}"],
                                  # title=[''],
                                  show=False)
                    plt.gca().set_xlabel("")
                    plt.gca().set_ylabel("")
                    plt.savefig(spatial_name, dpi=300, bbox_inches='tight')
                plt.show()

                # UMAP visualizations and PAGA trajectory inference
                if config['showpaga']:
                    if not os.path.exists(f"{config['fig_path']}/{config['dataset']}/paga"):
                        os.makedirs(f"{config['fig_path']}/{config['dataset']}/paga")

                    adata.obsm['embedding'] = embeddings
                    sc.pp.neighbors(adata, use_rep='embedding')

                    sc.tl.umap(adata)
                    plt.rcParams["figure.figsize"] = (5, 5)
                    sc.pl.umap(adata,
                               color="ground_truth",
                               title='',
                               show=False)
                    umap_name = f"{config['fig_path']}/{config['dataset']}/paga/{config['dataset']}_umap.png"
                    plt.savefig(umap_name, dpi=300, bbox_inches='tight')
                    plt.show()

                    sc.tl.paga(adata, groups='ground_truth')
                    plt.rcParams["figure.figsize"] = (5, 5)
                    sc.pl.paga(adata,
                               threshold=0.0,
                               color='ground_truth',
                               frameon=False,
                               title='',
                               show=False,
                               # labels=None,
                               # text_kwds={'alpha':0})           # no node label text
                               )
                    page_name = f"{config['fig_path']}/{config['dataset']}/paga/{config['dataset']}_paga.png"
                    plt.savefig(page_name, dpi=300, bbox_inches='tight')
                    plt.show()

                    plt.rcParams["figure.figsize"] = (5, 5)
                    sc.pl.paga_compare(adata,
                                       legend_fontsize=10,
                                       frameon=False,
                                       size=50,
                                       title='',
                                       legend_fontoutline=2,
                                       show=False,
                                       # labels=None,                 # no labels
                                       # node_labels=False,           # no node labels
                                       # text_kwds={'alpha': 0},      # no node label text
                                       )
                    page_cmp_name = f"{config['fig_path']}/{config['dataset']}/paga/{config['dataset']}_paga_cmp.png"
                    plt.savefig(page_cmp_name, dpi=300, bbox_inches='tight')
                    plt.show()

        elif config['task'] == "SC" and config['save_res']:
            pred_labels = best_pred.astype('int')
            embd_dict[f"{seed}"] = embeddings
            pred_label_dict[f"{seed}"] = pred_labels

        results['acc'].append(acc_best)
        results['nmi'].append(nmi_best)
        results['ari'].append(ari_best)
        results['f1'].append(f1_best)

    if config['save_res']:
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])
        sio.savemat(f"{config['save_path']}/emb_dict_{config['dataset']}.mat", embd_dict)
        sio.savemat(f"{config['save_path']}/pred_label_dict_{config['dataset']}.mat", pred_label_dict)

    return results
