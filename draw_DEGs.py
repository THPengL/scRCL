
import os
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


dataset = "Quake_Smart-seq2_Lung"
print(f"Dataset: {dataset}")

FONTSIZE = 15
TITLESIZE = 35
LABELSIZE = 35
TICKLABELSIZE = 45
LEGENDFONTSIZE = 3

data_file = f"./datasets/scdata/{dataset}.h5ad"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"'{data_file}' not found.")

adata = sc.read_h5ad(data_file)
adata.X = adata.X.astype(np.float32)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
sc.pp.log1p(adata)
if 'cell_type1' in adata.obs.keys():
    celltype = adata.obs['cell_type1']
    true_label = celltype.values.codes
    cell_index = adata.obs['cell_type1'].index
else:
    celltype = adata.obs['celltype']
    true_label = celltype.values
    cell_index = adata.obs['celltype'].index
print(f"annData: {adata}")
print(f"Gene expressed matrix: {adata.X.shape}")

pred_file_name = f"./results/SC/pred_label_dict_{dataset}.mat"
if not os.path.exists(pred_file_name):
    raise FileNotFoundError(f"'{pred_file_name}' not found.")
pred_dict = sio.loadmat(pred_file_name)

# seeds = [5, 15, 25, 35, 45]
seeds = [45]
for seed in seeds:
    kmeans_pred = np.squeeze(pred_dict[f'{seed}'])
    adata.obs['kmeans'] = pd.Series(kmeans_pred.astype(str), index = cell_index, dtype="category")

    n_genes = 3
    fig_root = f"./figures/SC/{dataset}/DEGs"
    if not os.path.exists(fig_root):
        os.makedirs(fig_root)

    sc.tl.rank_genes_groups(adata,
                            groupby='kmeans',
                            n_genes=20,
                            method='wilcoxon',
                            key_added="wilcoxon")
    sc.tl.filter_rank_genes_groups(adata,
                                   groupby='kmeans',
                                   key='wilcoxon',
                                   min_in_group_fraction=0.5,
                                   max_out_group_fraction=0.25,
                                   min_fold_change=1)

    plt.figure(figsize=(20, 20), dpi=300)
    sc.pl.rank_genes_groups(adata,
                            n_genes=20,
                            sharey=False,
                            key="rank_genes_groups_filtered")
    # plt.rcParams.update({
    #     'font.size': FONTSIZE,
    #     'axes.titlesize': TITLESIZE,
    #     'axes.labelsize': LABELSIZE,
    #     'xtick.labelsize': TICKLABELSIZE,
    #     'ytick.labelsize': TICKLABELSIZE,
    #     'legend.fontsize': LEGENDFONTSIZE
    # })
    # plt.tight_layout()
    # myranks.savefig(f"{fig_root}/{dataset}_rank_genes_groups_{seed}.png", dpi=300, bbox_inches='tight')
    # plt.close()

    plt.figure(figsize=(20, 20), dpi=300)
    mydotplot = sc.pl.rank_genes_groups_dotplot(adata,
                                                n_genes=n_genes,
                                                key="rank_genes_groups_filtered",
                                                groupby="kmeans",
                                                dendrogram=False,
                                                return_fig=True)
    # plt.rcParams.update({
    #     'font.size': FONTSIZE,                  # General font size   legend 15, normal 20
    #     'axes.titlesize': TITLESIZE,            # Title font size
    #     'axes.labelsize': LABELSIZE,            # X and Y axis label font size
    #     'xtick.labelsize': TICKLABELSIZE,       # X-axis tick label font size
    #     'ytick.labelsize': TICKLABELSIZE,       # Y-axis tick label font size
    #     'legend.fontsize': LEGENDFONTSIZE       # Legend font size
    # })
    plt.tight_layout()
    mydotplot.savefig(f"{fig_root}/{dataset}_dotplot_{seed}.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 20), dpi=300)
    sc.pl.rank_genes_groups_heatmap(adata,
                                    n_genes=n_genes,
                                    key="rank_genes_groups_filtered",
                                    groupby="kmeans",
                                    show_gene_labels=True,
                                    show=False,
                                    dendrogram=False,
                                    return_fig=True)
    # plt.rcParams.update({
    #     'font.size': FONTSIZE,
    #     'axes.titlesize': TITLESIZE,
    #     'axes.labelsize': LABELSIZE,
    #     'xtick.labelsize': TICKLABELSIZE,
    #     'ytick.labelsize': TICKLABELSIZE,
    #     'legend.fontsize': LEGENDFONTSIZE
    # })
    plt.tight_layout()
    plt.savefig(f"{fig_root}/{dataset}_heatmap_{seed}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # mydotplot = sc.pl.rank_genes_groups_dotplot(adata,
    #                                             n_genes=n_genes,
    #                                             key="wilcoxon",
    #                                             groupby="kmeans",
    #                                             dendrogram=True,
    #                                             return_fig=True)
    # mydotplot.savefig(f"{fig_root}/{dataset}_dotplot_{seed}.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(20, 20), dpi=300)
    myviolin = sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=n_genes,
                                                      key="rank_genes_groups_filtered",
                                                      groupby="kmeans",
                                                      dendrogram=False,
                                                      return_fig=True)
    # plt.rcParams.update({
    #     'font.size': FONTSIZE,
    #     'axes.titlesize': TITLESIZE,
    #     'axes.labelsize': LABELSIZE,
    #     'xtick.labelsize': TICKLABELSIZE,
    #     'ytick.labelsize': TICKLABELSIZE,
    #     'legend.fontsize': LEGENDFONTSIZE
    # })
    plt.tight_layout()
    myviolin.savefig(f"{fig_root}/{dataset}_violin_{seed}.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 20), dpi=300)
    mymatrixplot = sc.pl.rank_genes_groups_matrixplot(adata, n_genes=n_genes,
                                                      key="rank_genes_groups_filtered",
                                                      groupby="kmeans",
                                                      dendrogram=False,
                                                      return_fig=True)
    # plt.rcParams.update({
    #     'font.size': FONTSIZE,
    #     'axes.titlesize': TITLESIZE,
    #     'axes.labelsize': LABELSIZE,
    #     'xtick.labelsize': TICKLABELSIZE,
    #     'ytick.labelsize': TICKLABELSIZE,
    #     'legend.fontsize': LEGENDFONTSIZE
    # })
    plt.tight_layout()
    mymatrixplot.savefig(f"{fig_root}/{dataset}_matrixplot_{seed}.png", dpi=300, bbox_inches='tight')
    plt.close()

