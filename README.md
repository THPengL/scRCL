# Official PyTorch Implementation of scRCL

***Refinement Contrastive Learning of Cell-Gene Associations for Unsupervised Cell Type Identification (AAAI'26 Post)***



### Overview

Unsupervised cell type identification is crucial for uncovering and characterizing heterogeneous populations in single cell omics studies. Although a range of clustering methods have been developed, most focus exclusively on intrinsic cellular structure and ignore the pivotal role of cell-gene associations, which limits their ability to distinguish closely related cell types. To this end, we propose a Refinement Contrastive Learning framework (scRCL) that explicitly incorporates cell-gene interactions to derive more informative representations. Specifically, we introduce two contrastive distribution alignment components that reveal reliable intrinsic cellular structures by effectively exploiting cell-cell structural relationships. Additionally, we develop a refinement module that integrates gene-correlation structure learning to enhance cell embeddings by capturing underlying cell-gene associations. This module strengthens connections between cells and their associated genes, refining the representation learning to exploiting biologically meaningful relationships. Extensive experiments on several single-cell RNA sequencing and spatial transcriptomics benchmark datasets demonstrate that our method consistently outperforms state-of-the-art baselines in cell-type identification accuracy. Moreover, downstream biological analyses confirm that the recovered cell populations exhibit coherent gene-expression signatures, further validating the biological relevance of our approach.



### Requirements

- CUDA Version: 11.6

- python>=3.9.16

- numpy==1.26.1

- torch>=1.12.1+cu116

- torch_geometric==2.4.0

- scanpy==1.9.8

- tqdm==4.66.1

- matplotlib==3.9.4

  

### Datasets

##### SC

For scRNA-seq datasets, we provide the compressed file `./datasets/scdata/scdata.zip` containing `li_tumor` (Tumor), `Human_ESC`, and `Zeisel` datasets. The datasets `Quake_Smart-seq2_Diaphragm` (Diaphragm), `Quake_Smart-seq2_Lung` (Lung), `Quake_Smart-seq2_Trachea` (Trachea), `Quake_10x_Bladder` (Bladder), `Quake_10x_Limb_Muscle` (Limb_Muscle), `Quake_10x_Spleen` (Spleen), and `Baron_human` can be downloaded from [here](https://cblast.gao-lab.org/download). 

Before running the `runSC.py` script, please place the downloaded dataset files in the `./datasets/scdata/` directory, e.g., `./datasets/scdata/Quake_Smart-seq2_Diaphragm.h5ad`. 

##### ST

With respect to spatial transcriptomics datasets, please refer to the [GraphST](https://www.nature.com/articles/s41467-023-36796-3) paper to download the `DLPFC`, `Human_Breast_Cancer`, and `Mouse_Brain_Anterior` datasets. The `Mouse_Embryo_E9.5` dataset can be downloaded form [here](https://db.cngb.org/stomics/mosta/). 

Before running the `runST.py` script, place the dataset files in the `./datasets/stdata/` directory. For example:

- `./datasets/stdata/DLPFC/151507/...` 
- `./datasets/stdata/Human_Breast_Cancer/...` 



### Examples

##### SC

For the scRNA-seq cell type identification task, you can run the script as follows:

```
python runSC.py --dataset Quake_Smart-seq2_Diaphragm
```

##### ST

Regarding spatial transcriptomics domain identification task, you can run the following script:

```
python runST.py --dataset 151673
```

##### DEGs

For differentially expressed genes (DEGs) analysis, please refer to this [tutorial](https://nbisweden.github.io/workshop-scRNAseq/labs/scanpy/scanpy_05_dge.html). After saving the model-predicted labels in the `./results/SC` directory, you can run the `draw_DEGs.py` file. 

```
python draw_DEGs.py
```



### Acknowledgments

Our code is primarily inspired by the following works:

- [GraphST](https://github.com/JinmiaoChenLab/GraphST) 
- [Spatial-MGCN](https://github.com/cs-wangbo/Spatial-MGCN) 
- [Differential gene expression](https://nbisweden.github.io/workshop-scRNAseq/labs/scanpy/scanpy_05_dge.html) 
- [单细胞转录组数据分析|| scanpy教程：PAGA轨迹推断](https://www.jianshu.com/p/e21e7ad6cb9e) 

