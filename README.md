# B2G: Batch-to-Group

**B2G** (Batch-to-Group) is an intelligent batch grouping tool for single-cell RNA-seq data that combines metacell/Leiden clustering with PERMANOVA-based prior selection.

![img.png](img.png)
---


## Why B2G and What We Do

### Motivation

Recent studies increasingly focus on dissecting cellular subpopulations from single-cell atlases of longitudinal clinical cohorts. However, residual batch effects within these subpopulations are difficult to eliminate due to their highly similar cellular states. Existing methods typically process samples individually during subpopulation-level batch correction, failing to recognize that patients with similar clinical metadata often form batch-effect-free groups. This leads to over-correction and loss of biologically meaningful signals.

**Key Challenges:**
- **Residual Batch Effects**: Global batch correction lacks resolution, leaving batch effects in specific cell types
- **Over-correction Risk**: High transcriptional similarity within subpopulations makes it difficult to distinguish technical noise from biological variation
- **Manual Grouping Limitations**: The complexity of possible grouping combinations makes manual optimization impractical
- **Cell Type Specificity**: Batch effects vary across different cell types, requiring distinct grouping strategies

### Solution: B2G Framework

We propose **Batch2Group (B2G)**, which infers batch-effect-free groups based on clinical metadata to enable precise batch correction. B2G addresses these challenges through:

1. **Adaptive Prior Selection**: Automatically evaluates and selects informative biological priors using PERMANOVA
2. **Intelligent Grouping**: Identifies patients with similar clinical features that exhibit minimal batch effects
3. **Dual Clustering Support**: Provides both metacell and Leiden clustering for flexible analysis
4. **Group-level Correction**: Performs batch correction at group level rather than individual sample level

### Main Findings

Across multiple single-cell datasets, B2G demonstrates superior performance in:
- **Better Batch Effect Removal**: More effective elimination of technical and batch effects
- **Biological Variation Preservation**: Maintains genuine biological signals while removing technical noise
- **Automated Workflow**: Eliminates need for manual trial-and-error in grouping strategy
- **Scalable Analysis**: Efficiently handles large-scale longitudinal cohort data

---

## Features

- **Adaptive Prior Selection**: Automatically selects informative biological priors using PERMANOVA
- **Flexible Clustering**: Supports both metacell and Leiden clustering methods
- **Batch Effect Mitigation**: Groups batches intelligently to minimize confounding effects
- **Seamless Integration**: Works directly with Scanpy/AnnData workflows
- **Comprehensive Visualization**: Generates dendrograms and evaluation plots

## Installation

### From PyPI (recommended)

```bash
pip install b2g_tools
```

### From Source

```bash
git clone https://github.com/lyotvincent/b2g.git
cd b2g
pip install -e .
```

### Environment Setup with Conda (alternative)

```bash
# Create environment with all dependencies
conda create -n b2g python=3.10 scanpy scikit-learn scikit-bio leidenalg -c conda-forge -c bioconda -y
conda activate b2g

# Install pip-only packages
pip install metacells dynamicTreeCut scib
pip install b2g-tools
```

## User Guide

### Basic Usage (No Prior Knowledge)

```python
import scanpy as sc
import b2g

# Load your data (with raw counts)
adata = sc.read_h5ad('your_data.h5ad')

# Run B2G grouping with default settings
b2g.group(
    adata, 
    batch_key='donor_id',
    method='metacell',
    target_metacell_size=48,
    key_added='groups'
)

# The grouping result is now in adata.obs['groups']
print(adata.obs['groups'].value_counts())

# Continue with your standard scanpy workflow
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=30)

# Use the grouping for batch correction
sc.external.pp.harmony_integrate(adata, key='groups')
```

### Usage with Biological Priors

```python
import scanpy as sc
import b2g

# Load data
adata = sc.read_h5ad('lung_dataset.h5ad')

# Filter to specific cell type (optional but recommended)
adata_subset = adata[adata.obs['cell_type'] == 'Endothelial cells'].copy()

# Run B2G with biological priors
b2g.group(
    adata_subset,
    batch_key='donor_id',
    additional_features=[
        {'column': 'disease', 'description': 'Disease status'},
        {'column': 'sex', 'description': 'Biological sex'},
        {'column': 'self_reported_ethnicity', 'description': 'Ethnicity'},
        {'column': 'development_stage', 'description': 'Development stage'}
    ],
    method='metacell',
    target_metacell_size=48,
    min_priors=1,
    max_priors=3,
    gap_threshold=0.1,
    output_dir='b2g_results',
    fig_path='b2g_results/figures',
    met_path='b2g_results/metrics',
    key_added='groups'
)

# Check grouping results
print(adata_subset.obs['groups'].value_counts())
```

### Using Leiden Clustering Instead of Metacell

```python
b2g.group(
    adata,
    batch_key='donor_id',
    method='leiden',
    leiden_resolution=1.0,
    additional_features=[
        {'column': 'disease', 'description': 'Disease status'},
        {'column': 'sex', 'description': 'Biological sex'}
    ],
    output_dir='b2g_leiden_results',
    fig_path='b2g_leiden_results/figures',
    met_path='b2g_leiden_results/metrics',
    key_added='groups'
)
```



## Parameters

### Main Function: `b2g.group()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | Required | AnnData object with raw counts |
| `batch_key` | str | `'batch'` | Column name in `.obs` for batches |
| `method` | str | `'metacell'` | Clustering method: `'metacell'` or `'leiden'` |
| `additional_features` | list | `None` | Biological priors (see format below) |
| `target_metacell_size` | int | `48` | Target metacell size (metacell method only) |
| `leiden_resolution` | float | `1.0` | Leiden resolution (leiden method only) |
| `min_priors` | int | `1` | Minimum priors to select |
| `max_priors` | int | `None` | Maximum priors to select |
| `gap_threshold` | float | `0.1` | Minimum score threshold for filtering |
| `output_dir` | str | `None` | Output directory path |
| `fig_path` | str | `None` | Figures directory path |
| `met_path` | str | `None` | Metrics directory path |
| `key_added` | str | `None` | Column name for grouping results |

### Additional Features Format

```python
additional_features = [
    {'column': 'column_name_in_obs', 'description': 'Human readable description'},
    # ... more features
]
```

## Input Requirements

- **Data Format**: AnnData object compatible with Scanpy
- **Count Matrix**: Raw UMI counts (integers) required for metacell method
  - Should be in `adata.X` or `adata.raw.X`
  - Not normalized, log-transformed, or scaled
- **Batch Column**: Categorical column in `adata.obs` identifying batches
- **Prior Columns** (optional): Categorical columns in `adata.obs` for biological priors

## Output

### In AnnData Object

- `adata.obs[key_added]`: Batch group assignments (e.g., 'G1', 'G2', 'G3')
- Additional columns created during processing (e.g., `metacell`, `prior_group`)

### Output Files (if paths specified)

```
output_dir/
├── figures/
│   ├── dendrogram.png              # Hierarchical clustering dendrogram
│   ├── prior_selection_results.png # Prior evaluation visualization
│   └── alpha_optimization.png      # Alpha parameter optimization
└── metrics/
    ├── prior_selection_log.csv     # Detailed prior selection log
    ├── alpha_optimization.csv      # Alpha evaluation results
    └── clustering_results/
        ├── distance_matrix_square.npy
        ├── distance_matrix_condensed.npy
        └── linkage_matrix_Z.npy
```

## Advanced Usage

### Custom Configuration

```python
from b2g import AnalysisConfig, group_batches

# Create custom configuration
config = AnalysisConfig()
config.clustering_method = 'metacell'
config.b2g_metacell_params['target_metacell_size'] = 64
config.b2g_min_priors = 2
config.b2g_max_priors = 4
config.leiden_params['resolution'] = 2.0

# Run with custom config
adata = group_batches(adata, config, key_added='custom_groups')
```

### Integration with Batch Correction Methods

```python
import scanpy as sc
import b2g

# Step 1: Load data and subset to specific cell type
adata = sc.read_h5ad('data.h5ad')
adata_subset = adata[adata.obs['cell_type'] == 'Endothelial cells'].copy()

# Step 2: Run B2G grouping
b2g.group(
    adata_subset,
    batch_key='donor_id',
    method='metacell',
    target_metacell_size=48,
    additional_features=[
        {'column': 'disease', 'description': 'Disease status'},
        {'column': 'sex', 'description': 'Biological sex'}
    ],
    key_added='groups'
)

# Step 3: Standard preprocessing
sc.pp.normalize_total(adata_subset, target_sum=1e4)
sc.pp.log1p(adata_subset)
sc.pp.highly_variable_genes(adata_subset, n_top_genes=2000)
sc.pp.pca(adata_subset, n_comps=50)

# Step 4: Use B2G groups for batch correction
# With Harmony
sc.external.pp.harmony_integrate(adata_subset, key='groups')

# With Scanorama
import scanorama
adata_list = [adata_subset[adata_subset.obs['groups'] == g].copy() 
              for g in adata_subset.obs['groups'].unique()]
scanorama.integrate_scanpy(adata_list)

# With Combat
import scanpy.external as sce
sce.pp.combat(adata_subset, key='groups')
```

## How It Works

1. **Prior Selection**: Evaluates biological priors using PERMANOVA to identify informative features
2. **Clustering**: Builds metacells or Leiden clusters, optionally grouped by selected priors
3. **Feature Extraction**: Creates batch-feature matrix from clustering results
4. **Distance Calculation**: Computes weighted distances between batches
5. **Hierarchical Clustering**: Groups batches based on their feature similarity
6. **Dynamic Tree Cutting**: Automatically determines optimal batch groups

---
## Benchmark Datasets
B2G has been validated on multiple large-scale single-cell RNA-seq datasets and imaging datasets:

### 1. Human Skin Dataset
- **Scale**: 155,402 cells × 32,983 genes from 24 samples
- **Sample Types**: Basal cell carcinoma in the face (BCCface), healthy human skin in the face (UV-exposed), and inguino-iliac skin in the body (UV-protected)
- **Cell Types**: T cells, myeloid cells, fibroblasts, pericytes, Neuronal_Schwann cells
- **Clinical Metadata**: Treatment status (group), tissue location, age
- **Original Data**: [ArrayExpress E-MTAB-13085](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-13085/)
- **Processed Data**: [Spatial Skin Atlas](https://spatial-skin-atlas.cellgeni.sanger.ac.uk/)

### 2. Breast Tissue Dataset
- **Scale**: 714,331 cells × 32,383 genes from 126 female donors
- **Cell Types**: Fibroblasts, basal cells, vascular cells, T cells, myeloid cells
- **Clinical Metadata**: Sample source, suspension dissociation time, sequencing platform, tissue location, BMI group, procedure group, age group, breast density, self-reported ethnicity, developmental stage
- **Original Data**: [GEO GSE195665](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195665)
- **Processed Data**: [CZ CELLxGENE](https://cellxgene.cziscience.com/e/842c6f5d-4a94-4eef-8510-8c792d1124bc.cxg/)

### 3. Intestinal Tissue Dataset
- **Scale**: 155,232 cells × 30,172 genes
- **Sample Source**: Multiple endodermal organs of the respiratory and gastrointestinal tracts during human development
- **Cell Types**: Mesenchymal cells, epithelial cells, immune cells, neuronal cells, endothelial cells
- **Clinical Metadata**: Sequencing platform, annotated organ, annotated tissue, sex, tissue type, developmental stage, alignment software
- **Original Data**: [ArrayExpress E-MTAB-10187](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-10187/)
- **Processed Data**: [CZ CELLxGENE](https://cellxgene.cziscience.com/e/9968be68-ab65-4a38-9e1a-c9b6abece194.cxg/)

### 4. Lung Tissue Dataset
- **Scale**: 116,313 nuclei × 33,523 genes from 26 individuals
- **Technology**: Single-nucleus RNA sequencing (snRNA-seq)
- **Sample Source**: Autopsy lung tissues from 19 deceased COVID-19 patients and lung tissues from 7 pre-pandemic control individuals
- **Cell Types**: Epithelial cells, myeloid cells, fibroblasts, endothelial cells, T cells
- **Clinical Metadata**: Disease status, sex, self-reported ethnicity, developmental stage
- **Original Data**: [GEO GSE171524](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171524)
- **Processed Data**: [CZ CELLxGENE](https://cellxgene.cziscience.com/e/d8da613f-e681-4c69-b463-e94f5e66847f.cxg/)

### 5. Cell Painting Dataset
- **Scale**: 259 plates (99,440 wells total) from 12 laboratories
- **Technology**: Open reading frame (ORF) overexpression dataset from the Joint Undertaking for Morphological Profiling (JUMP, cpg0016)
- **Features**: 1,446 selected features (from 7,638 pre-extracted features using CellProfiler for brightfield and fluorescence images)
- **Plate Format**: Typically 16 rows × 24 columns (384 wells per plate)
- **Data Processing**: Excluded Laboratory 12 and BR00123528A plate due to quality issues
- **Data**: 
  - Raw data: [AWS Open Data Registry](https://registry.opendata.aws/cellpainting-gallery/)
  - Metadata and annotations: [JUMP-MOA](https://github.com/jump-cellpainting/JUMP-MOA) and [JUMP Datasets](https://github.com/jump-cellpainting/datasets)


---

## Requirements

**Tested Environment:**

- Python: 3.10.19

**Package Versions:**
```
scanpy: 1.11.5
anndata: 0.11.4
numpy: 2.2.6
pandas: 2.3.3
scikit-learn: 1.7.2
scikit-bio: 0.7.1.post1
metacells: 0.9.5
dynamicTreeCut: 0.1.1
leidenalg: 0.11.0
```

These versions have been tested and verified to work correctly with B2G.