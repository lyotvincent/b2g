"""
Clustering functions - Metacell and Leiden
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse
import metacells as mc
import gc
from itertools import product


def build_metacells_pipeline(adata_subset, config, group_name=""):
    """
    Metacells construction pipeline
    """
    print(f"\n{'=' * 70}")
    print(f"Building Metacells{f' - {group_name}' if group_name else ''}")
    print(f"{'=' * 70}")

    n_cells = adata_subset.n_obs
    print(f"  Number of cells: {n_cells:,}")

    # Check if cell count is sufficient
    min_threshold = config.b2g_metacell_params['target_metacell_size']
    if n_cells < min_threshold:
        print(f"  !!! Insufficient cells, skipping")
        return None

    # Ensure data type is float32
    if adata_subset.X.dtype != np.float32:
        adata_subset.X = adata_subset.X.astype(np.float32)

    try:
        # 1. Exclude genes
        mc.pl.exclude_genes(
            adata_subset,
            excluded_gene_names=config.b2g_metacell_params['excluded_gene_names'],
            excluded_gene_patterns=config.b2g_metacell_params['excluded_gene_patterns'],
            random_seed=config.b2g_metacell_params['random_seed']
        )

        # 2. Manually add excluded_cell column
        adata_subset.obs['excluded_cell'] = False

        # 3. Extract clean data
        mc.pl.extract_clean_data(adata_subset)

        # 4. Mark lateral genes
        mc.pl.mark_lateral_genes(
            adata_subset,
            lateral_gene_names=config.b2g_metacell_params['lateral_gene_names'],
            lateral_gene_patterns=config.b2g_metacell_params['lateral_gene_patterns']
        )

        # 5. Mark noisy genes
        mc.pl.mark_noisy_genes(adata_subset)

        # 6. Run divide and conquer pipeline
        mc.pl.divide_and_conquer_pipeline(
            adata_subset,
            target_metacell_size=config.b2g_metacell_params['target_metacell_size'],
            random_seed=config.b2g_metacell_params['random_seed']
        )

        # 7. Collect metacells results
        mdata = mc.pl.collect_metacells(
            adata_subset,
            name=f'metacells_{group_name}',
            random_seed=config.b2g_metacell_params['random_seed']
        )

        n_metacells = mdata.n_obs
        compression_ratio = n_cells / n_metacells if n_metacells > 0 else 0

        print(f"  | Complete: {n_metacells} metacells, compression ratio {compression_ratio:.2f}x")

        return mdata, adata_subset

    except Exception as e:
        print(f"  !!! Construction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_metacells_globally(adata, config):
    """
    Build metacells globally (without using prior knowledge grouping)
    """
    result = build_metacells_pipeline(adata.copy(), config, "global")

    if result is None:
        return None, adata

    mdata_global, adata_global = result

    # Write global_metacell information back to original adata
    adata.obs['global_metacell'] = -1
    adata.obs['global_metacell'] = adata_global.obs['metacell'].values

    # Add identifier
    mdata_global.obs['build_type'] = 'global'
    mdata_global.obs.index = [f"global_metacell_{i}" for i in range(mdata_global.n_obs)]

    return mdata_global, adata


def build_metacells_with_prior_knowledge(adata, config):
    """
    Build metacells using prior knowledge grouping
    """
    print("\n" + "=" * 70)
    print("Building Metacells with Prior Knowledge Grouping")
    print("=" * 70)

    # Ensure data type
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)

    # Get prior columns
    prior_columns = [feat['column'] for feat in config.additional_features]

    # Generate all valid combinations
    unique_values = {col: adata.obs[col].unique().tolist() for col in prior_columns}
    all_combinations = list(product(*[unique_values[col] for col in prior_columns]))

    # Filter valid combinations
    valid_combinations = []
    for combo in all_combinations:
        mask = np.all([adata.obs[col] == val for col, val in zip(prior_columns, combo)], axis=0)
        if mask.sum() > 0:
            valid_combinations.append(combo)

    print(f"Number of valid combinations: {len(valid_combinations)}")

    # Initialize
    all_mdata_list = []
    metacell_offset = 0
    adata.obs['metacell'] = -1
    adata.obs['prior_group'] = ""

    # Build metacells for each combination
    for combo_idx, combo in enumerate(valid_combinations):
        combo_str = ' | '.join([f"{col}={val}" for col, val in zip(prior_columns, combo)])
        print(f"\n--- Combination {combo_idx + 1}/{len(valid_combinations)}: {combo_str} ---")

        # Filter cells
        mask = np.all([adata.obs[col] == val for col, val in zip(prior_columns, combo)], axis=0)
        adata_subset = adata[mask].copy()

        # Build metacells
        result = build_metacells_pipeline(adata_subset, config, f"combo{combo_idx}")

        if result is None:
            continue

        mdata_subset, adata_subset = result
        n_metacells = mdata_subset.n_obs

        if n_metacells > 0:
            # Adjust metacell IDs
            valid_mc_mask = adata_subset.obs['metacell'] >= 0
            adata_subset.obs.loc[valid_mc_mask, 'metacell'] += metacell_offset

            # Write back to original adata
            adata.obs.loc[mask, 'metacell'] = adata_subset.obs['metacell'].values
            adata.obs.loc[mask, 'prior_group'] = combo_str

            # Add identifier
            mdata_subset.obs['prior_group'] = combo_str
            mdata_subset.obs.index = [f"metacell_{metacell_offset + i}" for i in range(n_metacells)]

            all_mdata_list.append(mdata_subset)
            metacell_offset += n_metacells

        gc.collect()

    if len(all_mdata_list) == 0:
        raise ValueError("Failed to build any metacells")

    # Merge
    mdata_merged = ad.concat(all_mdata_list, join='outer', merge='same') if len(all_mdata_list) > 1 else all_mdata_list[0]

    print(f"\nComplete: {mdata_merged.n_obs} metacells, {len(all_mdata_list)} groups")

    return mdata_merged, adata


def build_leiden_clusters_pipeline(adata_subset, config, group_name=""):
    """
    Leiden clustering construction pipeline
    """
    print(f"\n{'=' * 70}")
    print(f"Building Leiden Clusters{f' - {group_name}' if group_name else ''}")
    print(f"{'=' * 70}")

    n_cells = adata_subset.n_obs
    print(f"  Number of cells: {n_cells:,}")

    # Check if cell count is sufficient
    if n_cells < config.leiden_params['min_cluster_size']:
        print(f"  !!! Insufficient cells, skipping")
        return None

    try:
        # Data preprocessing
        adata_work = adata_subset.copy()

        # Normalization and log transformation
        sc.pp.normalize_total(adata_work, target_sum=1e4)
        sc.pp.log1p(adata_work)

        # Highly variable genes
        sc.pp.highly_variable_genes(adata_work, n_top_genes=2000)

        # PCA
        n_pcs = min(config.leiden_params['n_pcs'], n_cells - 1, adata_work.n_vars - 1)
        sc.tl.pca(adata_work, n_comps=n_pcs, svd_solver='arpack')

        # Neighbor graph
        sc.pp.neighbors(adata_work,
                        n_neighbors=min(config.leiden_params['n_neighbors'], n_cells - 1),
                        n_pcs=n_pcs,
                        random_state=config.leiden_params['random_state'])

        # Leiden clustering
        sc.tl.leiden(adata_work,
                     resolution=config.leiden_params['resolution'],
                     random_state=config.leiden_params['random_state'],
                     key_added='leiden_cluster')

        # Filter small clusters
        cluster_counts = adata_work.obs['leiden_cluster'].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= config.leiden_params['min_cluster_size']].index

        # Mark small clusters as -1
        adata_work.obs['leiden_cluster_filtered'] = adata_work.obs['leiden_cluster'].astype(str)
        small_cluster_mask = ~adata_work.obs['leiden_cluster'].isin(valid_clusters)
        adata_work.obs.loc[small_cluster_mask, 'leiden_cluster_filtered'] = '-1'

        n_valid_clusters = len(valid_clusters)
        n_filtered = small_cluster_mask.sum()

        print(f"  Original clusters: {len(cluster_counts)}")
        print(f"  Valid clusters: {n_valid_clusters} (size >= {config.leiden_params['min_cluster_size']})")
        if n_filtered > 0:
            print(f"  Filtered cells: {n_filtered} ({n_filtered / n_cells * 100:.1f}%)")

        # Calculate cluster centers
        cluster_centers = []
        cluster_ids = []

        for cluster_id in sorted(valid_clusters, key=lambda x: int(x)):
            cluster_mask = adata_work.obs['leiden_cluster_filtered'] == cluster_id
            cluster_cells = adata_work[cluster_mask]

            # Use normalized data to calculate mean
            if scipy.sparse.issparse(cluster_cells.X):
                center = np.array(cluster_cells.X.mean(axis=0)).flatten()
            else:
                center = cluster_cells.X.mean(axis=0)

            cluster_centers.append(center)
            cluster_ids.append(cluster_id)

        # Create AnnData object for cluster centers
        cluster_adata = ad.AnnData(
            X=np.array(cluster_centers),
            var=adata_work.var,
            obs=pd.DataFrame(index=[f"{group_name}_cluster_{cid}" for cid in cluster_ids])
        )

        # Write cluster information back to original data
        adata_subset.obs['leiden_cluster'] = '-1'
        adata_subset.obs.loc[adata_work.obs.index, 'leiden_cluster'] = \
            adata_work.obs['leiden_cluster_filtered'].values

        compression_ratio = n_cells / n_valid_clusters if n_valid_clusters > 0 else 0
        print(f"  | Complete: {n_valid_clusters} cluster centers, compression ratio {compression_ratio:.2f}x")

        return cluster_adata, adata_subset

    except Exception as e:
        print(f"  !!! Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_leiden_clusters_globally(adata, config):
    """
    Build Leiden clusters globally
    """
    result = build_leiden_clusters_pipeline(adata.copy(), config, "global")

    if result is None:
        return None, adata

    cluster_adata_global, adata_global = result

    # Write global_leiden_cluster information back to original adata
    adata.obs['global_leiden_cluster'] = '-1'
    adata.obs['global_leiden_cluster'] = adata_global.obs['leiden_cluster'].values

    # Add identifier
    cluster_adata_global.obs['build_type'] = 'global'
    cluster_adata_global.obs.index = [f"global_cluster_{i}"
                                      for i in range(cluster_adata_global.n_obs)]

    return cluster_adata_global, adata


def build_leiden_clusters_with_prior_knowledge(adata, config):
    """
    Build Leiden clusters using prior knowledge grouping
    """
    print("\n" + "=" * 70)
    print("Building Leiden Clusters with Prior Knowledge Grouping")
    print("=" * 70)

    # Get prior columns
    prior_columns = [feat['column'] for feat in config.additional_features]

    # Generate all valid combinations
    unique_values = {col: adata.obs[col].unique().tolist() for col in prior_columns}
    all_combinations = list(product(*[unique_values[col] for col in prior_columns]))

    # Filter valid combinations
    valid_combinations = []
    for combo in all_combinations:
        mask = np.all([adata.obs[col] == val for col, val in zip(prior_columns, combo)], axis=0)
        if mask.sum() > 0:
            valid_combinations.append(combo)

    print(f"Number of valid combinations: {len(valid_combinations)}")

    # Initialize
    all_cluster_adata_list = []
    cluster_offset = 0
    adata.obs['leiden_cluster'] = '-1'
    adata.obs['prior_group'] = ""

    # Build clusters for each combination
    for combo_idx, combo in enumerate(valid_combinations):
        combo_str = ' | '.join([f"{col}={val}" for col, val in zip(prior_columns, combo)])
        print(f"\n--- Combination {combo_idx + 1}/{len(valid_combinations)}: {combo_str} ---")

        # Filter cells
        mask = np.all([adata.obs[col] == val for col, val in zip(prior_columns, combo)], axis=0)
        adata_subset = adata[mask].copy()

        # Build clusters
        result = build_leiden_clusters_pipeline(adata_subset, config, f"combo{combo_idx}")

        if result is None:
            continue

        cluster_adata_subset, adata_subset = result
        n_clusters = cluster_adata_subset.n_obs

        if n_clusters > 0:
            # Adjust cluster IDs (convert to integer for offset)
            valid_cluster_mask = adata_subset.obs['leiden_cluster'] != '-1'
            if valid_cluster_mask.sum() > 0:
                old_clusters = adata_subset.obs.loc[valid_cluster_mask, 'leiden_cluster'].astype(int)
                adata_subset.obs.loc[valid_cluster_mask, 'leiden_cluster'] = \
                    (old_clusters + cluster_offset).astype(str)

            # Write back to original adata
            adata.obs.loc[mask, 'leiden_cluster'] = adata_subset.obs['leiden_cluster'].values
            adata.obs.loc[mask, 'prior_group'] = combo_str

            # Add identifier
            cluster_adata_subset.obs['prior_group'] = combo_str
            cluster_adata_subset.obs.index = [f"cluster_{cluster_offset + i}"
                                              for i in range(n_clusters)]

            all_cluster_adata_list.append(cluster_adata_subset)
            cluster_offset += n_clusters

        gc.collect()

    if len(all_cluster_adata_list) == 0:
        raise ValueError("Failed to build any clusters")

    # Merge
    cluster_adata_merged = ad.concat(all_cluster_adata_list, join='outer', merge='same') \
        if len(all_cluster_adata_list) > 1 else all_cluster_adata_list[0]

    print(f"\nComplete: {cluster_adata_merged.n_obs} cluster centers, {len(all_cluster_adata_list)} groups")

    return cluster_adata_merged, adata