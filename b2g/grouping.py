"""
Main grouping function
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import linkage

from .utils import (
    detect_raw_counts,
    compute_weighted_distance_matrix,
    dynamic_tree_cut_python
)
from .prior_selection import (
    detect_prior_collinearity,
    select_representative_from_collinear_group,
    comprehensive_prior_evaluation,
    prior_selection
)
from .clustering import (
    build_metacells_globally,
    build_metacells_with_prior_knowledge,
    build_leiden_clusters_globally,
    build_leiden_clusters_with_prior_knowledge
)
from .visualization import (
    visualize_prior_selection,
    plot_alpha_evaluation,
    plot_dendrogram
)


def select_alpha_by_calinski_harabasz(metacell_features, alpha_candidates, config):
    """
    Select optimal Alpha based on Calinski-Harabasz Index
    """
    print("\n" + "=" * 70)
    print("Selecting Alpha based on Calinski-Harabasz Index")
    print("=" * 70)

    results = []
    distance_matrices = {}

    for alpha in alpha_candidates:
        print(f"\n--- Evaluating alpha = {alpha} ---")

        try:
            # Calculate distance and linkage
            dist_square, dist_condensed = compute_weighted_distance_matrix(metacell_features, alpha)
            distance_matrices[alpha] = dist_square
            Z = linkage(dist_condensed, method='ward')

            # Dynamic tree cutting
            cluster_labels, _ = dynamic_tree_cut_python(
                Z, dist_square,
                min_cluster_size=config.b2g_dynamic_tree_params['min_cluster_size'],
                deep_split=config.b2g_dynamic_tree_params['deep_split']
            )

            # Handle unassigned samples
            unassigned_mask = cluster_labels == 0
            n_unassigned = unassigned_mask.sum()
            if n_unassigned > 0:
                outlier_label = cluster_labels.max() + 1 if cluster_labels.max() > 0 else 1
                cluster_labels[unassigned_mask] = outlier_label

            # Calculate evaluation metrics
            unique_labels = np.unique(cluster_labels[cluster_labels > 0])
            n_clusters = len(unique_labels)

            if n_clusters >= 2:
                ch_index = calinski_harabasz_score(metacell_features, cluster_labels)
                ch_normalized = ch_index / (1.0 + ch_index)
            else:
                ch_index = ch_normalized = 0.0

            # Cluster size balance
            cluster_sizes = np.array([np.sum(cluster_labels == label) for label in unique_labels])
            cv = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
            balance_penalty = 1.0 / (1.0 + cv)

            final_score = ch_normalized * balance_penalty

            print(f"  Clusters: {n_clusters}, CH Index: {ch_index:.4f}, Final score: {final_score:.4f}")

            results.append({
                'alpha': alpha,
                'n_clusters': n_clusters,
                'n_outliers': n_unassigned,
                'ch_index': ch_index,
                'ch_normalized': ch_normalized,
                'cv': cv,
                'balance_penalty': balance_penalty,
                'final_score': final_score,
                'min_cluster_size': cluster_sizes.min(),
                'max_cluster_size': cluster_sizes.max(),
                'avg_cluster_size': cluster_sizes.mean()
            })
        except Exception as e:
            print(f"  !!! Evaluation failed: {e}")
            continue

    if len(results) == 0:
        raise ValueError("All alpha value evaluations failed!")

    results_df = pd.DataFrame(results)

    # Calculate distance changes
    distance_changes = []
    sorted_alphas = sorted(distance_matrices.keys())
    for i in range(len(sorted_alphas)):
        if i < len(sorted_alphas) - 1:
            diff = distance_matrices[sorted_alphas[i + 1]] - distance_matrices[sorted_alphas[i]]
            change = np.linalg.norm(diff, 'fro')
        else:
            change = np.nan
        distance_changes.append(change)

    results_df['distance_change'] = distance_changes

    # Select optimal alpha
    best_idx = results_df['final_score'].argmax()
    best_alpha = results_df.iloc[best_idx]['alpha']

    print(f"\nOptimal Alpha: {best_alpha}")
    print(results_df.to_string(index=False))

    return best_alpha, results_df


def group_batches(adata, config, key_added='groups_metacell_adaptive'):
    """
    Batch grouping using adaptive prior selection + Metacell/Leiden clustering

    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    config : AnalysisConfig
        Configuration object
    key_added : str
        Column name to add to .obs for grouping

    Returns
    -------
    adata : AnnData
        AnnData object with grouping information added
    """
    print("\n" + "=" * 70)
    method_name = "Metacell" if config.clustering_method == 'metacell' else "Leiden Clustering"

    # Check if using prior knowledge
    use_prior_knowledge = len(config.additional_features) > 0

    if use_prior_knowledge:
        print(f"【Method】Adaptive Prior Selection + {method_name} Grouping")
    else:
        print(f"【Method】Global {method_name} Grouping (No Prior Knowledge)")

    print("=" * 70)

    batch_col = config.column_mapping['batch']
    batches = np.array(sorted(adata.obs[batch_col].unique()))

    # Create output directories
    for dir_path in [config.output_dir, config.fig_path, config.met_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Phase 1: Data preparation
    print("\n【Phase 1】Data Preparation")
    print(f"  Number of cells: {adata.n_obs:,}, Number of batches: {len(batches)}")

    # Check and set data matrix (only for metacell method) - Must be done before QC!
    if config.clustering_method == 'metacell':
        try:
            raw_counts, location = detect_raw_counts(adata)

            # If raw counts are in adata.raw.X, need to move them to adata.X
            if location == 'adata.raw.X':
                print(f"\n| Moving {location} to adata.X for Metacell use")
                # Save current adata.X to layers
                adata.layers['processed_x'] = adata.X.copy()
                # Move raw counts to adata.X (before QC, shapes match)
                adata.X = adata.raw.X.copy()
                # Clear adata.raw to avoid confusion later
                adata.raw = None
                print("  ✓ Saved original adata.X to adata.layers['processed_x']")
                print("  ✓ Moved raw counts to adata.X")
                print("  ✓ Cleared adata.raw")
            else:
                print(f"\n| Using {location} as raw count matrix")

        except ValueError as e:
            print(f"\n{e}")
            raise
    else:
        print("\n| Using Leiden clustering method, data will be normalized before clustering")

    # Basic QC (after processing raw data)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After QC: {adata.shape}")

    # Convert data types
    adata.obs[batch_col] = adata.obs[batch_col].astype('category')
    for feat in config.additional_features:
        adata.obs[feat['column']] = adata.obs[feat['column']].astype('category')

    if use_prior_knowledge:
        # Complete workflow with prior knowledge

        # Calculate PCA for PERMANOVA evaluation
        if 'X_pca' not in adata.obsm:
            print(f"  Computing PCA (for prior evaluation)...")
            X_original = adata.X
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.tl.pca(adata, n_comps=min(50, adata.shape[0] - 1, adata.shape[1] - 1))
            adata.X = X_original
            del X_original
            gc.collect()

        use_embedding = 'X_pca'

        # Phase 2: Evaluate prior importance
        print("\n【Phase 2】Evaluating Prior Importance")

        prior_columns = [f['column'] for f in config.additional_features]
        evaluation_results = []

        for prior in prior_columns:
            result = comprehensive_prior_evaluation(
                adata, config, prior, batch_col, use_embedding, n_permutations=999
            )
            evaluation_results.append(result)

        evaluation_df = pd.DataFrame(evaluation_results).sort_values('final_score', ascending=False)

        # Filter high confounding priors
        print("\n【Filtering High Confounding Priors】")
        high_confounding = ['Complete Confounding', 'Severe Confounding']
        low_confounding_mask = ~evaluation_df['confounding_type'].isin(high_confounding)
        low_confounding_priors = evaluation_df[low_confounding_mask]['prior'].tolist()

        if len(low_confounding_priors) == 0:
            print("!!! All priors are highly confounded, forcing to keep highest score")
            low_confounding_priors = [evaluation_df.iloc[0]['prior']]

        # Detect collinearity
        print("\n【Detecting Collinearity】")
        if len(low_confounding_priors) > 1:
            collinear_groups, independent_priors = detect_prior_collinearity(
                adata, low_confounding_priors, threshold=0.95
            )
            priors_to_use = list(independent_priors)
            for group in collinear_groups:
                rep = select_representative_from_collinear_group(group, evaluation_df)
                priors_to_use.append(rep)
        else:
            priors_to_use = low_confounding_priors

        # Save evaluation results
        evaluation_df.to_csv(f"{config.met_path}/prior_evaluation.csv", index=False)

        # Phase 3: Select important priors
        print("\n【Phase 3】Selecting Important Priors")
        evaluation_df_filtered = evaluation_df[evaluation_df['prior'].isin(priors_to_use)]
        selected_priors, selection_log = prior_selection(evaluation_df_filtered, config)

        # Save selection log
        selection_log_df = pd.DataFrame(selection_log)
        selection_log_path = f"{config.met_path}/prior_selection_log.csv"
        selection_log_df.to_csv(selection_log_path, index=False)
        print(f"\n| Selection log saved to: {selection_log_path}")

        # Visualization
        try:
            visualize_prior_selection(
                evaluation_df, selected_priors, config,
                threshold=0.5 * np.mean([c['final_score'] for c in evaluation_df_filtered.to_dict('records')]),
                top_score=evaluation_df_filtered.iloc[0]['final_score']
            )
        except Exception as e:
            print(f"\n!!! Visualization failed: {e}")

        # Update config
        config.additional_features = [f for f in config.additional_features
                                      if f['column'] in selected_priors]

        # Phase 4: Build clustering/Metacells (with prior grouping)
        print(f"\n【Phase 4】Building {method_name}")
        print(f"Using priors: {selected_priors}")

        if config.clustering_method == 'metacell':
            feature_data_prior, adata = build_metacells_with_prior_knowledge(adata, config)
            feature_data_global, adata = build_metacells_globally(adata, config)
        else:  # leiden
            feature_data_prior, adata = build_leiden_clusters_with_prior_knowledge(adata, config)
            feature_data_global, adata = build_leiden_clusters_globally(adata, config)

        # Merge
        if feature_data_global is not None and feature_data_global.n_obs > 0:
            if 'build_type' not in feature_data_prior.obs.columns:
                feature_data_prior.obs['build_type'] = 'prior_grouped'
            feature_data = ad.concat([feature_data_prior, feature_data_global],
                                     join='outer', merge='same')
        else:
            feature_data = feature_data_prior
            if 'build_type' not in feature_data.obs.columns:
                feature_data.obs['build_type'] = 'prior_grouped'

        # Phase 5: Build batch-feature matrix (with priors)
        print(f"\n【Phase 5】Building Batch-{method_name} Feature Matrix")

        feature_ids = list(range(feature_data.n_obs))
        feature_df = pd.DataFrame(0, index=batches, columns=feature_ids)

        for batch in batches:
            batch_mask = adata.obs[batch_col] == batch

            # Prior-grouped clusters/metacells
            if config.clustering_method == 'metacell':
                prior_mask = batch_mask & (adata.obs['metacell'] >= 0)
            else:
                prior_mask = batch_mask & (adata.obs['leiden_cluster'] != '-1')

            if prior_mask.sum() > 0:
                cluster_series = adata.obs.loc[prior_mask,
                'metacell' if config.clustering_method == 'metacell' else 'leiden_cluster']
                if config.clustering_method == 'leiden':
                    cluster_series = cluster_series.astype(int)

                for cluster_id, count in cluster_series.value_counts().items():
                    if cluster_id < len(feature_ids):
                        feature_df.loc[batch, cluster_id] = count

            # Globally built clusters/metacells
            if config.clustering_method == 'metacell':
                global_mask = batch_mask & (adata.obs['global_metacell'] >= 0)
            else:
                global_mask = batch_mask & (adata.obs['global_leiden_cluster'] != '-1')

            if global_mask.sum() > 0:
                n_prior = (feature_data.obs['build_type'] == 'prior_grouped').sum()
                global_cluster_series = adata.obs.loc[global_mask,
                'global_metacell' if config.clustering_method == 'metacell' else 'global_leiden_cluster']
                if config.clustering_method == 'leiden':
                    global_cluster_series = global_cluster_series.astype(int)

                for cluster_id, count in global_cluster_series.value_counts().items():
                    combined_id = n_prior + cluster_id
                    if combined_id < len(feature_ids):
                        feature_df.loc[batch, combined_id] = count

    else:
        # Simplified workflow without prior knowledge

        print("\n【Skipping Phase 2-3】No prior knowledge, skipping PCA computation and prior evaluation")
        selected_priors = None

        # Phase 4: Global build only
        print(f"\n【Phase 4】Global {method_name} Building")

        if config.clustering_method == 'metacell':
            feature_data, adata = build_metacells_globally(adata, config)
            cluster_id_col = 'global_metacell'
        else:  # leiden
            feature_data, adata = build_leiden_clusters_globally(adata, config)
            cluster_id_col = 'global_leiden_cluster'

        if feature_data is None or feature_data.n_obs == 0:
            raise ValueError("Global build failed, cannot continue")

        # Ensure build_type column exists
        if 'build_type' not in feature_data.obs.columns:
            feature_data.obs['build_type'] = 'global'

        # Phase 5: Build batch-feature matrix (without priors)
        print(f"\n【Phase 5】Building Batch-{method_name} Feature Matrix")

        feature_ids = list(range(feature_data.n_obs))
        feature_df = pd.DataFrame(0, index=batches, columns=feature_ids)

        for batch in batches:
            batch_mask = adata.obs[batch_col] == batch

            # Global build only
            if config.clustering_method == 'metacell':
                global_mask = batch_mask & (adata.obs[cluster_id_col] >= 0)
            else:
                global_mask = batch_mask & (adata.obs[cluster_id_col] != '-1')

            if global_mask.sum() > 0:
                cluster_series = adata.obs.loc[global_mask, cluster_id_col]
                if config.clustering_method == 'leiden':
                    cluster_series = cluster_series.astype(int)

                for cluster_id, count in cluster_series.value_counts().items():
                    if cluster_id < len(feature_ids):
                        feature_df.loc[batch, cluster_id] = count

    # Phase 6-7: Standardization and Alpha selection (applies to both modes)
    # Standardization
    valid_batches = feature_df.sum(axis=1) > 0
    feature_df = feature_df[valid_batches]
    valid_batch_names = feature_df.index.tolist()

    feature_normdf = feature_df.div(feature_df.sum(axis=1), axis=0)
    feature_matrix = StandardScaler().fit_transform(feature_normdf)

    print(f"  Feature matrix: {feature_matrix.shape}")

    # Phase 6: Select optimal Alpha
    print("\n【Phase 6】Selecting Optimal Alpha")

    alpha_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    best_alpha, alpha_results = select_alpha_by_calinski_harabasz(
        feature_matrix, alpha_candidates, config
    )

    alpha_results.to_csv(f"{config.met_path}/alpha_optimization.csv", index=False)
    plot_alpha_evaluation(alpha_results, config.fig_path)

    # Phase 7: Final clustering
    print("\n【Phase 7】Final Clustering")
    print(f"Using Alpha = {best_alpha}")

    dist_square, dist_condensed = compute_weighted_distance_matrix(feature_matrix, best_alpha)
    Z = linkage(dist_condensed, method='ward')

    # Save clustering results
    save_dir = os.path.join(config.met_path, 'clustering_results')
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'distance_matrix_square.npy'), dist_square)
    np.save(os.path.join(save_dir, 'distance_matrix_condensed.npy'), dist_condensed)
    np.save(os.path.join(save_dir, 'linkage_matrix_Z.npy'), Z)

    # Dynamic tree cutting
    cluster_labels, _ = dynamic_tree_cut_python(
        Z, dist_square,
        min_cluster_size=config.b2g_dynamic_tree_params['min_cluster_size'],
        deep_split=config.b2g_dynamic_tree_params['deep_split']
    )

    # Handle unassigned samples
    unassigned_mask = cluster_labels == 0
    if unassigned_mask.sum() > 0:
        if config.b2g_dynamic_tree_params.get('unassigned_as_outlier_group', True):
            outlier_label = cluster_labels.max() + 1 if cluster_labels.max() > 0 else 1
            cluster_labels[unassigned_mask] = outlier_label
            print(f"  Unassigned samples ({unassigned_mask.sum()}) grouped separately: G{outlier_label}")

    # Plot dendrogram
    plot_dendrogram(Z, valid_batch_names, cluster_labels, config, method_name,
                    use_prior_knowledge, selected_priors)

    # Organize grouping results
    final_groups = {}
    for i, batch in enumerate(valid_batch_names):
        group_name = f"G{int(cluster_labels[i])}"
        final_groups.setdefault(group_name, []).append(batch)

    # Write back to adata
    batch_to_group = {batch: f"G{int(cluster_labels[i])}"
                      for i, batch in enumerate(valid_batch_names)}
    adata.obs[key_added] = adata.obs[batch_col].map(batch_to_group).fillna('Unknown')

    print(f"\nComplete: {len(final_groups)} groups")
    for group, batches_list in sorted(final_groups.items()):
        print(f"  {group}: {len(batches_list)} batches")

    return adata