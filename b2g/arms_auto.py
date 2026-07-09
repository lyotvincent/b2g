"""
ARMS-based auto mode helpers.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


def prepare_adata_for_ARMS(adata, name):
    print(f"Preparing {name}...")
    if "log1p" not in adata.uns_keys():
        adata.uns["log1p"] = {"base": None}
    elif "base" not in adata.uns["log1p"]:
        adata.uns["log1p"]["base"] = None

    if adata.raw is None:
        adata.raw = adata.copy()
    return adata


def _normalize_harmony_series(series: pd.Series, missing_label: str = "__MISSING__") -> pd.Series:
    normalized = series.astype(object).copy()
    normalized[pd.isna(normalized)] = missing_label
    normalized = normalized.map(lambda value: str(value).strip())
    normalized = normalized.replace(
        {
            "": missing_label,
            "nan": missing_label,
            "None": missing_label,
            "<NA>": missing_label,
        }
    )
    return normalized.astype("category")


def cluster_mapping_align(
    adata_ref: sc.AnnData,
    method1_key: str,
    method2_key: str,
) -> Tuple[sc.AnnData, pd.DataFrame]:
    print("\n[ARMS] Step 1: Exact cell-overlap cluster mapping")

    method1_cluster2cells = {
        clu: set(adata_ref.obs_names[adata_ref.obs[method1_key] == clu])
        for clu in adata_ref.obs[method1_key].cat.categories
    }
    method2_cluster2cells = {
        clu: set(adata_ref.obs_names[adata_ref.obs[method2_key] == clu])
        for clu in adata_ref.obs[method2_key].cat.categories
    }

    method2_to_method1 = {}
    mapping_detail = []
    for method2_clu, method2_cells in method2_cluster2cells.items():
        max_overlap, best_method1_clu = 0.0, None
        for method1_clu, method1_cells in method1_cluster2cells.items():
            overlap_ratio = len(method2_cells & method1_cells) / len(method2_cells)
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_method1_clu = method1_clu
        method2_to_method1[method2_clu] = best_method1_clu
        mapping_detail.append(
            {
                f"{method2_key}_original_cluster": method2_clu,
                f"{method1_key}_matched_cluster": best_method1_clu,
                "cell_overlap_ratio": max_overlap,
            }
        )

    adata_ref.obs[f"{method1_key}_aligned"] = adata_ref.obs[method1_key].astype(str)
    adata_ref.obs[f"{method2_key}_aligned"] = adata_ref.obs[method2_key].map(method2_to_method1).astype(str)
    return adata_ref, pd.DataFrame(mapping_detail)


def get_aligned_cluster_markers(
    adata_ref: sc.AnnData,
    method1_aligned_key: str,
    method2_aligned_key: str,
    n_genes: int = 200,
    top_marker_num: int = 30,
    padj_cutoff: float = 0.01,
    logfc_cutoff: float = 0.8,
    rank_key_prefix: str = "aligned_markers",
) -> Tuple[List[Dict[str, List[str]]], List[str]]:
    for key in [method1_aligned_key, method2_aligned_key]:
        if key not in adata_ref.obs.columns:
            raise KeyError(f"Aligned cluster column {key} not found in adata.obs")

    marker_dict_list = []
    all_cluster_sets = []
    for key in [method1_aligned_key, method2_aligned_key]:
        print(f"\n[ARMS] Mining markers for {key}...")
        sc.tl.rank_genes_groups(
            adata_ref,
            groupby=key,
            n_genes=n_genes,
            key_added=f"{rank_key_prefix}_{key}",
        )

        marker_dict = {}
        for clu in adata_ref.obs[key].cat.categories:
            df = sc.get.rank_genes_groups_df(adata_ref, group=clu, key=f"{rank_key_prefix}_{key}")
            df_filtered = df[(df["pvals_adj"] < padj_cutoff) & (df["logfoldchanges"] > logfc_cutoff)]
            marker_dict[clu] = df_filtered["names"].tolist()[:top_marker_num]

        marker_dict_list.append(marker_dict)
        all_cluster_sets.append(set(marker_dict.keys()))

    common_aligned_clusters = sorted(list(set.intersection(*all_cluster_sets))) if all_cluster_sets else []
    return marker_dict_list, common_aligned_clusters


def calculate_ARMS(
    adata_ref: sc.AnnData,
    method1_marker_dict: Dict[str, List[str]],
    method2_marker_dict: Dict[str, List[str]],
    aligned_clusters: List[str],
    method1_aligned_key: str,
    method2_aligned_key: str,
    method1_name: str,
    method2_name: str,
    min_marker_num: int = 5,
    min_common_marker: int = 3,
    min_cell_recall_ratio: float = 0.0,
) -> Tuple[pd.DataFrame, float, float]:
    print(f"\n[ARMS] Calculating ARMS | {method1_name} vs {method2_name}")
    score_adata = adata_ref.copy()
    arms_result = []
    total_cells = score_adata.n_obs

    common_marker_pool = []
    for clu in aligned_clusters:
        cm = set(method1_marker_dict.get(clu, [])) & set(method2_marker_dict.get(clu, []))
        common_marker_pool.extend(cm)
    common_marker_pool = list(set(common_marker_pool))

    if not common_marker_pool:
        return pd.DataFrame(), 0.0, 0.0

    if not common_marker_pool:
        return pd.DataFrame(), 0.0, 0.0

    sc.tl.score_genes(adata_ref, gene_list=common_marker_pool, score_name="_global_score")
    global_scores = adata_ref.obs["_global_score"]
    q1, q3 = np.percentile(global_scores, [25, 75])
    iqr = q3 - q1
    low = q1 - 3 * iqr
    high = q3 + 3 * iqr

    def robust_normalize(value):
        if pd.isna(value):
            return np.nan
        if high <= low:
            return 0.5
        return np.clip((value - low) / (high - low), 0.0, 1.0)

    for aligned_clu in aligned_clusters:
        method1_markers = method1_marker_dict.get(aligned_clu, [])
        method2_markers = method2_marker_dict.get(aligned_clu, [])
        if len(method1_markers) < min_marker_num or len(method2_markers) < min_marker_num:
            continue

        common_markers = list(set(method1_markers) & set(method2_markers))
        if len(common_markers) < min_common_marker:
            continue

        method1_cell_cnt = score_adata.obs[score_adata.obs[method1_aligned_key] == aligned_clu].shape[0]
        method2_cell_cnt = score_adata.obs[score_adata.obs[method2_aligned_key] == aligned_clu].shape[0]
        method1_cell_ratio = method1_cell_cnt / total_cells if total_cells > 0 else 0
        method2_cell_ratio = method2_cell_cnt / total_cells if total_cells > 0 else 0
        if method1_cell_cnt == 0 or method2_cell_cnt == 0:
            continue
        if method1_cell_ratio < min_cell_recall_ratio or method2_cell_ratio < min_cell_recall_ratio:
            continue

        method1_weight = method1_cell_cnt / total_cells if total_cells > 0 else 1.0
        method2_weight = method2_cell_cnt / total_cells if total_cells > 0 else 1.0

        sc.tl.score_genes(score_adata, gene_list=common_markers, score_name="_tmp_common")
        method1_score = score_adata.obs.loc[
            score_adata.obs[method1_aligned_key] == aligned_clu, "_tmp_common"
        ].mean()
        method2_score = score_adata.obs.loc[
            score_adata.obs[method2_aligned_key] == aligned_clu, "_tmp_common"
        ].mean()

        method1_arms = np.clip(robust_normalize(method1_score) * method1_weight, 0.0, 1.0)
        method2_arms = np.clip(robust_normalize(method2_score) * method2_weight, 0.0, 1.0)
        common_ratio_method1 = len(common_markers) / len(method1_markers)
        common_ratio_method2 = len(common_markers) / len(method2_markers)

        arms_result.append(
            {
                "aligned_cluster": aligned_clu,
                f"{method1_name}_marker_count": len(method1_markers),
                f"{method2_name}_marker_count": len(method2_markers),
                "common_marker_count": len(common_markers),
                f"common_marker_ratio_vs_{method1_name}(%)": round(common_ratio_method1 * 100, 2),
                f"common_marker_ratio_vs_{method2_name}(%)": round(common_ratio_method2 * 100, 2),
                f"{method1_name}_cell_count": method1_cell_cnt,
                f"{method2_name}_cell_count": method2_cell_cnt,
                f"{method1_name}_cluster_weight": round(method1_weight, 4),
                f"{method2_name}_cluster_weight": round(method2_weight, 4),
                f"{method1_name}_ARMS": round(method1_arms, 4),
                f"{method2_name}_ARMS": round(method2_arms, 4),
                "ARMS_improvement": round(method2_arms - method1_arms, 4),
                "ARMS_improvement_rate(%)": round((method2_arms / method1_arms - 1) * 100, 2)
                if method1_arms > 0
                else 0.0,
            }
        )

    arms_df = pd.DataFrame(arms_result)
    arms_df_clean = arms_df.dropna()
    if arms_df_clean.empty:
        return arms_df, 0.0, 0.0

    method1_arms_mean = round(arms_df_clean[f"{method1_name}_ARMS"].sum(), 4)
    method2_arms_mean = round(arms_df_clean[f"{method2_name}_ARMS"].sum(), 4)
    return arms_df, method1_arms_mean, method2_arms_mean


def run_ARMS_analysis(
    adata_ref: sc.AnnData,
    method1_cluster_key: str,
    method2_cluster_key: str,
    top_marker_num: int = 50,
) -> Tuple[pd.DataFrame, float, float]:
    adata_ref, _ = cluster_mapping_align(adata_ref, method1_cluster_key, method2_cluster_key)
    marker_dict_list_forward, aligned_clusters_forward = get_aligned_cluster_markers(
        adata_ref=adata_ref,
        method1_aligned_key=f"{method1_cluster_key}_aligned",
        method2_aligned_key=f"{method2_cluster_key}_aligned",
        top_marker_num=top_marker_num,
        padj_cutoff=0.01,
        logfc_cutoff=0.8,
    )
    method1_markers = marker_dict_list_forward[0]
    method2_markers = marker_dict_list_forward[1]
    arms_df_forward, method1_arms_forward, method2_arms_forward = calculate_ARMS(
        adata_ref=adata_ref,
        method1_marker_dict=method1_markers,
        method2_marker_dict=method2_markers,
        aligned_clusters=aligned_clusters_forward,
        method1_aligned_key=f"{method1_cluster_key}_aligned",
        method2_aligned_key=f"{method2_cluster_key}_aligned",
        method1_name=method1_cluster_key,
        method2_name=method2_cluster_key,
        min_marker_num=5,
    )

    adata_ref, _ = cluster_mapping_align(adata_ref, method2_cluster_key, method1_cluster_key)
    marker_dict_list_reverse, aligned_clusters_reverse = get_aligned_cluster_markers(
        adata_ref=adata_ref,
        method1_aligned_key=f"{method2_cluster_key}_aligned",
        method2_aligned_key=f"{method1_cluster_key}_aligned",
        top_marker_num=top_marker_num,
        padj_cutoff=0.01,
        logfc_cutoff=0.8,
    )
    method2_markers_rev = marker_dict_list_reverse[0]
    method1_markers_rev = marker_dict_list_reverse[1]
    arms_df_reverse, method2_arms_reverse, method1_arms_reverse = calculate_ARMS(
        adata_ref=adata_ref,
        method1_marker_dict=method2_markers_rev,
        method2_marker_dict=method1_markers_rev,
        aligned_clusters=aligned_clusters_reverse,
        method1_aligned_key=f"{method2_cluster_key}_aligned",
        method2_aligned_key=f"{method1_cluster_key}_aligned",
        method1_name=method2_cluster_key,
        method2_name=method1_cluster_key,
        min_marker_num=5,
    )

    final_arms_df = pd.concat([arms_df_forward, arms_df_reverse], ignore_index=True)
    method1_final_arms = (method1_arms_forward + method1_arms_reverse) / 2
    method2_final_arms = (method2_arms_forward + method2_arms_reverse) / 2
    return final_arms_df, method1_final_arms, method2_final_arms


def _run_harmony_and_cluster(
    adata: sc.AnnData,
    group_key: str,
    batch_key: str,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    leiden_resolution: float,
    max_iter_harmony: int,
    cluster_key: str,
    embedding_key: str,
) -> sc.AnnData:
    import harmonypy as hm

    adata_work = adata.copy()
    prepare_adata_for_ARMS(adata_work, cluster_key)
    adata_work.obs[group_key] = _normalize_harmony_series(adata_work.obs[group_key])
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)
    sc.pp.highly_variable_genes(adata_work, n_top_genes=n_top_genes)
    pcs = min(n_pcs, adata_work.n_obs - 1, adata_work.n_vars - 1)
    if pcs < 2:
        raise ValueError(f"Not enough cells/genes for PCA: shape={adata_work.shape}")
    sc.pp.pca(adata_work, n_comps=pcs, svd_solver="arpack", use_highly_variable=True)

    pca_embedding = np.asarray(adata_work.obsm["X_pca"])
    harmony_meta = adata_work.obs[[group_key]].copy()
    harmony_output = hm.run_harmony(
        pca_embedding,
        harmony_meta,
        group_key,
        max_iter_harmony=max_iter_harmony,
    )
    corrected = np.asarray(harmony_output.Z_corr).T

    if corrected.shape != pca_embedding.shape:
        if corrected.T.shape == pca_embedding.shape:
            corrected = corrected.T
        else:
            raise ValueError(
                f"Unexpected Harmony output shape {corrected.shape}; "
                f"expected {pca_embedding.shape}"
            )

    adata_work.obsm[embedding_key] = corrected
    sc.pp.neighbors(
        adata_work,
        n_neighbors=min(n_neighbors, max(2, adata_work.n_obs - 1)),
        use_rep=embedding_key,
    )
    sc.tl.leiden(adata_work, resolution=leiden_resolution, key_added=cluster_key)
    return adata_work


def build_harmony_arms_evaluator(
    batch_key: str,
    reference_cluster_key: str,
    top_marker_num: int = 50,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 50,
    leiden_resolution: float = 1.0,
    max_iter_harmony: int = 20,
):
    def evaluator(candidate_results: Dict[str, Dict[str, Any]]):
        score_rows = []
        payload = {"scores": []}

        for mode, result in candidate_results.items():
            adata_mode = result["adata"].copy()
            group_key = result["key_added"]
            cluster_key = f"leiden_{mode}"
            embedding_key = f"X_pca_harmony_{mode}"

            if reference_cluster_key not in adata_mode.obs.columns:
                raise KeyError(f"reference_cluster_key not found: {reference_cluster_key}")
            adata_mode.obs[reference_cluster_key] = adata_mode.obs[reference_cluster_key].astype("category")

            harmonized = _run_harmony_and_cluster(
                adata=adata_mode,
                group_key=group_key,
                batch_key=batch_key,
                n_top_genes=n_top_genes,
                n_pcs=n_pcs,
                n_neighbors=n_neighbors,
                leiden_resolution=leiden_resolution,
                max_iter_harmony=max_iter_harmony,
                cluster_key=cluster_key,
                embedding_key=embedding_key,
            )

            _, ref_arms, mode_arms = run_ARMS_analysis(
                adata_ref=harmonized,
                method1_cluster_key=reference_cluster_key,
                method2_cluster_key=cluster_key,
                top_marker_num=top_marker_num,
            )

            row = {
                "mode": mode,
                "reference_ARMS": ref_arms,
                "mode_ARMS": mode_arms,
                "arms_gain": mode_arms - ref_arms,
            }
            score_rows.append(row)
            payload["scores"].append(row)

        ranking = sorted(score_rows, key=lambda item: (item["mode_ARMS"], item["arms_gain"]), reverse=True)
        best_mode = ranking[0]["mode"]
        payload["ranking"] = ranking
        payload["best_mode"] = best_mode
        return payload

    return evaluator
