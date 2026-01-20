"""
Prior feature selection related functions
"""

import numpy as np
import pandas as pd
import gc
from scipy.stats import chi2_contingency
from sklearn.metrics import euclidean_distances
import scipy.sparse


try:
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False


def detect_prior_confounding(adata, prior_column, batch_column):
    """
    Check if prior spans multiple batches
    """
    # Count mapping relationships
    prior_to_batches = {}
    for prior_val in adata.obs[prior_column].unique():
        mask = adata.obs[prior_column] == prior_val
        prior_to_batches[prior_val] = len(adata.obs.loc[mask, batch_column].unique())

    n_priors = len(prior_to_batches)
    prior_one_to_one_ratio = sum(1 for n in prior_to_batches.values() if n == 1) / n_priors
    confounding_penalty = np.exp(-prior_one_to_one_ratio)

    # Classification
    if prior_one_to_one_ratio >= 0.8:
        confounding_type, warning_flag = "Severe Confounding", True
    elif prior_one_to_one_ratio >= 0.5:
        confounding_type, warning_flag = "Moderate Confounding", True
    elif prior_one_to_one_ratio >= 0.3:
        confounding_type, warning_flag = "Mild Confounding", False
    else:
        confounding_type, warning_flag = "Independent", False

    print(f"    Prior→Batch mapping:")
    for prior, n_batches in sorted(prior_to_batches.items()):
        print(f"      {'!!!' if n_batches == 1 else '|'} {prior}: {n_batches} batches")
    print(f"    One-to-one ratio: {prior_one_to_one_ratio:.0%}")
    print(f"    Confounding type: {confounding_type}")
    print(f"    Penalty coefficient: {confounding_penalty:.2f}")

    return {
        'confounding_type': confounding_type,
        'confounding_score': prior_one_to_one_ratio,
        'confounding_penalty': confounding_penalty,
        'warning_flag': warning_flag
    }


def detect_prior_collinearity(adata, prior_columns, threshold=0.95):
    """
    Detect collinearity among priors
    """
    print(f"\nDetecting prior collinearity (threshold={threshold}):")

    n_priors = len(prior_columns)
    association_matrix = np.zeros((n_priors, n_priors))

    # Calculate Cramér's V coefficient
    for i, prior1 in enumerate(prior_columns):
        for j in range(i + 1, n_priors):
            prior2 = prior_columns[j]
            contingency_table = pd.crosstab(adata.obs[prior1], adata.obs[prior2])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1

            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
            association_matrix[i, j] = association_matrix[j, i] = cramers_v

    # Print association matrix
    print(f"\n  Association matrix (Cramér's V):")
    for i, row_name in enumerate(prior_columns):
        print(f"    {row_name:>15}", end=" ")
        for j in range(n_priors):
            if i == j:
                print(f"{'1.000':>15}", end=" ")
            else:
                val = association_matrix[i, j]
                print(f"{'**' + f'{val:.3f}' + '**' if val >= threshold else f'{val:.3f}':>15}", end=" ")
        print()

    # Use union-find to identify collinear groups
    parent = list(range(n_priors))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n_priors):
        for j in range(i + 1, n_priors):
            if association_matrix[i, j] >= threshold:
                union(i, j)

    # Group
    groups = {}
    for i in range(n_priors):
        root = find(i)
        groups.setdefault(root, []).append(i)

    collinear_groups = [[prior_columns[i] for i in indices] for indices in groups.values() if len(indices) > 1]
    independent_priors = [prior_columns[i] for indices in groups.values() if len(indices) == 1 for i in indices]

    print(f"\nDetection results:")
    print(f"  Number of collinear groups: {len(collinear_groups)}")
    print(f"  Number of independent priors: {len(independent_priors)}")

    if collinear_groups:
        print(f"\n  Collinear groups:")
        for i, group in enumerate(collinear_groups, 1):
            print(f"    Group {i}: {', '.join(group)}")

    return collinear_groups, independent_priors


def select_representative_from_collinear_group(group, evaluation_results):
    """
    Select representative prior from collinear group
    """
    print(f"\n  Selecting representative from collinear group: {', '.join(group)}")

    scores = []
    for prior in group:
        result = evaluation_results[evaluation_results['prior'] == prior].iloc[0]
        selection_score = result['n_categories'] * (-np.e) + result['final_score']
        scores.append({
            'prior': prior,
            'n_categories': result['n_categories'],
            'independence': 1 - result['confounding_score'],
            'final_score': result['final_score'],
            'selection_score': selection_score
        })
        print(f"    {prior}: Categories {result['n_categories']}, "
              f"Independence {1 - result['confounding_score']:.3f}, "
              f"Score {result['final_score']:.3f}")

    selected_prior = max(scores, key=lambda x: x['selection_score'])['prior']
    print(f"    → Selected: {selected_prior}")

    return selected_prior


def evaluate_prior_with_permanova(adata, prior_column, use_embedding='X_pca',
                                   n_permutations=999, downsample_threshold=50000,
                                   downsample_size=50000, random_seed=None):
    """
    Evaluate prior importance using PERMANOVA (with downsampling support)
    """
    if not SKBIO_AVAILABLE:
        return {'F_statistic': 0.0, 'p_value': 1.0, 'df_between': 0,
                'df_within': 0, 'method': 'fallback', 'n_cells_used': 0}

    # Get data
    if use_embedding in adata.obsm:
        X_original = adata.obsm[use_embedding]
    else:
        X_original = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

    n_cells = X_original.shape[0]

    # Downsampling
    if n_cells > downsample_threshold:
        print(f"    Downsampling: {n_cells:,} → {downsample_size:,} cells")
        if random_seed is not None:
            np.random.seed(random_seed)
        sample_indices = np.sort(np.random.choice(n_cells, downsample_size, replace=False))
        X_sampled = X_original[sample_indices]
        grouping_sampled = adata.obs[prior_column].values[sample_indices]
    else:
        X_sampled = X_original
        grouping_sampled = adata.obs[prior_column].values

    # Calculate distance matrix
    print(f"    Computing distance matrix... ({X_sampled.shape[0]:,} × {X_sampled.shape[0]:,})")
    X_compute = X_sampled.astype(np.float64) if X_sampled.dtype != np.float64 else X_sampled
    distance_matrix = euclidean_distances(X_compute)

    # Symmetrize
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    del X_compute, X_sampled
    gc.collect()

    # Create DistanceMatrix and run PERMANOVA
    try:
        dm = DistanceMatrix(distance_matrix, ids=[f"cell_{i}" for i in range(len(distance_matrix))])
        print(f"    Running PERMANOVA ({n_permutations} permutations)...")

        perm_result = permanova(dm, grouping_sampled, permutations=n_permutations)
        F_statistic = perm_result['test statistic']
        p_value = perm_result['p-value']

        n_samples = len(grouping_sampled)
        n_groups = len(np.unique(grouping_sampled))
        df_between = n_groups - 1
        df_within = n_samples - n_groups

        print(f"    F={F_statistic:.4f}, P={p_value:.4e}")

        return {
            'F_statistic': F_statistic,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'method': 'permanova',
            'n_cells_used': n_samples
        }
    except Exception as e:
        print(f"    !!! PERMANOVA failed: {e}")
        return {'F_statistic': 0.0, 'p_value': 1.0, 'df_between': 0,
                'df_within': 0, 'method': 'fallback', 'n_cells_used': len(grouping_sampled)}


def comprehensive_prior_evaluation(adata, config, prior_column, batch_column,
                                    use_embedding='X_pca', n_permutations=999):
    """
    Prior importance evaluation
    """
    print(f"\n  Evaluating prior: {prior_column}")
    print("  " + "-" * 66)

    n_categories = len(adata.obs[prior_column].unique())
    print(f"    Number of categories: {n_categories}")

    # Multiple downsampling evaluations (if needed)
    n_cells = adata.n_obs
    if n_cells > config.b2g_downsample_threshold:
        print(f"    Performing {config.b2g_n_repeats} downsampling evaluations")
        F_stats, p_vals, df_bs, df_ws = [], [], [], []

        for i in range(config.b2g_n_repeats):
            print(f"    --- Round {i + 1}/{config.b2g_n_repeats} ---")
            result = evaluate_prior_with_permanova(
                adata, prior_column, use_embedding, n_permutations,
                config.b2g_downsample_threshold, config.b2g_downsample_size,
                config.b2g_random_seed_base + i
            )
            F_stats.append(result['F_statistic'])
            p_vals.append(result['p_value'])
            df_bs.append(result.get('df_between', 0))
            df_ws.append(result.get('df_within', 0))

        F_statistic = np.mean(F_stats)
        p_permanova = np.mean(p_vals)
        df_between = int(np.mean(df_bs))
        df_within = int(np.mean(df_ws))

        print(f"    Aggregated results: F={F_statistic:.4f}±{np.std(F_stats):.4f}, P={p_permanova:.4e}±{np.std(p_vals):.4e}")
    else:
        result = evaluate_prior_with_permanova(
            adata, prior_column, use_embedding, n_permutations,
            config.b2g_downsample_threshold, config.b2g_downsample_size, None
        )
        F_statistic = result['F_statistic']
        p_permanova = result['p_value']
        df_between = result.get('df_between', 0)
        df_within = result.get('df_within', 0)

    # Calculate R² and normalized score
    if F_statistic > 0 and df_within > 0:
        ratio = F_statistic * df_between / df_within
        R2 = ratio / (1 + ratio)
    else:
        R2 = 0.0

    score_R2 = min(R2 / 0.10, 1.0)

    print(f"    R²={R2:.6f} ({R2 * 100:.2f}%), Normalized score={score_R2:.4f}")

    # Confounding detection
    print(f"\n    【Confounding Detection】")
    confounding_result = detect_prior_confounding(adata, prior_column, batch_column)

    # Final scoring
    raw_score = score_R2
    final_score = raw_score * confounding_result['confounding_penalty']

    print(f"\n    Final score: {final_score:.4f} (R² score × Confounding penalty)")

    # Rating
    if final_score >= 0.7:
        rating = "***** Extremely Important"
    elif final_score >= 0.5:
        rating = "**** Very Important"
    elif final_score >= 0.3:
        rating = "*** Moderately Important"
    elif final_score >= 0.15:
        rating = "** Somewhat Important"
    else:
        rating = "* Not Important"

    print(f"    Rating: {rating}")
    print("  " + "-" * 66)

    return {
        'prior': prior_column,
        'n_categories': n_categories,
        'F_statistic': F_statistic,
        'R2': R2,
        'p_permanova': p_permanova,
        'score_R2': score_R2,
        'confounding_score': confounding_result['confounding_score'],
        'confounding_type': confounding_result['confounding_type'],
        'confounding_penalty': confounding_result['confounding_penalty'],
        'raw_score': raw_score,
        'final_score': final_score,
        'rating': rating,
        'warning_flag': confounding_result['warning_flag']
    }


def prior_selection(evaluation_results, config):
    """
    Relative gap filtering method
    """
    print("\n" + "=" * 70)
    print("【Relative Gap Filtering Method】")
    print("=" * 70)

    sorted_df = evaluation_results.sort_values('final_score', ascending=False)

    print(f"\nSelection criteria:")
    print(f"  - Minimum score: {config.b2g_gap_threshold:.2f}")
    print(f"  - Minimum priors: {config.b2g_min_priors}")
    if config.b2g_max_priors:
        print(f"  - Maximum priors: {config.b2g_max_priors}")

    # Filter candidates meeting basic conditions
    candidates = sorted_df[sorted_df['final_score'] >= config.b2g_gap_threshold].to_dict('records')

    if len(candidates) == 0:
        print("\n!!! No candidates meet criteria, forcing to select highest score")
        candidates = [sorted_df.iloc[0].to_dict()]

    print(f"\nNumber of candidates meeting basic conditions: {len(candidates)}")

    # Relative gap filtering
    selected_priors = []
    selection_log = []
    top_score = candidates[0]['final_score']
    average_score = np.mean([c['final_score'] for c in candidates])
    threshold = 0.5 * average_score

    print(f"Highest score: {top_score:.4f}")
    print(f"Average score: {average_score:.4f}")
    print(f"Gap threshold: {threshold:.4f}")

    for i, candidate in enumerate(candidates):
        prior = candidate['prior']
        score = candidate['final_score']
        absolute_gap = top_score - score

        # Decision logic
        if i == 0:
            decision = 'SELECTED'
            reasons = [f"Highest score: {score:.4f}", "Automatically selected as baseline"]
        elif absolute_gap < threshold:
            if config.b2g_max_priors is None or len(selected_priors) < config.b2g_max_priors:
                decision = 'SELECTED'
                reasons = [f"Score: {score:.4f}",
                           f"Gap: {absolute_gap:.4f} < {threshold:.4f}"]
            else:
                decision = 'REJECTED'
                reasons = [f"Reached maximum limit ({config.b2g_max_priors})"]
        else:
            decision = 'REJECTED'
            reasons = [f"Score: {score:.4f}",
                       f"Gap: {absolute_gap:.4f} >= {threshold:.4f}"]

        # Record log
        selection_log.append({
            'prior': prior,
            'final_score': score,
            'absolute_gap': absolute_gap if i > 0 else 0.0,
            'rating': candidate.get('rating', ''),
            'p_value': candidate.get('p_permanova', 0),
            'decision': decision,
            'reasons': '; '.join(reasons)
        })

        # Selection handling
        if decision == 'SELECTED':
            selected_priors.append(prior)
            print(f"  {i + 1}. {prior}: Score={score:.4f}, Gap={absolute_gap:.4f} → Selected")
        else:
            print(f"  {i + 1}. {prior}: Score={score:.4f}, Gap={absolute_gap:.4f} → Rejected")
            if config.b2g_max_priors and len(selected_priors) >= config.b2g_max_priors:
                break

    # Minimum quantity guarantee
    if len(selected_priors) < config.b2g_min_priors:
        print(f"\nSupplementing to minimum quantity {config.b2g_min_priors}")
        for c in candidates[len(selected_priors):config.b2g_min_priors]:
            selected_priors.append(c['prior'])
            print(f"  → Forced addition: {c['prior']}")
            # Update corresponding log
            for log in selection_log:
                if log['prior'] == c['prior']:
                    log['decision'] = 'SELECTED (forced)'
                    log['reasons'] += '; Forced addition to meet minimum requirement'

    print(f"\nFinal selection: {len(selected_priors)} priors")
    for i, prior in enumerate(selected_priors, 1):
        print(f"  {i}. {prior}")

    return selected_priors, selection_log