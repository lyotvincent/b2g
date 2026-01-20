"""
Utility functions
"""

import numpy as np
import scipy.sparse
import scanpy as sc


def detect_raw_counts(adata):
    """
    Detect and return the location of raw count matrix

    Returns:
        tuple: (data_matrix, location_name)
        - data_matrix: raw count matrix
        - location_name: 'adata.X' or 'adata.raw.X'

    Raises:
        ValueError: if raw count matrix is not found
    """

    def is_raw_counts(X):
        """Determine if matrix is raw counts"""
        if X is None:
            return False

        if scipy.sparse.issparse(X):
            data = X.data
        else:
            data = X.flatten()

        # Get non-zero values
        non_zero = data[data > 0]

        if len(non_zero) == 0:
            return False

        # Randomly sample 10% of non-zero values
        sample_size = max(1, int(len(non_zero) * 0.1))
        sample = np.random.choice(non_zero, size=sample_size, replace=False)

        # Check if all are integers (allow floating-point error 1e-6)
        is_integer = np.abs(sample - np.round(sample)) < 1e-6
        integer_ratio = np.sum(is_integer) / len(sample)

        # If more than 95% are integers, consider it raw counts
        return integer_ratio > 0.95

    print("\nDetecting raw count matrix location...")

    # Check adata.X
    print("  Checking adata.X...")
    if is_raw_counts(adata.X):
        if scipy.sparse.issparse(adata.X):
            max_val = adata.X.data.max()
        else:
            max_val = adata.X.max()
        print(f"  ✓ adata.X is raw count matrix (max value: {max_val:.0f})")
        return adata.X, 'adata.X'
    else:
        if scipy.sparse.issparse(adata.X):
            max_val = adata.X.data.max()
        else:
            max_val = adata.X.max()
        print(f"  ✗ adata.X is not raw count matrix (max value: {max_val:.2f})")

    # Check adata.raw.X
    if adata.raw is not None:
        print("  Checking adata.raw.X...")
        if is_raw_counts(adata.raw.X):
            if scipy.sparse.issparse(adata.raw.X):
                max_val = adata.raw.X.data.max()
            else:
                max_val = adata.raw.X.max()
            print(f"  ✓ adata.raw.X is raw count matrix (max value: {max_val:.0f})")
            return adata.raw.X, 'adata.raw.X'
        else:
            if scipy.sparse.issparse(adata.raw.X):
                max_val = adata.raw.X.data.max()
            else:
                max_val = adata.raw.X.max()
            print(f"  ✗ adata.raw.X is not raw count matrix (max value: {max_val:.2f})")
    else:
        print("  ✗ adata.raw does not exist")

    # If neither is raw counts, raise error
    raise ValueError(
        "\nError: Cannot find raw count matrix!\n"
        "Metacell method requires raw UMI count data (integer values, like 1, 2, 5, etc.)\n"
        "Please ensure:\n"
        "  1. adata.X or adata.raw.X contains raw count matrix\n"
        "  2. Data has not been normalized, log-transformed, or standardized\n"
        "  3. If using Leiden clustering method, this check will be skipped"
    )


def compute_weighted_distance_matrix(Z, alpha=2.0):
    """
    Weighted Euclidean distance calculation based on power function weights
    """
    from sklearn.preprocessing import MinMaxScaler
    from scipy.spatial.distance import squareform

    # Calculate weights and normalize
    W = np.power(np.abs(Z), alpha)
    W_prime = MinMaxScaler().fit_transform(W)

    # Calculate weighted distance matrix
    n = Z.shape[0]
    distance_matrix_square = np.zeros((n, n))

    for p in range(n):
        for q in range(p + 1, n):
            final_weights = np.maximum(W_prime[p], W_prime[q])
            weighted_sum = np.sum(final_weights * (Z[p] - Z[q]) ** 2)
            distance = np.sqrt(weighted_sum)
            distance_matrix_square[p, q] = distance_matrix_square[q, p] = distance

    return distance_matrix_square, squareform(distance_matrix_square)


def dynamic_tree_cut_python(Z, distance_matrix_square, min_cluster_size=2, deep_split=2):
    """
    Dynamic tree cutting using Python's dynamicTreeCut package
    """
    from dynamicTreeCut import cutreeHybrid

    print(f"\nUsing Python dynamicTreeCut for dynamic tree cutting...")
    print(f"  Parameters: minClusterSize={min_cluster_size}, deepSplit={deep_split}")

    try:
        result = cutreeHybrid(
            Z,
            distance_matrix_square,
            minClusterSize=min_cluster_size,
            deepSplit=deep_split,
            pamStage=True,
            pamRespectsDendro=True
        )

        # Get cluster labels
        cluster_labels = np.array(result['labels']) if isinstance(result, dict) else np.array(result)

        n_clusters = len(np.unique(cluster_labels[cluster_labels > 0]))
        n_unassigned = np.sum(cluster_labels == 0)

        print(f"| Python dynamicTreeCut complete!")
        print(f"  Number of identified clusters: {n_clusters}")
        print(f"  Number of unassigned samples: {n_unassigned}")

        return cluster_labels, Z

    except Exception as e:
        print(f"!!! Dynamic tree cutting failed: {e}")
        raise