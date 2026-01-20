"""
B2G: Batch-to-Group - Adaptive Batch Grouping for Single-Cell RNA-seq Data

A tool for intelligent batch grouping using metacells/leiden clustering and 
PERMANOVA-based prior selection.
"""

__version__ = "0.1.0"

from .config import AnalysisConfig
from .grouping import group_batches

__all__ = [
    'AnalysisConfig',
    'group_batches',
    '__version__',
]


def group(
    adata,
    batch_key='batch',
    method='metacell',
    additional_features=None,
    target_metacell_size=48,
    leiden_resolution=1.0,
    min_priors=1,
    max_priors=None,
    gap_threshold=0.1,
    output_dir=None,
    fig_path=None,
    met_path=None,
    key_added=None,
    copy=False,
    **kwargs
):
    """
    Intelligent batch grouping

    This is the main interface function of B2G, which can be used directly in scanpy workflow.

    Parameters
    ----------
    adata : AnnData
        AnnData object to be grouped. Must contain raw count matrix.
    batch_key : str, default='batch'
        Column name in .obs representing batches
    method : str, default='metacell'
        Clustering method, options: 'metacell' or 'leiden'
    additional_features : list of dict, optional
        Additional prior features, format: [{'column': 'feature_name', 'description': 'description'}, ...]
    target_metacell_size : int, default=48
        Target metacell size (only used when method='metacell')
    leiden_resolution : float, default=1.0
        Leiden clustering resolution (only used when method='leiden')
    min_priors : int, default=1
        Minimum number of priors to select
    max_priors : int, optional
        Maximum number of priors to select
    gap_threshold : float, default=0.1
        Minimum score threshold for prior filtering
    output_dir : str, optional
        Output directory (for saving processed data)
    fig_path : str, optional
        Directory for saving figures
    met_path : str, optional
        Directory for saving metrics
    key_added : str, optional
        Column name to add to .obs for grouping, default is 'groups_{method}_adaptive'
    copy : bool, default=False
        Whether to return a copy of the data
    **kwargs
        Other parameters passed to configuration

    Returns
    -------
    adata : AnnData
        AnnData object with grouping information added (in .obs[key_added])

    Examples
    --------
    Basic usage (using batch information only):
    >>> import scanpy as sc
    >>> import b2g
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> b2g.group(adata, batch_key='batch')

    Using prior features:
    >>> b2g.group(
    ...     adata,
    ...     batch_key='donor_id',
    ...     additional_features=[
    ...         {'column': 'disease', 'description': 'Disease status'},
    ...         {'column': 'sex', 'description': 'Sex'}
    ...     ]
    ... )

    Using Leiden clustering:
    >>> b2g.group(
    ...     adata,
    ...     batch_key='batch',
    ...     method='leiden',
    ...     leiden_resolution=1.5
    ... )
    """

    if copy:
        adata = adata.copy()

    # Create configuration object
    config = AnalysisConfig()

    # Set basic parameters
    config.column_mapping = {'batch': batch_key}
    config.clustering_method = method

    # Set additional features
    if additional_features is not None:
        config.additional_features = additional_features
    else:
        config.additional_features = []

    # Set prior selection parameters
    config.b2g_min_priors = min_priors
    config.b2g_max_priors = max_priors
    config.b2g_gap_threshold = gap_threshold

    # Set clustering parameters
    if method == 'metacell':
        config.b2g_metacell_params['target_metacell_size'] = target_metacell_size
    elif method == 'leiden':
        config.leiden_params['resolution'] = leiden_resolution

    # Set output paths
    if output_dir is not None:
        config.output_dir = output_dir
    if fig_path is not None:
        config.fig_path = fig_path
    if met_path is not None:
        config.met_path = met_path

    # Set grouping column name
    if key_added is None:
        key_added = f'groups_{method}_adaptive'

    # Apply other parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Execute grouping
    adata = group_batches(adata, config, key_added=key_added)

    return adata