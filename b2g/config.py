"""
Configuration class definition
"""

import os


class AnalysisConfig:
    def __init__(self):
        # 1. Input/output configuration
        self.input_file = None
        self.output_dir = "corrected_data"
        self.output_filename = "output.h5ad"
        self.fig_path = "figures"
        self.met_path = "metrics"

        # 2. Dataset column name mapping
        self.column_mapping = {'batch': 'batch'}
        self.cell_annotation_column = None
        self.selected_cell_type = None

        # 3. Additional features configuration
        self.additional_features = []

        # Method parameters
        self.b2g_min_priors = 1
        self.b2g_max_priors = None
        self.b2g_gap_threshold = 0.1

        # Downsampling parameters
        self.b2g_downsample_threshold = 50000
        self.b2g_downsample_size = 10000
        self.b2g_n_repeats = 3
        self.b2g_random_seed_base = 42

        self.clustering_method = 'metacell'  # Options: 'metacell' or 'leiden'

        # Leiden clustering parameters
        self.leiden_params = {
            'resolution': 1.0,
            'n_neighbors': 15,
            'n_pcs': 50,
            'random_state': 42,
            'min_cluster_size': 10,
        }

        # Metacell construction parameters
        self.b2g_metacell_params = {
            'target_metacell_size': 48,
            'excluded_gene_names': ['XIST', 'TSIX'],
            'excluded_gene_patterns': ['MT-.*', 'RPL.*', 'RPS.*'],
            'lateral_gene_names': ['MKI67', 'TOP2A', 'CDK1'],
            'lateral_gene_patterns': ['HIST.*'],
            'random_seed': 123456
        }

        # Dynamic tree cutting parameters
        self.b2g_dynamic_tree_params = {
            'min_cluster_size': 2,
            'deep_split': 2,
            'unassigned_as_outlier_group': True
        }

    def validate(self):
        """Validate configuration validity"""
        print("\n" + "=" * 70)
        print("Configuration Validation")
        print("=" * 70)

        # Check clustering method
        if self.clustering_method not in ['metacell', 'leiden']:
            raise ValueError(f"!!! Invalid clustering method: {self.clustering_method}. Please choose 'metacell' or 'leiden'")

        print(f"| Clustering method: {self.clustering_method.upper()}")

        # Check additional features configuration
        if len(self.additional_features) == 0:
            print("| ! No additional features configured - will use global mode only")
        else:
            print(f"| Detected {len(self.additional_features)} additional features")

        # Check input file (if provided)
        if self.input_file and not os.path.exists(self.input_file):
            raise FileNotFoundError(f"!!! Input file does not exist: {self.input_file}")

        if self.input_file:
            print(f"| Input file exists: {self.input_file}")

        # Check cell type filtering configuration
        if self.selected_cell_type is not None:
            if not self.cell_annotation_column:
                raise ValueError("!!! selected_cell_type is specified but cell_annotation_column is not set")
            print(f"| Will filter cell type: '{self.selected_cell_type}' (from column '{self.cell_annotation_column}')")

        print("\n| Configuration validation passed")
        print("=" * 70)

    def print_config(self):
        """Print current configuration"""
        print("\n" + "=" * 70)
        print("Current Analysis Configuration")
        print("=" * 70)

        print(f"\n【Input/Output Configuration】")
        if self.input_file:
            print(f"  Input file: {self.input_file}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Figures directory: {self.fig_path}")
        print(f"  Metrics directory: {self.met_path}")

        if self.cell_annotation_column:
            print(f"\n【Cell Type Filtering】")
            print(f"  Annotation column: {self.cell_annotation_column}")
            print(f"  Selected type: {self.selected_cell_type or 'None (use all data)'}")

        print(f"\n【Clustering Method】")
        print(f"  Method: {self.clustering_method.upper()}")

        if self.clustering_method == 'leiden':
            print(f"\n【Leiden Parameters】")
            print(f"  Resolution: {self.leiden_params['resolution']}")
            print(f"  Number of neighbors: {self.leiden_params['n_neighbors']}")
            print(f"  Number of PCs: {self.leiden_params['n_pcs']}")
            print(f"  Minimum cluster size: {self.leiden_params['min_cluster_size']}")
            print(f"  Random seed: {self.leiden_params['random_state']}")

        print(f"\n【Prior Knowledge Features】(total {len(self.additional_features)} features)")
        for i, feat in enumerate(self.additional_features, 1):
            print(f"  Feature {i}: {feat['column']} ({feat['description']})")

        print(f"\n【Method - Adaptive Prior Selection】")
        print(f"  Minimum priors: {self.b2g_min_priors}")
        print(f"  Maximum priors: {self.b2g_max_priors if self.b2g_max_priors else 'Unlimited'}")
        print(f"  Minimum score threshold: {self.b2g_gap_threshold}")

        print("=" * 70 + "\n")