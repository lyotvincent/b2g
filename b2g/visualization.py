"""
Visualization functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# matplotlib global settings
plt.ioff()
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12


def visualize_prior_selection(evaluation_results, selected_priors,
                               config, threshold=None, top_score=None):
    """
    Visualize prior selection process
    """
    os.makedirs(config.fig_path, exist_ok=True)

    # Prepare data
    sorted_df = evaluation_results.sort_values('final_score', ascending=False)
    priors = sorted_df['prior'].tolist()
    confounding_scores = sorted_df['confounding_score'].tolist()
    final_scores = sorted_df['final_score'].tolist()
    colors = ['#2E7D32' if p in selected_priors else '#BDBDBD' for p in priors]

    # Create figure: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(priors) * 0.4)))
    y_pos = range(len(priors))

    # ===== Left plot: Confounding detection scores =====
    bars1 = ax1.barh(y_pos, confounding_scores, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(priors, fontsize=10)
    ax1.set_xlabel('Confounding Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    ax1.set_title('Confounding Detection', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

    # Add score labels
    for bar, score in zip(bars1, confounding_scores):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', ha='left', va='center', fontsize=9)

    # Add confounding severity reference lines
    ax1.axvline(x=0.8, color='red', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Severe (0.8)')
    ax1.axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Moderate (0.5)')
    ax1.axvline(x=0.3, color='yellow', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Mild (0.3)')
    ax1.legend(loc='lower right', fontsize=9)

    # ===== Right plot: Final scores =====
    bars2 = ax2.barh(y_pos, final_scores, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([''] * len(priors))  # Don't show y-axis labels (already in left plot)
    ax2.set_xlabel('Final Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(final_scores) * 1.15 if max(final_scores) > 0 else 1)
    ax2.set_title('Final Scores', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

    # Add score labels
    for bar, score in zip(bars2, final_scores):
        ax2.text(bar.get_width() + max(final_scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', ha='left', va='center', fontsize=9)

    # Add threshold line
    if threshold is not None and top_score is not None:
        threshold_line = top_score - threshold
        ax2.axvline(x=threshold_line, color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label=f'Threshold: {threshold_line:.3f}')
        ax2.axvline(x=top_score, color='blue', linestyle='--', linewidth=2,
                    alpha=0.5, label=f'Top: {top_score:.3f}')
        ax2.legend(loc='lower right', fontsize=9)

    # Add main title and legend
    plt.suptitle('Prior Selection Results', fontsize=14, fontweight='bold', y=0.98)

    # Create overall legend (at bottom of figure)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', alpha=0.8, label=f'Selected ({len(selected_priors)})'),
        Patch(facecolor='#BDBDBD', alpha=0.8, label=f'Rejected ({len(priors) - len(selected_priors)})')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    output_file = os.path.join(config.fig_path, 'prior_selection_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization results saved to: {output_file}")
    plt.close()


def plot_alpha_evaluation(results_df, fig_path):
    """
    Visualize alpha evaluation results
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Alpha Parameter Evaluation (Calinski-Harabasz Method)',
                 fontsize=16, fontweight='bold')

    alphas = results_df['alpha'].values
    best_alpha = results_df.iloc[results_df['final_score'].argmax()]['alpha']

    # Final score
    ax = axes[0]
    ax.plot(alphas, results_df['final_score'], 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
    ax.axvline(best_alpha, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Best: α={best_alpha}')
    ax.scatter(best_alpha, results_df.loc[results_df['alpha'] == best_alpha, 'final_score'].values[0],
               s=400, c='red', marker='*', zorder=5, edgecolors='black', linewidths=2)
    ax.set_xlabel('Alpha', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final Score', fontsize=13, fontweight='bold')
    ax.set_title('(A) Final Score vs Alpha', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Distance change
    ax = axes[1]
    ax.plot(alphas, results_df['distance_change'], 'o-', linewidth=2.5, markersize=10, color='#F18F01')
    ax.axvline(best_alpha, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Alpha', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distance Matrix Change', fontsize=13, fontweight='bold')
    ax.set_title('(B) Distance Change vs Alpha', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{fig_path}/alpha_optimization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n| Alpha evaluation chart saved")


def plot_dendrogram(Z, valid_batch_names, cluster_labels, config, method_name,
                    use_prior_knowledge, selected_priors=None):
    """
    Plot colored dendrogram
    """
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib import cm

    plt.figure(figsize=(12, 5))

    # Create color mapping
    unique_labels = sorted(np.unique(cluster_labels))
    colors_map = {}
    cmap = cm.get_cmap('tab10')
    for i, label in enumerate(unique_labels):
        color = cmap(i % 10)
        colors_map[label] = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    # First plot temporary dendrogram to get leaf order
    temp_dend = dendrogram(Z, no_plot=True, labels=valid_batch_names)
    leaf_order = temp_dend['leaves']

    # Assign colors to each leaf node
    leaf_colors = {}
    for i, leaf_idx in enumerate(leaf_order):
        cluster_label = cluster_labels[leaf_idx]
        leaf_colors[i] = colors_map[cluster_label]

    # Define link color function
    def link_color_func(k):
        n_samples = len(valid_batch_names)
        if k < n_samples:
            return leaf_colors.get(k, 'grey')
        else:
            return 'grey'

    # Plot colored dendrogram
    gr = dendrogram(Z,
                    labels=valid_batch_names,
                    leaf_font_size=10,
                    link_color_func=link_color_func,
                    above_threshold_color='grey')

    # Color x-axis labels
    ax = plt.gca()
    xlabels = ax.get_xmajorticklabels()
    for i, label in enumerate(xlabels):
        batch_idx = leaf_order[i]
        cluster_label = cluster_labels[batch_idx]
        label.set_color(colors_map[cluster_label])
        label.set_weight('bold')

    # Build title
    n_clusters = len(np.unique(cluster_labels))
    title_suffix = f'(Dynamic Tree Cut: {n_clusters} groups)'

    if use_prior_knowledge:
        selected_priors_str = ' + '.join(selected_priors)
        title_text = f'PERMANOVA-Selected [{selected_priors_str}] + {method_name} {title_suffix}'
    else:
        title_text = f'Global {method_name} Grouping {title_suffix}'

    plt.title(title_text, fontsize=14)
    plt.xlabel('Batch/Sample', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{config.fig_path}/dendrogram.png",
                bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\n| Dendrogram saved to {config.fig_path}/dendrogram.png")