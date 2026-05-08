"""
Heatmap visualizations for homology experiments.

This module creates heatmaps showing homology metrics as a function of epsilon and noise.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import zero_overlap_probability


def plot_homology_noise_epsilon_heatmaps(
    results,
    scenario_num,
    num_parents_list=[3, 6, 12, 32, 48],
    figsize=(20, 15),
    output_file=None,
    use_median=False
):
    """
    Create heatmap plots showing homology metrics vs epsilon and noise.

    Creates a 2x5 grid where:
    - Top row: w1_h0 heatmaps for different num_parents
    - Bottom row: ltmax_h1 heatmaps for different num_parents

    Args:
        results: DataFrame with homology results (pre-loaded for all num_parents)
        scenario_num: Scenario number (1, 2, 3, or 4)
        num_parents_list: List of num_parents values to plot
        figsize: Figure size
        output_file: Optional output filename (if None, no save)
        use_median: If True, use median aggregation; if False, use mean

    Returns:
        Matplotlib figure object
    """
    conditions = {
        "h0": {"metric": "w1_h0", "vlim": [0.1, 0.8]},
        "h1": {"metric": "ltmax_h1", "vlim": [0.0, 0.2]},
    }

    fig, axes = plt.subplots(2, 5, figsize=figsize, sharey=True, constrained_layout=True)
    axes = axes.flatten()

    agg_func = 'median' if use_median else 'mean'

    for i, num_parents in enumerate(num_parents_list):
        # Filter results for this num_parents
        res_filtered = results[results['n_parent_features'] == num_parents]

        if len(res_filtered) == 0:
            print(f"Warning: No data for num_parents={num_parents}")
            continue

        # Aggregate by depsilon and noise_scale
        df = (
            res_filtered[["depsilon", "noise_scale", "w1_h0", "ltmax_h1"]]
            .groupby(by=["depsilon", "noise_scale"])
            .agg(agg_func)
            .reset_index()
        )

        # Top row: H0 heatmap
        vmin, vmax = conditions["h0"]["vlim"]
        data = df.pivot(index="depsilon", columns="noise_scale", values=conditions["h0"]["metric"])
        sns.heatmap(data, ax=axes[i], cbar=False, vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cmap="viridis")
        axes[i].set_ylabel("depsilon")
        axes[i].set_xlabel("noise_scale")

        # Calculate max_features from context (assuming it's in the data)
        max_features = res_filtered['max_features'].iloc[0] if 'max_features' in res_filtered.columns else 64

        axes[i].set_title(
            f"n_parent/n_max = {num_parents/max_features:.2f}\nMax overlap proba {1.0 - zero_overlap_probability(max_features, num_parents-1):.3f}"
        )

        # Bottom row: H1 heatmap
        vmin, vmax = conditions["h1"]["vlim"]
        data = df.pivot(index="depsilon", columns="noise_scale", values=conditions["h1"]["metric"])
        sns.heatmap(data, ax=axes[i + 5], cbar=False, vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cmap="viridis")
        axes[i + 5].set_ylabel("depsilon")
        axes[i + 5].set_xlabel("noise_scale")

    # Add colorbars
    cbar = fig.colorbar(axes[0].collections[0], ax=axes[0:5], location="bottom", label="W1 H0")
    cbar = fig.colorbar(axes[5].collections[0], ax=axes[5:], location="bottom", label="ltmax H1")

    agg_label = "Median" if use_median else "Mean"
    plt.suptitle(f"Scenario {scenario_num} - {agg_label} Homology Metrics")

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved: {output_file}")

    return fig


def plot_homology_noise_epsilon_heatmaps_all_scenarios(
    data_dict,
    num_parents_list=[3, 6, 12, 32, 48],
    figsize=(20, 15),
    output_dir=".",
    use_median=False
):
    """
    Create heatmap plots for all scenarios.

    Args:
        data_dict: Dictionary mapping scenario_num to DataFrame
        num_parents_list: List of num_parents values to plot
        figsize: Figure size
        output_dir: Directory to save output files
        use_median: If True, use median aggregation; if False, use mean

    Returns:
        Dictionary mapping scenario_num to figure object
    """
    figures = {}
    suffix = "median" if use_median else "mean"

    for scenario_num, results in data_dict.items():
        if results is None:
            print(f"Skipping scenario {scenario_num}: No data")
            continue

        output_file = f"{output_dir}/homology_noise_epsilon_scenario_{scenario_num}_{suffix}.png"
        fig = plot_homology_noise_epsilon_heatmaps(
            results=results,
            scenario_num=scenario_num,
            num_parents_list=num_parents_list,
            figsize=figsize,
            output_file=output_file,
            use_median=use_median
        )
        figures[scenario_num] = fig
        plt.close(fig)

    return figures
