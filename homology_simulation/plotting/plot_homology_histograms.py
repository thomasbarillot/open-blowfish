"""
2x2 Grid Homology KDE Plots with Mean/Std Plots.

This module creates 2x2 grid visualizations:
- Top-left: H0 KDE at specific depsilon/noise
- Top-right: H1 KDE at specific depsilon/noise
- Bottom-left: H0 mean ± std vs depsilon
- Bottom-right: H1 mean ± std vs depsilon

Supports both regular scenario comparisons and projection comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})

def plot_homology_scales(
    results_s1,
    results_s2,
    depsilon_filter=0.4,
    noise_filter=0.01,
    figsize=(12, 4),
    fontsize=15.0,
    palette={"parent_to_children": "red", "children_to_parent": "blue", "parent_to_children_mixed": "green"},
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Configuration for colors
    # Prepare data for KDE plots (top row)
    # Filter by active dimensions to exclude edge cases
    s1_filtered = results_s1  # [results_s1.corpus_active_dims < results_s1.n_parent_features - 3].copy()
    s2_filtered = results_s2  # [results_s2.query_active_dims < results_s2.n_parent_features - 3].copy()

    s1_hist = s1_filtered[(s1_filtered.depsilon == depsilon_filter) & (s1_filtered.noise_scale == noise_filter)]
    s2_hist = s2_filtered[(s2_filtered.depsilon == depsilon_filter) & (s2_filtered.noise_scale == noise_filter)]

    # Get number of dimensions for linestyle variation
    # Use corpus_active_dims for s1, query_active_dims for s2

    # Prepare data for line plots (bottom row)
    s12_line = pd.concat([s1_filtered, s2_filtered]).reset_index(drop=True)
    s12_line = s12_line[s12_line.noise_scale == noise_filter]
    
    # BOTTOM LEFT: H0 mean vs depsilon
    sns.lineplot(
        data=s12_line,
        x="depsilon",
        y="w1_h0",
        style="scenario",
        color="k",
        ax=axes[0],
        linewidth=2.5,
        errorbar=None
        
    )
    axes[0].set_xlabel("ε", fontsize=fontsize)
    axes[0].set_ylabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    # axes[1, 0].set_title(f"H0 Mean vs ε (noise={noise_filter})", fontsize=fontsize)
    axes[0].tick_params(axis="x", labelsize=fontsize)
    axes[0].tick_params(axis="y", labelsize=fontsize)
    for spine in axes[0].spines.values():
        spine.set_linewidth(2.0)
    axes[0].grid(True, alpha=0.3)
    
    sns.lineplot(
        data=s12_line,
        x="depsilon",
        y="ltmax_h1",
        style="scenario",
        color="k",
        ax=axes[1],
        linewidth=2.5,
        errorbar=None
    )
    axes[1].set_xlabel("ε", fontsize=fontsize)
    axes[1].set_ylabel(r"""$LT_{max}(H_{1})$""", fontsize=fontsize)
    # axes[1, 0].set_title(f"H0 Mean vs ε (noise={noise_filter})", fontsize=fontsize)
    axes[1].tick_params(axis="x", labelsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize)
    for spine in axes[1].spines.values():
        spine.set_linewidth(2.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_homology_histogram_with_scale(
    results_s1,
    results_s2,
    depsilon_filter=0.4,
    noise_filter=0.01,
    figsize=(12, 6),
    fontsize=15.0,
    metric="w1_h0",
    palette={"parent_to_children": "red", "children_to_parent": "blue", "parent_to_children_mixed": "green"},
):
    """
    Create 2x2 grid plot showing KDE plots and mean trends for H0 and H1.

    Top row: KDE distributions at specific depsilon and noise_scale
    Bottom row: Mean ± std as a function of depsilon

    Args:
        results_s1: DataFrame with scenario 1 results (parent_to_children)
        results_s2: DataFrame with scenario 2 results (children_to_parent)
        depsilon_filter: Depsilon value for KDE plots (default: 0.4)
        noise_filter: Noise scale for KDE plots (default: 0.01)
        figsize: Figure size (default: (12, 10))
        fontsize: Font size for labels (default: 15.0)

    Returns:
        Matplotlib figure object
    """
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    scenario_1_name = list(set(results_s1.scenario))[0]
    scenario_2_name = list(set(results_s2.scenario))[0]
    # Configuration for colors
    # Prepare data for KDE plots (top row)
    # Filter by active dimensions to exclude edge cases
    s1_filtered = results_s1  # [results_s1.corpus_active_dims < results_s1.n_parent_features - 3].copy()
    s2_filtered = results_s2  # [results_s2.query_active_dims < results_s2.n_parent_features - 3].copy()

    s1_hist = s1_filtered[(s1_filtered.depsilon == depsilon_filter) & (s1_filtered.noise_scale == noise_filter)]
    s2_hist = s2_filtered[(s2_filtered.depsilon == depsilon_filter) & (s2_filtered.noise_scale == noise_filter)]

    # Get number of dimensions for linestyle variation
    # Use corpus_active_dims for s1, query_active_dims for s2
    s1_dims = sorted(s1_hist["embedding_dim"].unique()) if len(s1_hist) > 0 else []
    s2_dims = sorted(s2_hist["embedding_dim"].unique()) if len(s2_hist) > 0 else []

    linestyles = ["-", "--", "-.", ":"]

    # TOP LEFT: H0 KDE
    axes[0].set_xlim(0.1, 1.1)

    for idx, dim in enumerate(s1_dims):
        data = s1_hist[s1_hist["embedding_dim"] == dim][metric].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0],
                color=palette[scenario_1_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"P→C (dim={dim})",
            )

    for idx, dim in enumerate(s2_dims):
        data = s2_hist[s2_hist["embedding_dim"] == dim][metric].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0],
                color=palette[scenario_2_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"C→P (dim={dim})",
            )

    axes[0].set_xlabel(metric, fontsize=fontsize)
    axes[0].set_ylabel("Density", fontsize=fontsize)
    # axes[0, 0].set_title(f"H0 Distribution (ε={depsilon_filter}, noise={noise_filter})", fontsize=fontsize)
    axes[0].tick_params(axis="x", labelsize=fontsize)
    axes[0].tick_params(axis="y", labelsize=fontsize)
    axes[0].legend(fontsize=10)
    for spine in axes[0].spines.values():
        spine.set_linewidth(2.0)

    # Prepare data for line plots (bottom row)
    s12_line = pd.concat([s1_filtered, s2_filtered]).reset_index(drop=True)
    s12_line = s12_line[s12_line.noise_scale == noise_filter]
    # s12_line['scenario'] = s12_line.apply(
    #    lambda row: 'parent_to_children' if 'corpus_active_dims' in row and pd.notna(row.get('corpus_active_dims')) else 'children_to_parent',
    #    axis=1
    # )

    # BOTTOM LEFT: H0 mean vs depsilon
    sns.lineplot(
        data=s12_line,
        x="depsilon",
        y=metric,
        hue="scenario",
        style="embedding_dim",
        palette=palette,
        ax=axes[1],
        linewidth=2.5,
    )
    axes[1].set_xlabel("ε", fontsize=fontsize)
    axes[1].set_ylabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    # axes[1, 0].set_title(f"H0 Mean vs ε (noise={noise_filter})", fontsize=fontsize)
    axes[1].tick_params(axis="x", labelsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize)
    for spine in axes[1].spines.values():
        spine.set_linewidth(2.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_homology_histograms_2x2(
    results_s1,
    results_s2,
    depsilon_filter=0.4,
    noise_filter=0.01,
    figsize=(12, 10),
    fontsize=15.0,
    palette={"parent_to_children": "red", "children_to_parent": "blue", "parent_to_children_mixed": "green"},
):
    """
    Create 2x2 grid plot showing KDE plots and mean trends for H0 and H1.

    Top row: KDE distributions at specific depsilon and noise_scale
    Bottom row: Mean ± std as a function of depsilon

    Args:
        results_s1: DataFrame with scenario 1 results (parent_to_children)
        results_s2: DataFrame with scenario 2 results (children_to_parent)
        depsilon_filter: Depsilon value for KDE plots (default: 0.4)
        noise_filter: Noise scale for KDE plots (default: 0.01)
        figsize: Figure size (default: (12, 10))
        fontsize: Font size for labels (default: 15.0)

    Returns:
        Matplotlib figure object
    """
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    scenario_1_name = list(set(results_s1.scenario))[0]
    scenario_2_name = list(set(results_s2.scenario))[0]
    # Configuration for colors
    # Prepare data for KDE plots (top row)
    # Filter by active dimensions to exclude edge cases
    s1_filtered = results_s1  # [results_s1.corpus_active_dims < results_s1.n_parent_features - 3].copy()
    s2_filtered = results_s2  # [results_s2.query_active_dims < results_s2.n_parent_features - 3].copy()

    s1_hist = s1_filtered[(s1_filtered.depsilon == depsilon_filter) & (s1_filtered.noise_scale == noise_filter)]
    s2_hist = s2_filtered[(s2_filtered.depsilon == depsilon_filter) & (s2_filtered.noise_scale == noise_filter)]

    # Get number of dimensions for linestyle variation
    # Use corpus_active_dims for s1, query_active_dims for s2
    s1_dims = sorted(s1_hist["embedding_dim"].unique()) if len(s1_hist) > 0 else []
    s2_dims = sorted(s2_hist["embedding_dim"].unique()) if len(s2_hist) > 0 else []

    linestyles = ["-", "--", "-.", ":"]

    # TOP LEFT: H0 KDE
    axes[0, 0].set_xlim(0.1, 1.1)

    for idx, dim in enumerate(s1_dims):
        data = s1_hist[s1_hist["embedding_dim"] == dim]["w1_h0"].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0, 0],
                color=palette[scenario_1_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"P→C (dim={dim})",
            )

    for idx, dim in enumerate(s2_dims):
        data = s2_hist[s2_hist["embedding_dim"] == dim]["w1_h0"].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0, 0],
                color=palette[scenario_2_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"C→P (dim={dim})",
            )

    axes[0, 0].set_xlabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    axes[0, 0].set_ylabel("Density", fontsize=fontsize)
    # axes[0, 0].set_title(f"H0 Distribution (ε={depsilon_filter}, noise={noise_filter})", fontsize=fontsize)
    axes[0, 0].tick_params(axis="x", labelsize=fontsize)
    axes[0, 0].tick_params(axis="y", labelsize=fontsize)
    axes[0, 0].legend(fontsize=10)
    for spine in axes[0, 0].spines.values():
        spine.set_linewidth(2.0)

    # TOP RIGHT: H1 KDE
    # Plot S1 with varying linestyle by dimension
    for idx, dim in enumerate(s1_dims):
        data = s1_hist[s1_hist["embedding_dim"] == dim]["ltmax_h1"].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0, 1],
                color=palette[scenario_1_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"P→C (dim={dim})",
            )

    for idx, dim in enumerate(s2_dims):
        data = s2_hist[s2_hist["embedding_dim"] == dim]["ltmax_h1"].dropna()
        if len(data) > 1:
            sns.kdeplot(
                data=data,
                ax=axes[0, 1],
                color=palette[scenario_2_name],
                fill=True,
                alpha=0.125,
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5,
                label=f"C→(dim={dim})",
            )

    axes[0, 1].set_xlabel(r"""$LT_{max}(H_{1})$""", fontsize=fontsize)
    axes[0, 1].set_ylabel("Density", fontsize=fontsize)
    # axes[0, 1].set_title(f"H1 Distribution (ε={depsilon_filter}, noise={noise_filter})", fontsize=fontsize)
    axes[0, 1].tick_params(axis="x", labelsize=fontsize)
    axes[0, 1].tick_params(axis="y", labelsize=fontsize)
    axes[0, 1].legend(fontsize=10)
    for spine in axes[0, 1].spines.values():
        spine.set_linewidth(2.0)

    # Prepare data for line plots (bottom row)
    s12_line = pd.concat([s1_filtered, s2_filtered]).reset_index(drop=True)
    s12_line = s12_line[s12_line.noise_scale == noise_filter]
    # s12_line['scenario'] = s12_line.apply(
    #    lambda row: 'parent_to_children' if 'corpus_active_dims' in row and pd.notna(row.get('corpus_active_dims')) else 'children_to_parent',
    #    axis=1
    # )

    # BOTTOM LEFT: H0 mean vs depsilon
    sns.lineplot(
        data=s12_line,
        x="depsilon",
        y="w1_h0",
        hue="scenario",
        style="embedding_dim",
        palette=palette,
        ax=axes[1, 0],
        linewidth=2.5,
    )
    axes[1, 0].set_xlabel("ε", fontsize=fontsize)
    axes[1, 0].set_ylabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    # axes[1, 0].set_title(f"H0 Mean vs ε (noise={noise_filter})", fontsize=fontsize)
    axes[1, 0].tick_params(axis="x", labelsize=fontsize)
    axes[1, 0].tick_params(axis="y", labelsize=fontsize)
    for spine in axes[1, 0].spines.values():
        spine.set_linewidth(2.0)
    axes[1, 0].grid(True, alpha=0.3)

    # BOTTOM RIGHT: H1 mean vs depsilon
    sns.lineplot(
        data=s12_line,
        x="depsilon",
        y="ltmax_h1",
        hue="scenario",
        style="embedding_dim",
        palette=palette,
        ax=axes[1, 1],
        linewidth=2.5,
    )
    axes[1, 1].set_xlabel("ε", fontsize=fontsize)
    axes[1, 1].set_ylabel(r"""$LT_{max}(H_{1})$""", fontsize=fontsize)
    # axes[1, 1].set_title(f"H1 Mean vs ε (noise={noise_filter})", fontsize=fontsize)
    axes[1, 1].tick_params(axis="x", labelsize=fontsize)
    axes[1, 1].tick_params(axis="y", labelsize=fontsize)
    for spine in axes[1, 1].spines.values():
        spine.set_linewidth(2.0)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_projection_comparison_2x2(
    s1p,
    s2p,
    depsilon_filter=0.4,
    noise_filter=0.01,
    figsize=(12, 10),
    fontsize=15.0,
    output_file=None,
):
    """
    Create 2x2 grid comparing projection dimensions using KDE plots and line plots.

    Top row: KDE distributions at specific depsilon and noise
    Bottom row: Mean homology vs depsilon for different projection dimensions

    Args:
        s1p: DataFrame with scenario 1 projection results
        s2p: DataFrame with scenario 2 projection results
        depsilon_filter: Depsilon value for KDE plots (default: 0.4)
        noise_filter: Noise scale for plots (default: 0.01)
        figsize: Figure size (default: (12, 10))
        fontsize: Font size for labels (default: 15.0)
        output_file: Optional output filename

    Returns:
        Matplotlib figure object
    """
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Filter out 'original' projection dimension
    s1p = s1p[s1p.projection_dim != "original"]
    s2p = s2p[s2p.projection_dim != "original"]

    # Get unique projection dimensions
    projection_dims = sorted([int(pd) for pd in s1p["projection_dim"].unique()])
    print(f"Available projection dimensions: {projection_dims}")

    # Create color palettes with different shades for each projection dimension
    # Red shades for parent_to_children (s1p), blue shades for children_to_parent (s2p)
    n_dims = len(projection_dims)
    red_colors = sns.light_palette("red", n_colors=n_dims + 2)[2:]  # Skip lightest shades
    blue_colors = sns.light_palette("blue", n_colors=n_dims + 2)[2:]

    linestyles = ["-", "--", "-.", ":"]

    # Top row: KDE plots at specific depsilon and noise_scale
    for idx, proj_dim in enumerate(projection_dims):
        # Filter data for this projection dimension
        s1_filtered = s1p[
            (s1p.projection_dim == proj_dim) & (s1p.depsilon == depsilon_filter) & (s1p.noise_scale == noise_filter)
        ].reset_index(drop=True)
        s2_filtered = s2p[
            (s2p.projection_dim == proj_dim) & (s2p.depsilon == depsilon_filter) & (s2p.noise_scale == noise_filter)
        ].reset_index(drop=True)

        # Plot w1_h0
        if len(s1_filtered) > 0:
            sns.kdeplot(
                data=s1_filtered,
                x="w1_h0",
                ax=axes[0][0],
                color=red_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                label=f"P→C (dim={proj_dim})",
            )
        if len(s2_filtered) > 0:
            sns.kdeplot(
                data=s2_filtered,
                x="w1_h0",
                ax=axes[0][0],
                color=blue_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                label=f"C→P (dim={proj_dim})",
            )

        # Plot ltmax_h1
        if len(s1_filtered) > 0:
            sns.kdeplot(
                data=s1_filtered,
                x="ltmax_h1",
                ax=axes[0][1],
                color=red_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                label=f"P→C (dim={proj_dim})",
            )
        if len(s2_filtered) > 0:
            sns.kdeplot(
                data=s2_filtered,
                x="ltmax_h1",
                ax=axes[0][1],
                color=blue_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                label=f"C→P (dim={proj_dim})",
            )

    axes[0][0].set_xlim(0.1, 1.0)
    axes[0][0].set_xlabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    axes[0][0].set_ylabel("Density", fontsize=fontsize)
    #axes[0][0].set_title(f"ε={depsilon_filter}, noise={noise_filter}", fontsize=fontsize)
    axes[0][0].legend(fontsize=8, loc="best")

    axes[0][1].set_xlabel(r"""$LT_{max}(H_{1})$""", fontsize=fontsize)
    axes[0][1].set_ylabel("Density", fontsize=fontsize)
    #axes[0][1].set_title(f"ε={depsilon_filter}, noise={noise_filter}", fontsize=fontsize)
    axes[0][1].legend(fontsize=8, loc="best")

    for ax in axes[0]:
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.grid(True, alpha=0.3)

    # Bottom row: Line plots vs depsilon for different projection dimensions
    for idx, proj_dim in enumerate(projection_dims):
        s1_filtered = s1p[(s1p.projection_dim == proj_dim) & (s1p.noise_scale == noise_filter)]
        s2_filtered = s2p[(s2p.projection_dim == proj_dim) & (s2p.noise_scale == noise_filter)]

        if len(s1_filtered) > 0:
            s1_grouped = s1_filtered.groupby("depsilon")[["w1_h0", "ltmax_h1"]].mean().reset_index()
            axes[1][0].plot(
                s1_grouped["depsilon"],
                s1_grouped["w1_h0"],
                color=red_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                marker="o",
                markersize=5,
                label=f"P→C (dim={proj_dim})",
            )
            axes[1][1].plot(
                s1_grouped["depsilon"],
                s1_grouped["ltmax_h1"],
                color=red_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                marker="o",
                markersize=5,
                label=f"P→C (dim={proj_dim})",
            )

        if len(s2_filtered) > 0:
            s2_grouped = s2_filtered.groupby("depsilon")[["w1_h0", "ltmax_h1"]].mean().reset_index()
            axes[1][0].plot(
                s2_grouped["depsilon"],
                s2_grouped["w1_h0"],
                color=blue_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                marker="s",
                markersize=5,
                label=f"C→P (dim={proj_dim})",
            )
            axes[1][1].plot(
                s2_grouped["depsilon"],
                s2_grouped["ltmax_h1"],
                color=blue_colors[idx],
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                marker="s",
                markersize=5,
                label=f"C→P (dim={proj_dim})",
            )

    axes[1][0].set_xlabel("ε", fontsize=fontsize)
    axes[1][0].set_ylabel(r"""$W_{1}(H_{0})$""", fontsize=fontsize)
    #axes[1][0].set_title(f"noise={noise_filter}", fontsize=fontsize)
    axes[1][0].legend(fontsize=8, loc="best")

    axes[1][1].set_xlabel("ε", fontsize=fontsize)
    axes[1][1].set_ylabel(r"""$LT_{max}(H_{1})$""", fontsize=fontsize)
    #axes[1][1].set_title(f"noise={noise_filter}", fontsize=fontsize)
    axes[1][1].legend(fontsize=8, loc="best")

    for ax in axes[1]:
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")

    return fig


def plot_homology_histograms_2x2_multi_noise(
    results_s1,
    results_s2,
    depsilon_filter=0.4,
    noise_levels=None,
    figsize=(18, 12),
    fontsize=15.0,
):
    """
    Create 2x2 grid plots for multiple noise levels (one figure per noise level).

    Args:
        results_s1: DataFrame with scenario 1 results
        results_s2: DataFrame with scenario 2 results
        depsilon_filter: Depsilon value for KDE plots
        noise_levels: List of noise levels to plot (default: all unique noise levels)
        figsize: Figure size per noise level
        fontsize: Font size for labels

    Returns:
        Dictionary mapping noise_level to figure object
    """
    if noise_levels is None:
        noise_levels = sorted(set(results_s1["noise_scale"].unique()) & set(results_s2["noise_scale"].unique()))

    figures = {}
    for noise in noise_levels:
        fig = plot_homology_histograms_2x2(
            results_s1=results_s1,
            results_s2=results_s2,
            depsilon_filter=depsilon_filter,
            noise_filter=noise,
            figsize=figsize,
            fontsize=fontsize,
        )
        figures[noise] = fig

    return figures
