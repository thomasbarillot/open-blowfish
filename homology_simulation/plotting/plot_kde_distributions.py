"""
KDE and distribution plots for homology experiments.

This module contains functions for KDE visualizations and distributions by dimension ratio.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_kde_histograms_by_dimratio(results_s1, results_s2, figsize=(12, 18)):
    """
    Plot KDE histograms of w1_h0 and ltmax_h1 for different dimension ratio ranges.
    Creates a grid where rows = noise levels, columns = features (w1_h0, ltmax_h1).
    Red: dim_ratio > 1 and <= 5 (parent -> children, Scenario 1 dominated)
    Blue: dim_ratio < 1 (child -> parent, Scenario 2 dominated)
    Uses depsilon = 0.4

    Args:
        results_s1: DataFrame with scenario 1 results
        results_s2: DataFrame with scenario 2 results
        figsize: Figure size
    """
    features = ["w1_h0", "ltmax_h1"]
    titles = {
        "w1_h0": "0th Homology (w1_h0) KDE",
        "ltmax_h1": "Max 1st Homology Lifetime KDE",
    }

    # Target depsilon
    target_depsilon = 0.4

    # Combine both scenarios
    combined_df = pd.concat([results_s1, results_s2], ignore_index=True)

    # Filter by depsilon
    combined_df = combined_df[combined_df["depsilon"] == target_depsilon].copy()

    # Get noise levels
    noise_levels = sorted(combined_df["noise_scale"].unique())

    print(f"Plotting KDE histograms: {len(noise_levels)} noise levels, depsilon={target_depsilon}")

    # Create grid: rows = noise levels, columns = features
    n_noise = len(noise_levels)
    n_features = len(features)
    fig, axes = plt.subplots(n_noise, n_features, figsize=figsize)

    # Ensure axes is 2D
    if n_noise == 1 and n_features == 1:
        axes = np.array([[axes]])
    elif n_noise == 1:
        axes = axes.reshape(1, -1)
    elif n_features == 1:
        axes = axes.reshape(-1, 1)

    BINS = np.arange(0, 1, 0.01)

    for noise_idx, noise_scale in enumerate(noise_levels):
        for feat_idx, feature in enumerate(features):
            ax = axes[noise_idx, feat_idx]

            # Filter data for this noise level
            noise_data = combined_df[
                (combined_df["noise_scale"] == noise_scale)
                & ~combined_df[feature].isna()
                & ~combined_df["dim_ratio"].isna()
            ].copy()

            if len(noise_data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Filter by dimension ratio ranges
            # Red: dim_ratio > 1 and <= 5 (parent -> children)
            high_dimratio = noise_data[(noise_data["dim_ratio"] > 1) & (noise_data["dim_ratio"] <= 5)][
                feature
            ].dropna()

            # Blue: dim_ratio < 1 (child -> parent)
            low_dimratio = noise_data[noise_data["dim_ratio"] < 1][feature].dropna()

            # Plot KDE histograms
            if len(high_dimratio) > 0:
                ax.hist(
                    high_dimratio,
                    bins=BINS,
                    alpha=0.5,
                    color="red",
                    label=f"dim_ratio > 1 (n={len(high_dimratio)})",
                    density=True,
                    edgecolor="darkred",
                )
                # KDE overlay
                if len(high_dimratio) > 5:
                    kde_high = gaussian_kde(high_dimratio)
                    x_range = np.linspace(high_dimratio.min(), high_dimratio.max(), 200)
                    ax.plot(x_range, kde_high(x_range), color="darkred", linewidth=2)

            if len(low_dimratio) > 0:
                ax.hist(
                    low_dimratio,
                    bins=BINS,
                    alpha=0.5,
                    color="blue",
                    label=f"dim_ratio < 1 (n={len(low_dimratio)})",
                    density=True,
                    edgecolor="darkblue",
                )
                # KDE overlay
                if len(low_dimratio) > 5:
                    kde_low = gaussian_kde(low_dimratio)
                    x_range = np.linspace(low_dimratio.min(), low_dimratio.max(), 200)
                    ax.plot(x_range, kde_low(x_range), color="darkblue", linewidth=2)

            # Set title for top row
            if noise_idx == 0:
                ax.set_title(titles[feature], fontweight="bold", fontsize=11)

            # Set ylabel for first column
            if feat_idx == 0:
                ax.set_ylabel(f"noise={noise_scale}\nDensity", fontweight="bold", fontsize=9)
            else:
                ax.set_ylabel("Density", fontsize=8)

            # Set xlabel for bottom row
            if noise_idx == n_noise - 1:
                ax.set_xlabel(feature, fontsize=9)

            # Only show legend for first subplot
            if noise_idx == 0 and feat_idx == 0:
                ax.legend(loc="best", fontsize=7)
            if feature == "w1_h0":
                ax.set_xlim(0.1, 0.8)
            else:
                ax.set_xlim(0.0, 0.2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle(
        f"KDE Histograms: Homology Features by Dimension Ratio\n"
        f"ε={target_depsilon} | Red: dim_ratio > 1 (Parent→Children) | "
        f"Blue: dim_ratio < 1 (Child→Parent)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_homology_vs_depsilon_by_dimratio(results_s1, results_s2, figsize=(12, 18)):
    """
    Plot w1_h0 and ltmax_h1 as a function of depsilon for different dimension ratio ranges.
    Creates a grid where rows = noise levels, columns = features (w1_h0, ltmax_h1).
    Red: dim_ratio > 1 and <= 5 (parent -> children)
    Blue: dim_ratio < 1 (child -> parent)
    Shows mean ± std error bars.

    Args:
        results_s1: DataFrame with scenario 1 results
        results_s2: DataFrame with scenario 2 results
        figsize: Figure size
    """
    features = ["w1_h0", "ltmax_h1"]
    titles = {
        "w1_h0": "0th Homology (w1_h0) vs ε",
        "ltmax_h1": "Max 1st Homology Lifetime vs ε",
    }

    # Combine both scenarios
    combined_df = pd.concat([results_s1, results_s2], ignore_index=True)

    # Get noise levels and depsilon values
    noise_levels = sorted(combined_df["noise_scale"].unique())
    depsilon_values = sorted(combined_df["depsilon"].unique())

    print(
        f"Plotting homology vs depsilon by dim ratio: {len(noise_levels)} noise levels, "
        f"{len(depsilon_values)} depsilon values"
    )

    # Create grid: rows = noise levels, columns = features
    n_noise = len(noise_levels)
    n_features = len(features)
    fig, axes = plt.subplots(n_noise, n_features, figsize=figsize)

    # Ensure axes is 2D
    if n_noise == 1 and n_features == 1:
        axes = np.array([[axes]])
    elif n_noise == 1:
        axes = axes.reshape(1, -1)
    elif n_features == 1:
        axes = axes.reshape(-1, 1)

    for noise_idx, noise_scale in enumerate(noise_levels):
        for feat_idx, feature in enumerate(features):
            ax = axes[noise_idx, feat_idx]

            # Filter data for this noise level
            noise_data = combined_df[
                (combined_df["noise_scale"] == noise_scale)
                & ~combined_df[feature].isna()
                & ~combined_df["dim_ratio"].isna()
                & ~combined_df["depsilon"].isna()
            ].copy()

            if len(noise_data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Process high dim_ratio (> 1 and <= 5)
            high_dimratio = noise_data[(noise_data["dim_ratio"] > 1) & (noise_data["dim_ratio"] <= 5)]
            if len(high_dimratio) > 0:
                grouped_high = high_dimratio.groupby("depsilon")[feature].agg(["mean", "std", "count"])
                grouped_high["std"] = grouped_high["std"].fillna(0)

                ax.errorbar(
                    grouped_high.index,
                    grouped_high["mean"],
                    yerr=grouped_high["std"],
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    capsize=4,
                    color="red",
                    label=f"dim_ratio > 1 (n={len(high_dimratio)})",
                    alpha=0.8,
                )

            # Process low dim_ratio (< 1)
            low_dimratio = noise_data[noise_data["dim_ratio"] < 1]
            if len(low_dimratio) > 0:
                grouped_low = low_dimratio.groupby("depsilon")[feature].agg(["mean", "std", "count"])
                grouped_low["std"] = grouped_low["std"].fillna(0)

                ax.errorbar(
                    grouped_low.index,
                    grouped_low["mean"],
                    yerr=grouped_low["std"],
                    marker="s",
                    markersize=5,
                    linewidth=2,
                    capsize=4,
                    color="blue",
                    label=f"dim_ratio < 1 (n={len(low_dimratio)})",
                    alpha=0.8,
                )

            # Set title for top row
            if noise_idx == 0:
                ax.set_title(titles[feature], fontweight="bold", fontsize=11)

            # Set ylabel for first column
            if feat_idx == 0:
                ax.set_ylabel(f"noise={noise_scale}\n{feature}", fontweight="bold", fontsize=9)
            else:
                ax.set_ylabel(feature, fontsize=8)

            # Set xlabel for bottom row
            if noise_idx == n_noise - 1:
                ax.set_xlabel("ε (depsilon)", fontsize=9)

            # Set x-axis to log scale for better visualization
            ax.set_xscale("linear")

            # Only show legend for first subplot
            if noise_idx == 0 and feat_idx == 0:
                ax.legend(loc="best", fontsize=7)

            # Set y-axis limits
            if feature == "w1_h0":
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylim(0.0, 0.25)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle(
        "Homology Features vs ε (depsilon) by Dimension Ratio\n"
        "Red: dim_ratio > 1 (Parent→Children) | Blue: dim_ratio < 1 (Child→Parent)\n"
        "Error bars show ±1 std",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig
