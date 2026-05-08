"""
Scenario comparison plots for homology experiments.

This module contains functions for comparing different scenarios:
- Scenario 1 vs Scenario 2 comparison
- Scenario 3 results
- Homology vs depsilon for different dimension ratios
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scenarios_comparison_by_noise(results_s1, results_s2, features_to_plot, figsize=(20, 16)):
    """
    Plot comparison between Scenario 1 and Scenario 2 for specific features.
    Creates a grid where rows = noise levels, columns = features.
    Color hue is based on depsilon values (same colors for both scenarios).

    Args:
        results_s1: DataFrame with scenario 1 results
        results_s2: DataFrame with scenario 2 results
        features_to_plot: List of feature names to plot
        figsize: Figure size
    """
    n_features = len(features_to_plot)

    feature_titles = {
        "w1_h0": "0th Homology (w1_h0)",
        "ltmax_h1": "Max 1st Homology Lifetime",
        "nneighbours": "Number of Neighbours",
        "mean_homology_birth": "Mean Birth Time",
        "std_homology_birth": "Std Birth Time",
        "mean_homology1st_lifetime": "Mean 1st Homology Lifetime",
    }

    # Get common noise levels and depsilon values
    noise_levels = sorted(set(results_s1["noise_scale"].unique()) & set(results_s2["noise_scale"].unique()))
    depsilon_values = sorted(set(results_s1["depsilon"].unique()) & set(results_s2["depsilon"].unique()))

    print(f"Plotting {len(noise_levels)} noise levels: {noise_levels}")
    print(f"Plotting {len(depsilon_values)} depsilon values: {depsilon_values}")

    # Color scheme based on depsilon (same for both scenarios)
    colors = plt.cm.viridis(np.linspace(0, 1, len(depsilon_values)))

    # Create grid: rows = noise levels, columns = features
    n_noise = len(noise_levels)
    fig, axes = plt.subplots(n_noise, n_features, figsize=figsize)

    # Ensure axes is 2D
    if n_noise == 1 and n_features == 1:
        axes = np.array([[axes]])
    elif n_noise == 1:
        axes = axes.reshape(1, -1)
    elif n_features == 1:
        axes = axes.reshape(-1, 1)

    for noise_idx, noise_scale in enumerate(noise_levels):
        for feat_idx, feature in enumerate(features_to_plot):
            ax = axes[noise_idx, feat_idx]

            # Plot each depsilon value
            for eps_idx, depsilon in enumerate(depsilon_values):
                # Process Scenario 1
                s1_data = results_s1[
                    (results_s1["noise_scale"] == noise_scale)
                    & (results_s1["depsilon"] == depsilon)
                    & ~results_s1[feature].isna()
                    & ~results_s1["dim_ratio"].isna()
                ].copy()

                if len(s1_data) > 0:
                    # Bin dim_ratio for better visualization
                    Bins = sorted(list(set(s1_data["dim_ratio"])))
                    Bins.append(Bins[-1] + 1)
                    s1_data["dim_ratio_binned"] = pd.cut(s1_data["dim_ratio"], bins=Bins)

                    s1_data["dim_ratio_center"] = s1_data["dim_ratio_binned"].apply(
                        lambda x: x.mid if pd.notna(x) else np.nan
                    )
                    grouped_s1 = s1_data.groupby("dim_ratio_center")[feature].agg(["mean", "std", "count"])
                    grouped_s1["std"] = grouped_s1["std"].fillna(0)

                    if len(grouped_s1) > 0:
                        ax.plot(
                            grouped_s1.index,
                            grouped_s1["mean"],
                            marker="o",
                            markersize=4,
                            linewidth=1.5,
                            color=colors[eps_idx],
                            label=f"S1: ε={depsilon}",
                            alpha=0.8,
                            linestyle="-",
                        )

                # Process Scenario 2
                s2_data = results_s2[
                    (results_s2["noise_scale"] == noise_scale)
                    & (results_s2["depsilon"] == depsilon)
                    & ~results_s2[feature].isna()
                    & ~results_s2["dim_ratio"].isna()
                ].copy()

                if len(s2_data) > 0:
                    Bins = sorted(list(set(s2_data["dim_ratio"])))
                    Bins.append(Bins[-1] + 1)
                    s2_data["dim_ratio_binned"] = pd.cut(s2_data["dim_ratio"], bins=Bins)

                    s2_data["dim_ratio_center"] = s2_data["dim_ratio_binned"].apply(
                        lambda x: x.mid if pd.notna(x) else np.nan
                    )
                    grouped_s2 = s2_data.groupby("dim_ratio_center")[feature].agg(["mean", "std", "count"])
                    grouped_s2["std"] = grouped_s2["std"].fillna(0)

                    if len(grouped_s2) > 0:
                        ax.plot(
                            grouped_s2.index,
                            grouped_s2["mean"],
                            marker="s",
                            markersize=4,
                            linewidth=1.5,
                            color=colors[eps_idx],
                            label=f"S2: ε={depsilon}",
                            alpha=0.6,
                            linestyle="--",
                        )

            # Set title for top row
            if noise_idx == 0:
                title = feature_titles.get(feature, feature)
                ax.set_title(title, fontweight="bold", fontsize=11)

            # Set ylabel for first column
            if feat_idx == 0:
                ax.set_ylabel(f"noise={noise_scale}\n{feature}", fontweight="bold", fontsize=9)
            else:
                ax.set_ylabel(feature, fontsize=8)

            # Set xlabel for bottom row
            if noise_idx == n_noise - 1:
                ax.set_xlabel("Dim Ratio", fontsize=9)

            # Only show legend for first subplot
            if noise_idx == 0 and feat_idx == 0:
                ax.legend(loc="best", fontsize=6, ncol=2)

            ax.set_ylim(0.0, 1.0)
            if feature == "ltmax_h1":
                ax.set_ylim(0, 0.3)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle(
        "Scenario 1 (Parent→Children) vs Scenario 2 (Child→Parent)\n"
        "Grid: Rows=Noise Levels, Cols=Features | "
        "Hue=ε (depsilon) | Solid=S1, Dashed=S2",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_scenario_3_results(results_df, figsize=(18, 16)):
    """
    Plot homology features for Scenario 3 with grid layout.
    Creates a grid where rows = noise levels, columns = features.
    Color hue is based on depsilon values.

    Args:
        results_df: DataFrame with scenario 3 results
        figsize: Figure size
    """
    features = [
        "w1_h0",
        "ltmax_h1",
        "mean_homology_birth",
        "std_homology_birth",
        "mean_homology1st_lifetime",
        "nneighbours",
    ]
    titles = [
        "0th Homology (w1_h0)",
        "Max 1st Homology Lifetime",
        "Mean Birth Time",
        "Std Birth Time",
        "Mean 1st Homology Lifetime",
        "Number of Neighbours",
    ]

    noise_levels = sorted(results_df["noise_scale"].unique())
    depsilon_values = sorted(results_df["depsilon"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(depsilon_values)))

    print(f"Scenario 3: {len(noise_levels)} noise levels, {len(depsilon_values)} depsilon values")

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
        for feat_idx, (feature, title) in enumerate(zip(features, titles)):
            ax = axes[noise_idx, feat_idx]

            # Remove NaN values for this noise level
            valid_mask = (
                (results_df["noise_scale"] == noise_scale)
                & ~results_df[feature].isna()
                & ~results_df["dim_ratio"].isna()
            )
            plot_df = results_df[valid_mask]

            # Plot for each depsilon value
            for eps_idx, depsilon in enumerate(depsilon_values):
                eps_df = plot_df[plot_df["depsilon"] == depsilon]

                if len(eps_df) == 0:
                    continue

                # Bin dim_ratio for better visualization
                eps_df = eps_df.copy()
                Bins = sorted(list(set(eps_df["dim_ratio"])))
                Bins.append(Bins[-1] + 1)
                eps_df["dim_ratio_binned"] = pd.cut(eps_df["dim_ratio"], bins=Bins)
                eps_df["dim_ratio_center"] = eps_df["dim_ratio_binned"].apply(
                    lambda x: x.mid if pd.notna(x) else np.nan
                )

                # Group by binned dim_ratio and calculate mean/std
                grouped = eps_df.groupby("dim_ratio_center")[feature].agg(["mean", "std", "count"])
                grouped["std"] = grouped["std"].fillna(0)

                if len(grouped) > 0:
                    ax.errorbar(
                        grouped.index,
                        grouped["mean"],
                        yerr=grouped["std"],
                        marker="^",
                        markersize=4,
                        linewidth=1.5,
                        capsize=2,
                        color=colors[eps_idx],
                        label=f"ε={depsilon}",
                        alpha=0.7,
                    )

            # Set title for top row
            if noise_idx == 0:
                ax.set_title(title, fontweight="bold", fontsize=11)

            # Set ylabel for first column
            if feat_idx == 0:
                ax.set_ylabel(f"noise={noise_scale}\n{feature}", fontweight="bold", fontsize=9)
            else:
                ax.set_ylabel(feature, fontsize=8)

            # Set xlabel for bottom row
            if noise_idx == n_noise - 1:
                ax.set_xlabel("Dim Ratio", fontsize=9)

            # Only show legend for first subplot
            if noise_idx == 0 and feat_idx == 0:
                ax.legend(loc="best", fontsize=6, ncol=2)

            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle(
        "Scenario 3: Variable Overlap\nGrid: Rows=Noise Levels, Cols=Features | Hue=ε (depsilon)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_homology_vs_depsilon(results_s1, results_s2, figsize=(16, 18)):
    """
    Plot homology features as a function of depsilon for each dimension ratio.
    Creates a grid where rows = noise levels, columns = features (w1_h0, ltmax_h1).
    Different lines represent different dimension ratios (binned).

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

    # Get noise levels and depsilon values
    noise_levels = sorted(set(results_s1["noise_scale"].unique()) & set(results_s2["noise_scale"].unique()))
    depsilon_values = sorted(set(results_s1["depsilon"].unique()) & set(results_s2["depsilon"].unique()))

    print(f"Plotting homology vs depsilon: {len(noise_levels)} noise levels, {len(depsilon_values)} depsilon values")

    # Create dimension ratio bins for line colors
    all_dim_ratios = pd.concat([results_s1["dim_ratio"], results_s2["dim_ratio"]]).dropna()
    dim_ratio_bins = pd.qcut(all_dim_ratios, q=5, duplicates="drop")
    dim_ratio_labels = sorted(dim_ratio_bins.unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(dim_ratio_labels)))

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

            # Process both scenarios
            for scenario_data, scenario_label, linestyle, marker in [
                (results_s1, "S1", "-", "o"),
                (results_s2, "S2", "--", "s"),
            ]:
                # Filter data for this noise level
                noise_data = scenario_data[
                    (scenario_data["noise_scale"] == noise_scale)
                    & ~scenario_data[feature].isna()
                    & ~scenario_data["dim_ratio"].isna()
                    & ~scenario_data["depsilon"].isna()
                ].copy()

                if len(noise_data) == 0:
                    continue

                # Bin dimension ratios
                noise_data["dim_ratio_binned"] = pd.cut(noise_data["dim_ratio"], bins=dim_ratio_labels)

                # Plot for each dimension ratio bin
                for bin_idx, dim_bin in enumerate(dim_ratio_labels):
                    bin_data = noise_data[noise_data["dim_ratio_binned"] == dim_bin]

                    if len(bin_data) == 0:
                        continue

                    # Group by depsilon and calculate mean
                    grouped = bin_data.groupby("depsilon")[feature].agg(["mean", "std", "count"])
                    grouped["std"] = grouped["std"].fillna(0)

                    if len(grouped) > 0:
                        # Create label for legend (only for first subplot)
                        if noise_idx == 0 and feat_idx == 0:
                            dim_ratio_mid = dim_bin.mid
                            label = f"{scenario_label}: ratio≈{dim_ratio_mid:.2f}"
                        else:
                            label = None

                        ax.plot(
                            grouped.index,
                            grouped["mean"],
                            marker=marker,
                            markersize=4,
                            linewidth=1.5,
                            color=colors[bin_idx],
                            label=label,
                            alpha=0.7,
                            linestyle=linestyle,
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
                ax.legend(loc="best", fontsize=5, ncol=2, framealpha=0.9, edgecolor="black")

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle(
        "Homology Features vs ε (depsilon) for Different Dimension Ratios\n"
        "Grid: Rows=Noise Levels, Cols=Features | Hue=Dim Ratio | Solid=S1, Dashed=S2",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig
