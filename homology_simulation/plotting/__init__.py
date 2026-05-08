"""
Plotting modules for homology experiments.

This package contains specialized plotting functions for homology analysis:
- plot_heatmaps: Heatmap visualizations
- plot_scenarios_comparison: Scenario comparison plots
- plot_homology_histograms: 2x2 grid KDE plots with mean/std (includes projection comparison)
- plot_kde_distributions: KDE and distribution plots
"""

from .plot_heatmaps import plot_homology_noise_epsilon_heatmaps
from .plot_homology_histograms import (
    plot_homology_histogram_with_scale,
    plot_homology_histograms_2x2,
    plot_homology_histograms_2x2_multi_noise,
    plot_homology_scales,
    plot_projection_comparison_2x2,
)
from .plot_kde_distributions import plot_homology_vs_depsilon_by_dimratio, plot_kde_histograms_by_dimratio
from .plot_scenarios_comparison import (
    plot_homology_vs_depsilon,
    plot_scenario_3_results,
    plot_scenarios_comparison_by_noise,
)

__all__ = [
    "plot_homology_noise_epsilon_heatmaps",
    "plot_scenarios_comparison_by_noise",
    "plot_scenario_3_results",
    "plot_homology_vs_depsilon",
    "plot_homology_histograms_2x2",
    "plot_projection_comparison_2x2",
    "plot_homology_histograms_2x2_multi_noise",
    "plot_kde_histograms_by_dimratio",
    "plot_homology_vs_depsilon_by_dimratio",
    "plot_homology_histogram_with_scale",
    "plot_homology_scales"
]
