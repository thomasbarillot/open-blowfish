# %%
import itertools

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from .utils import load_scenario_data_from_s3

# %%
# Setup
NOISE_SCALE_LOW = 0.0
NOISE_SCALE_HIGH = 0.9
x_h0 = np.arange(0, 2, 0.05)
x_h0c = (x_h0[1:] + x_h0[:-1]) / 2.0

mycmap = cm.viridis
mycmap.set_under("w")


def create_histogram(df):
    """Create normalized histogram from h0_dist data"""
    h0s = list(itertools.chain.from_iterable([a for a in df["h0_dist"].dropna()]))
    hist, _ = np.histogram(h0s, bins=x_h0)
    return hist / np.sum(hist) if np.sum(hist) > 0 else hist

#%%
# Load data for all scenarios
print("Loading data...")
data = {}
n_parent_features = 8
embedding_dim = 64
max_features = 16

for scenario_num in [1, 2, 3, 4]:
    print(f"Loading scenario {scenario_num}...")
    data[scenario_num] = load_scenario_data_from_s3(
        scenario_num=scenario_num,
        n_parent_features=n_parent_features,
        embedding_dim=embedding_dim,
        max_features=max_features,
        with_dist=True
    )

# %%
# Determine unique active dims for each scenario
active_dims_ranges = {}
for scenario_num in [1, 2, 3, 4]:
    df = data[scenario_num]
    if df is None:
        active_dims_ranges[scenario_num] = {"column": [], "values": []}
        continue
    df_filtered = df[df.noise_scale == NOISE_SCALE_LOW].dropna()

    if scenario_num == 2:
        # For scenario 2, use query_active_dims
        active_dims_col = "query_active_dims"
    else:
        # For scenarios 1, 3, 4, use corpus_active_dims
        active_dims_col = "corpus_active_dims"

    active_dims = [d for d in sorted(df_filtered[active_dims_col].unique()) if d != 1]

    active_dims_ranges[scenario_num] = {
        "column": active_dims_col,
        "values": active_dims,
    }
    print(f"Scenario {scenario_num} ({active_dims_col}): {active_dims_ranges[scenario_num]['values']}")

# %%
# Determine number of rows (max number of unique active dims across all scenarios)
max_rows = max(len(active_dims_ranges[s]["values"]) for s in [1, 2, 3, 4])
print(f"\nCreating grid with {max_rows} rows x 4 columns")

# Create the grid plot
fig, axes = plt.subplots(max_rows, 4, figsize=(20, 5 * max_rows), constrained_layout=True)

# Ensure axes is 2D even if max_rows == 1
if max_rows == 1:
    axes = axes.reshape(1, -1)

# Plot for each scenario
for col_idx, scenario_num in enumerate([1, 2, 3, 4]):
    print(f"\nPlotting scenario {scenario_num}...")
    df = data[scenario_num]
    if df is None:
        continue
    df_filtered_low = df[df.noise_scale == NOISE_SCALE_LOW].dropna()
    df_filtered_high = df[df.noise_scale == NOISE_SCALE_HIGH].dropna()

    active_dims_col = active_dims_ranges[scenario_num]["column"]
    active_dims_values = active_dims_ranges[scenario_num]["values"]

    for row_idx, active_dim_value in enumerate(active_dims_values):
        ax = axes[row_idx, col_idx]

        # Filter by active dimension value for low noise (0.01)
        case_df_low = df_filtered_low[df_filtered_low[active_dims_col] == active_dim_value].dropna()

        if len(case_df_low) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("H0 distance / 2")
            ax.set_ylabel("depsilon")
            continue

        # Create histogram grouped by depsilon for low noise
        h0_hist_low = case_df_low.groupby(by="depsilon").apply(create_histogram)

        # Plot low noise (0.01) as background heatmap
        pcm = ax.pcolormesh(
            x_h0c / 2.0,
            sorted(set(case_df_low.depsilon)),
            np.array(h0_hist_low.to_list()),
            cmap=mycmap,
            vmin=1e-9
        )

        # Filter by active dimension value for high noise (0.9)
        case_df_high = df_filtered_high[df_filtered_high[active_dims_col] == active_dim_value].dropna()

        # Add mean and variance overlay for high noise (0.9) in red
        if len(case_df_high) > 0:
            # Calculate mean and std for each depsilon value
            def get_mean_std(df):
                h0s = list(itertools.chain.from_iterable([a for a in df["h0_dist"].dropna()]))
                if len(h0s) > 0:
                    return pd.Series({'mean': np.mean(h0s), 'std': np.std(h0s)})
                return pd.Series({'mean': np.nan, 'std': np.nan})

            stats_high = case_df_high.groupby(by="depsilon").apply(get_mean_std).reset_index()
            stats_high = stats_high.dropna()

            if len(stats_high) > 0:
                depsilon_vals = stats_high['depsilon'].values
                mean_vals = stats_high['mean'].values / 2.0  # Divide by 2 to match x-axis scaling
                std_vals = stats_high['std'].values / 2.0

                # Plot mean as red line
                ax.plot(mean_vals, depsilon_vals, 'r-', linewidth=2, label=f'Mean (noise={NOISE_SCALE_HIGH})')

                # Plot variance as shaded region
                ax.fill_betweenx(
                    depsilon_vals,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color='red',
                    alpha=0.3,
                    label=f'±1 std (noise={NOISE_SCALE_HIGH})'
                )

        ax.set_ylim(0.01, 4)
        ax.set_xlim(-0.01, 0.8)
        ax.set_xlabel("H0 distance / 2")
        ax.set_ylabel("depsilon")
        ax.set_yscale("log")

        # Add title with scenario and active dims info
        active_dim_label = "query_active_dims" if scenario_num == 2 else "corpus_active_dims"
        ax.set_title(
            f"Scenario {scenario_num}\n{active_dim_label}={active_dim_value}",
            fontsize=10
        )

        # Add colorbar to rightmost column
        if col_idx == 3:
            plt.colorbar(pcm, ax=ax, label="Normalized count (noise=0.01)")

    # Hide empty subplots
    for row_idx in range(len(active_dims_values), max_rows):
        axes[row_idx, col_idx].axis("off")

# Add legend for the red overlay
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label=f'Mean H0 distance (noise={NOISE_SCALE_HIGH})'),
    Patch(facecolor='red', alpha=0.3, label=f'±1 std (noise={NOISE_SCALE_HIGH})')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.suptitle(
    f"H0 Distance Distributions Across Scenarios\n"
    f"Background heatmap: noise={NOISE_SCALE_LOW}, Red line/band: mean±std at noise={NOISE_SCALE_HIGH}",
    fontsize=16,
    y=1.0
)
output_file = f"homology_h0_dist_grid_all_scenarios_noise{NOISE_SCALE_LOW}_vs_{NOISE_SCALE_HIGH}.png"

plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.show()


# %%
import math


def combinations(n, k):

    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
# %%
