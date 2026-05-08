# Homology Simulation

This package contains tools for analyzing persistent homology on hierarchical feature-based clusters.

## Structure

```
homology_simulation/
├── __init__.py
├── requirements.txt                          # Third-party dependencies
├── feature_bank_generator.py                 # Feature bank + ClusterNode / FeaturesGenerator
├── homology_calculation.py                   # Homology computation (ripser-based)
├── simulation_scenarios.py                   # Scenario definitions + ScenarioFactory
├── run_homology_simulation.py                # Main analysis script (scenarios 1–4)
├── run_homology_simulation_with_projection.py# Analysis with random projection
├── utils.py                                  # S3 loading + data-processing helpers
├── plot_homologies_correlations.py           # Correlation/KDE plots (IPython cells)
├── plot_homologies_grid_scenarios.py         # H0-distance grid across scenarios
└── plotting/                                 # Plotting subpackage
    ├── __init__.py
    ├── plot_heatmaps.py                      # Heatmap visualizations
    ├── plot_scenarios_comparison.py          # Scenario comparison plots
    ├── plot_homology_histograms.py           # 2x2 KDE grids (includes projection)
    └── plot_kde_distributions.py             # KDE and distribution plots
```

## Installation

Install third-party dependencies:

```bash
pip install -r homology_simulation/requirements.txt
```

Dependencies: `boto3`, `matplotlib`, `numpy`, `pandas`, `ripser`, `scikit-learn`, `scipy`, `seaborn`, `tqdm`.

### External (non-PyPI) dependencies

Several scripts import modules that are **not** part of this repository and must be provided on the `PYTHONPATH` separately:

- `core.connections.get_boto_session` (or `common_tools.connections.get_boto_session`)
- `core.common_tools.EC2Logger`, `core.common_tools.LambdaLogger`
- `config.AWS_PROFILE_NAME`, `config.S3_EXPERIMENT_RETRIEVAL_AMBIGUITY`, `config.pre`

Files affected: `utils.py`, `run_homology_simulation.py`, `run_homology_simulation_with_projection.py`, `plot_homologies_correlations.py`.

## Usage

The package is imported as `homology_simulation`. Run it from the repository root so Python can resolve the package.

### Run a simulation

```bash
python -m homology_simulation.run_homology_simulation
python -m homology_simulation.run_homology_simulation_with_projection
```

### Load data from S3

```python
from homology_simulation.utils import load_scenario_data_from_s3

data = load_scenario_data_from_s3(
    scenario_num=1,
    n_parent_features=16,
    embedding_dim=256,
    max_features=64,
)
```

### Generate plots

```python
from homology_simulation.plotting import plot_homology_histograms_2x2
from homology_simulation.utils import load_scenario_data_from_s3

s1 = load_scenario_data_from_s3(scenario_num=1, ...)
s2 = load_scenario_data_from_s3(scenario_num=2, ...)

fig = plot_homology_histograms_2x2(
    results_s1=s1,
    results_s2=s2,
    depsilon_filter=0.4,
    noise_filter=0.01,
)
fig.savefig("output.png", dpi=150)
```

### Use the scenario factory

```python
from homology_simulation.simulation_scenarios import ScenarioFactory

run_scenario = ScenarioFactory.scenario_map[1]  # or 2, 3, 4
results_df = run_scenario(datapoints, noise_scale=0.1, depsilon=[0.1, 0.4, 1.0])
```

## Available Plot Functions

Re-exported from `homology_simulation.plotting`:

### `plot_homology_histograms_2x2`
2x2 grid with KDE distributions (top) and mean vs depsilon (bottom) for H0 and H1.
- **Parameters**: `results_s1`, `results_s2`, `depsilon_filter`, `noise_filter`, `figsize`, `fontsize`
- **Scenarios**: 1, 2

### `plot_projection_comparison_2x2`
Like `plot_homology_histograms_2x2` but for projection-dimension comparisons.
- **Parameters**: `s1p`, `s2p`, `depsilon_filter`, `noise_filter`, `figsize`, `fontsize`, `output_file`
- **Scenarios**: 1, 2 (projection data)

### `plot_homology_noise_epsilon_heatmaps`
Heatmaps showing homology metrics vs epsilon and noise.
- **Parameters**: `results`, `scenario_num`, `num_parents_list`, `figsize`, `output_file`, `use_median`
- **Scenarios**: single scenario

### `plot_scenarios_comparison_by_noise`
Grid comparison of scenario 1 vs 2 across noise levels and features.
- **Parameters**: `results_s1`, `results_s2`, `features_to_plot`, `figsize`
- **Scenarios**: 1, 2

### `plot_scenario_3_results`
Grid plot for scenario 3 results.
- **Parameters**: `results_df`, `figsize`
- **Scenarios**: 3

### `plot_homology_vs_depsilon`
Homology features as a function of depsilon for different dimension ratios.
- **Parameters**: `results_s1`, `results_s2`, `figsize`
- **Scenarios**: 1, 2

### `plot_kde_histograms_by_dimratio`
KDE histograms split by dimension-ratio ranges.
- **Parameters**: `results_s1`, `results_s2`, `figsize`
- **Scenarios**: 1, 2

### `plot_homology_vs_depsilon_by_dimratio`
Mean ± std vs depsilon split by dimension-ratio ranges.
- **Parameters**: `results_s1`, `results_s2`, `figsize`
- **Scenarios**: 1, 2

### `plot_homology_histogram_with_scale`, `plot_homology_scales`, `plot_homology_histograms_2x2_multi_noise`
Additional variants exported from `homology_simulation.plotting`.

## Utility Functions (`homology_simulation.utils`)

### S3 data loading
- `load_scenario_data_from_s3()` — load regular scenario data
- `load_scenario_data_from_s3_projection()` — load projection data
- `list_available_s3_files()` / `list_available_s3_files_projection()` — list files with filters
- `upload_dataframe_to_s3()` — upload a DataFrame to S3

### Data processing
- `create_histogram()` / `create_normalized_histogram()` — histogram from `h0_dist`
- `zero_overlap_probability()` — zero-overlap probability
- `combinations()` — binomial coefficient
- `parse_metadata_from_filename()` — extract metadata from an S3 filename

## Scenarios

Defined in `simulation_scenarios.py` and exposed via `ScenarioFactory.scenario_map`:

1. **`scenario_1_parent_to_children`** — query from parent outliers, corpus from children datapoints
2. **`scenario_2_child_to_parent`** — query from child datapoints, corpus from parent outliers
3. **`scenario_3_parent_to_child_no_overlap`** — parent to children with no feature overlap among the non-direct children
4. **`scenario_4_parent_to_child_var_overlap`** — parent to children with variable feature overlap
