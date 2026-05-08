# %%
"""
Persistent Homology Analysis on Recursive Feature-Based Clusters

This script analyzes persistent homology features for five specific scenarios
using hierarchical clusters generated with orthogonal feature-based clustering:

1. Query from parent outliers, corpus from children datapoints
2. Query from node datapoints, corpus from parent outliers
3. Query from node datapoints, corpus from uncorrelated node (same depth)
4. Query from node datapoints, corpus from children + uncorrelated nodes
   - Tests variable fractions (0% to 100%) of uncorrelated nodes in corpus
5. Query from node datapoints, corpus from parent + same-depth uncorrelated parents
   - Tests variable fractions (0% to 100%) of uncorrelated parents in corpus

For scenarios 4 and 5, the script explores how adding uncorrelated noise to the
corpus affects homology features across different mixing ratios.
"""

import json
import sys
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from core.connections import get_boto_session
from tqdm import tqdm

# Import the recursive clustering generator and scenarios
from .feature_bank_generator import ClusterNode, FeaturesGenerator
from .simulation_scenarios import (  # scenario_3_uncorrelated_nodes,; scenario_4_children_plus_uncorrelated,; scenario_5_parent_plus_uncorrelated_parents,
    ScenarioFactory,
    scenario_1_parent_to_children,
    scenario_2_child_to_parent,
    scenario_3_parent_to_child_no_overlap,
    scenario_4_parent_to_child_var_overlap,
)

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.common_tools import EC2Logger, LambdaLogger

from config import AWS_PROFILE_NAME, S3_EXPERIMENT_RETRIEVAL_AMBIGUITY, pre

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# Initialize S3 client and logger
s3_client = get_boto_session(profile_name=AWS_PROFILE_NAME).client("s3")
logger = EC2Logger(
    pre("HomologySimulationsBoostResearchMachine") + "Logs", "homology_simulation_with_dists_128only_logstream"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def upload_dataframe_to_s3(df, s3_key, bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY):
    """
    Upload a pandas DataFrame to S3 as JSON.

    Args:
        df: pandas DataFrame to upload
        s3_key: S3 key (path) for the object
        bucket_name: S3 bucket name (default from config)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert DataFrame to JSON (orient='records' for list of dicts)
        json_data = df.to_json(orient="records", date_format="iso")

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_data.encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return False


def _convert_numpy_to_serializable(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.

    Args:
        obj: Any object that may contain numpy arrays

    Returns:
        JSON-serializable version of the object
    """
    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: _convert_numpy_to_serializable(value) for key, value in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_serializable(item) for item in obj]

    # Return other types as-is
    return obj


def upload_datapoints_to_s3(datapoints_dict, s3_key, bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY):
    """
    Upload generated datapoints to S3 as JSON.

    Args:
        datapoints_dict: Dictionary containing datapoints and metadata
        s3_key: S3 key (path) for the object
        bucket_name: S3 bucket name (default from config)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert all nested numpy arrays to lists recursively
        serializable_dict = _convert_numpy_to_serializable(datapoints_dict)

        json_data = json.dumps(serializable_dict, indent=2)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json_data.encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading datapoints to S3: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def load_datapoints_from_s3(s3_key, bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY):
    """
    Load previously saved datapoints from S3.

    Args:
        s3_key: S3 key (path) for the object
        bucket_name: S3 bucket name (default from config)

    Returns:
        dict: Dictionary containing datapoints and metadata, or None if failed
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        json_data = response["Body"].read().decode("utf-8")
        datapoints_package = json.loads(json_data)

        # Convert lists back to numpy arrays where needed
        if "parent_datapoints" in datapoints_package:
            for dp in datapoints_package["parent_datapoints"]:
                if "embedding" in dp and isinstance(dp["embedding"], list):
                    dp["embedding"] = np.array(dp["embedding"])

        if "all_children_datapoints" in datapoints_package:
            for children_group in datapoints_package["all_children_datapoints"]:
                for dp in children_group:
                    if "embedding" in dp and isinstance(dp["embedding"], list):
                        dp["embedding"] = np.array(dp["embedding"])

        return datapoints_package
    except Exception as e:
        logger.error(f"Error loading datapoints from S3: {e}")
        return None


def list_available_datapoints(
    embedding_dim=None,
    seed=None,
    bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
):
    """
    List available saved datapoints in S3.

    Args:
        embedding_dim: Filter by embedding dimension (optional)
        seed: Filter by random seed (optional)
        bucket_name: S3 bucket name (default from config)

    Returns:
        list: List of S3 keys matching the filters
    """
    try:
        prefix = "homology_simulations/datapoints/"

        # Build filter pattern
        filter_pattern = "datapoints_"
        if embedding_dim is not None:
            filter_pattern += f"d{embedding_dim}_"
        if seed is not None:
            filter_pattern += f"seed{seed}_"

        # List objects
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        matching_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if filter_pattern in filename:
                    matching_keys.append(key)

        return matching_keys
    except Exception as e:
        logger.error(f"Error listing datapoints: {e}")
        return []


# %%
# ============================================================================
# DATAPOINT GENERATION
# ============================================================================


def generate_all_datapoints(
    embedding_dimensions,
    random_seeds,
    max_features_fractions,
    parent_features_fractions,
    noise_levels,
    max_corpus_size,
    s3_available,
    skip_existing=True,
):
    """
    Generate datapoints for all parameter combinations and save to S3.

    Args:
        embedding_dimensions: List of embedding dimensions to test
        random_seeds: List of random seeds for reproducibility
        max_features_fractions: List of fractions of embedding dimension for max_features
        parent_features_fractions: List of fractions of max_features for parent features
        noise_levels: List of noise scales to apply
        max_corpus_size: Maximum number of samples in corpus
        s3_available: Whether S3 is accessible for saving
        skip_existing: If True, skip generating datapoints that already exist in S3

    Returns:
        List of S3 keys for generated datapoints
    """
    logger.info("=" * 80)
    logger.info("GENERATING DATAPOINTS FOR ALL CONDITIONS")
    logger.info("=" * 80)

    generated_keys = []

    for embedding_dim in tqdm(embedding_dimensions, desc="Embedding dimensions"):
        for seed in tqdm(random_seeds, desc="Random seeds", leave=False):
            for max_feat_frac in tqdm(max_features_fractions, desc="Max features fraction", leave=False):

                # Calculate max_features from fraction of embedding dimension
                MAX_FEATURES = int(embedding_dim * max_feat_frac)
                if MAX_FEATURES < 2.0:
                    logger.warning(f"Skipping: MAX_FEATURES={MAX_FEATURES} too small")
                    continue

                logger.info(f"\n{'='*80}")
                logger.info(
                    f"Configuration: d={embedding_dim}, seed={seed}, "
                    f"max_features={MAX_FEATURES} ({max_feat_frac:.1%} of d)"
                )
                logger.info(f"{'='*80}")

                # Initialize features generator with current parameters
                fg = FeaturesGenerator(max_features=MAX_FEATURES, d=embedding_dim, seed=seed)
                root_node = ClusterNode(depth=0, n_features=MAX_FEATURES)
                root_datapoints = fg.generate_datapoints_from_features(root_node, n_points=1, noise_scale=0.0)

                # Loop over parent features fractions
                for parent_feat_frac in tqdm(parent_features_fractions, desc="Parent features fraction", leave=False):

                    # Calculate N_parent_features from fraction of MAX_FEATURES
                    N_parent_features = int(MAX_FEATURES * parent_feat_frac)

                    # Skip if N_parent_features is too small
                    if N_parent_features < 2:
                        logger.warning(f"Skipping: N_parent_features={N_parent_features} too small")
                        continue

                    logger.info(
                        f"  Processing: N_parent_features={N_parent_features} "
                        f"({parent_feat_frac:.1%} of max_features)"
                    )

                    parent_node = ClusterNode(depth=1, n_features=N_parent_features)
                    children_nodes = [
                        ClusterNode(depth=2, n_features=nf, parent=parent_node)
                        for nf in np.arange(1, parent_node.n_features, 1)
                    ]
                    parent_node.children = children_nodes

                    # Loop over noise levels
                    for noise_idx, noise_scale in tqdm(enumerate(noise_levels), desc="Noise levels", leave=False):

                        noise_str = f"noise_{noise_scale:.2f}".replace(".", "_")
                        datapoints_filename = (
                            f"datapoints_d{embedding_dim}_seed{seed}_"
                            f"maxfeat{MAX_FEATURES}_parentfeat{N_parent_features}_{noise_str}"
                        )
                        datapoints_s3_key = f"homology_simulations/datapoints/{datapoints_filename}.json"

                        # Check if datapoints already exist
                        if skip_existing and s3_available:
                            try:
                                s3_client.head_object(Bucket=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY, Key=datapoints_s3_key)
                                logger.info(f"      ⊙ Datapoints already exist, skipping: {datapoints_s3_key}")
                                generated_keys.append(datapoints_s3_key)
                                continue
                            except:
                                pass  # Datapoints don't exist, generate them

                        # Generate parent datapoints
                        parent_datapoints = fg.generate_datapoints_from_features(
                            parent_node,
                            feature_ids=root_datapoints[0]["features_ids"],
                            n_points=max_corpus_size,
                            noise_scale=noise_scale,
                        )

                        # Generate children datapoints for each parent
                        all_children_dp = []
                        for p_dp in parent_datapoints:
                            children_datapoints = []
                            for node in children_nodes:
                                children_datapoints += fg.generate_datapoints_from_features(
                                    node,
                                    feature_ids=p_dp["features_ids"],
                                    n_points=max_corpus_size,
                                    noise_scale=noise_scale,
                                    type="child",
                                )
                            all_children_dp.append(children_datapoints)

                        # Save generated datapoints to S3 (if available)
                        if s3_available:
                            # Package datapoints with metadata
                            datapoints_package = {
                                "metadata": {
                                    "embedding_dim": embedding_dim,
                                    "random_seed": seed,
                                    "max_features": MAX_FEATURES,
                                    "max_features_fraction": max_feat_frac,
                                    "n_parent_features": N_parent_features,
                                    "parent_features_fraction": parent_feat_frac,
                                    "noise_scale": noise_scale,
                                    "max_corpus_size": max_corpus_size,
                                    "n_parent_datapoints": len(parent_datapoints),
                                    "n_children_groups": len(all_children_dp),
                                },
                                "parent_datapoints": parent_datapoints,
                                "all_children_datapoints": all_children_dp,
                                "root_feature_ids": root_datapoints[0]["features_ids"],
                            }

                            upload_success_dp = upload_datapoints_to_s3(datapoints_package, datapoints_s3_key)
                            if upload_success_dp:
                                logger.info(f"      ✓ Datapoints saved to S3: {datapoints_s3_key}")
                                generated_keys.append(datapoints_s3_key)

    logger.info("\n" + "=" * 80)
    logger.info(f"DATAPOINT GENERATION COMPLETE - {len(generated_keys)} datasets generated")
    logger.info("=" * 80)

    return generated_keys


# %%
# ============================================================================
# SCENARIO EXECUTION
# ============================================================================


def run_scenarios_on_datapoints(
    datapoints_keys,
    depsilon_values,
    s3_available,
    scenarios_to_run=None,
):
    """
    Run scenarios on pre-generated datapoints.

    Args:
        datapoints_keys: List of S3 keys pointing to datapoint files
        depsilon_values: List of depsilon values for homology computation
        s3_available: Whether S3 is accessible for saving results
        scenarios_to_run: List of scenario numbers to run (e.g., [1, 2, 3]).
                         If None, runs all available scenarios.

    Returns:
        Dictionary mapping scenario names to result DataFrames
    """
    logger.info("=" * 80)
    logger.info("RUNNING SCENARIOS ON GENERATED DATAPOINTS")
    logger.info("=" * 80)

    if scenarios_to_run is None:
        scenarios_to_run = [1, 2, 3]  # Default to all scenarios

    all_results = {f"scenario_{i}": [] for i in scenarios_to_run}

    for datapoints_key in tqdm(datapoints_keys, desc="Processing datapoint sets"):
        # Load datapoints from S3
        datapoints_package = load_datapoints_from_s3(datapoints_key)
        if datapoints_package is None:
            logger.error(f"Failed to load datapoints from {datapoints_key}")
            continue

        # Extract metadata and datapoints
        metadata = datapoints_package["metadata"]
        parent_datapoints = datapoints_package["parent_datapoints"]
        all_children_dp = datapoints_package["all_children_datapoints"]

        logger.info(
            f"\nProcessing: d={metadata['embedding_dim']}, seed={metadata['random_seed']}, "
            f"max_feat={metadata['max_features']}, parent_feat={metadata['n_parent_features']}, "
            f"noise={metadata['noise_scale']:.2f}"
        )

        # Compute non-orthogonal parents for scenario 3
        non_orthogonal_parents = [
            [j for j, p in enumerate(parent_datapoints) if set(p["features_ids"]).intersection(ptest["features_ids"])]
            for ptest in parent_datapoints
        ]

        # Run scenarios
        all_res_s1 = []
        all_res_s2 = []
        all_res_s3 = []
        all_res_s4 = []

        for i, p_dp in enumerate(parent_datapoints):
            # Scenario 1: Parent outliers to children
            if 1 in scenarios_to_run:
                res_s1 = ScenarioFactory.scenario_map[1](
                    [p_dp] + all_children_dp[i],
                    noise_scale=metadata["noise_scale"],
                    depsilon=depsilon_values,
                )
                all_res_s1.append(res_s1)

            # Scenario 2: Children to parent outliers
            if 2 in scenarios_to_run:
                res_s2 = ScenarioFactory.scenario_map[2](
                    parent_datapoints + all_children_dp[i],
                    noise_scale=metadata["noise_scale"],
                    depsilon=depsilon_values,
                )
                all_res_s2.append(res_s2)

            # Scenario 3: Variable overlap with uncorrelated nodes
            if 3 in scenarios_to_run and len(non_orthogonal_parents[i]) < 50:
                res_s3 = ScenarioFactory.scenario_map[3](
                    [p_dp] + list(chain.from_iterable(all_children_dp)),
                    noise_scale=metadata["noise_scale"],
                    depsilon=depsilon_values,
                )
                all_res_s3.append(res_s3)

            if 4 in scenarios_to_run:
                res_s4 = ScenarioFactory.scenario_map[4](
                    [p_dp] + list(chain.from_iterable(all_children_dp)),
                    noise_scale=metadata["noise_scale"],
                    depsilon=depsilon_values,
                )
                all_res_s4.append(res_s4)

        # Combine results for each scenario
        results_dfs = {}

        if 1 in scenarios_to_run and all_res_s1:
            results_dfs["scenario_1"] = pd.concat(all_res_s1)

        if 2 in scenarios_to_run and all_res_s2:
            results_dfs["scenario_2"] = pd.concat(all_res_s2)

        if 3 in scenarios_to_run and all_res_s3:
            try:
                results_dfs["scenario_3"] = pd.concat(all_res_s3)
            except ValueError:
                logger.warning("No valid results for scenario 3")

        if 4 in scenarios_to_run and all_res_s4:
            results_dfs["scenario_4"] = pd.concat(all_res_s4)

        # Add metadata columns to all results
        for scenario_key, results_df in results_dfs.items():
            for key, value in metadata.items():
                if key not in ["n_parent_datapoints", "n_children_groups", "max_corpus_size"]:
                    results_df[key] = value

        # Create filename base for saving
        noise_str = f"noise_{metadata['noise_scale']:.2f}".replace(".", "_")
        filename_base = (
            f"scenario{{scenario}}_d{metadata['embedding_dim']}_seed{metadata['random_seed']}_"
            f"maxfeat{metadata['max_features']}_parentfeat{metadata['n_parent_features']}_{noise_str}"
        )

        # Upload to S3 (if available)
        upload_successes = {}
        for scenario_num in scenarios_to_run:
            scenario_key = f"scenario_{scenario_num}"
            if scenario_key in results_dfs:
                if s3_available:
                    s3_key = f"homology_simulations/{filename_base.format(scenario=scenario_num)}.json"
                    upload_successes[scenario_num] = upload_dataframe_to_s3(results_dfs[scenario_key], s3_key)

                    if upload_successes.get(scenario_num):
                        logger.info(f"  ✓ Scenario {scenario_num} saved to S3")
                        all_results[scenario_key].append(results_dfs[scenario_key])

    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO EXECUTION COMPLETE")
    logger.info("=" * 80)

    return all_results


# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("PERSISTENT HOMOLOGY ANALYSIS ON RECURSIVE FEATURE-BASED CLUSTERS")
    logger.info("=" * 80)

    # Check S3 bucket accessibility
    logger.info(f"Checking S3 bucket: {S3_EXPERIMENT_RETRIEVAL_AMBIGUITY}")
    try:
        s3_client.head_bucket(Bucket=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY)
        logger.info("✓ S3 bucket accessible - results will be uploaded to S3")
        s3_available = True
    except Exception as e:
        logger.warning(f"S3 bucket not accessible ({e})")
        logger.warning("Results will be saved locally only (CSV files)")
        s3_available = False

    # %%
    # Global parameters
    UNCORRELATED_FRACTIONS = [0.0]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.9]
    N_QUERIES_PER_GROUP = 20
    MIN_DEPTH = 1
    MAX_CORPUS_SIZE = 50  # Maximum 50 samples in corpus
    DEPSILON_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0]

    # New parameter sweeps
    EMBEDDING_DIMENSIONS = [128]  # [64, 256]
    RANDOM_SEEDS = [42]
    MAX_FEATURES_FRACTIONS = sorted([0.25, 0.125, 0.0625, 0.03125, 0.015625])  # Fraction of embedding dimension
    PARENT_FEATURES_FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]  # Fraction of max_features

    # %%
    # Step 1: Generate all datapoints
    generated_keys = generate_all_datapoints(
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        random_seeds=RANDOM_SEEDS,
        max_features_fractions=MAX_FEATURES_FRACTIONS,
        parent_features_fractions=PARENT_FEATURES_FRACTIONS,
        noise_levels=NOISE_LEVELS,
        max_corpus_size=MAX_CORPUS_SIZE,
        s3_available=s3_available,
        skip_existing=True,  # Set to False to regenerate all datapoints
    )

    # %%
    # Step 2: Run scenarios on generated datapoints
    # You can now run specific scenarios without regenerating datapoints
    results = run_scenarios_on_datapoints(
        datapoints_keys=generated_keys,
        depsilon_values=DEPSILON_VALUES,
        s3_available=s3_available,
        scenarios_to_run=[1, 2, 3, 4],  # Specify which scenarios to run
    )

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE - ALL SCENARIOS PROCESSED")
    logger.info("=" * 80)

# %%
