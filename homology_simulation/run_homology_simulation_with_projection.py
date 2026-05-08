# %%
"""
Random Projection Homology Analysis on Recursive Feature-Based Clusters

This script extends the base homology analysis to study how dimensionality reduction
via random projection affects persistent homology features in hierarchical cluster embeddings.

Key differences from base script:
- Applies GaussianRandomProjection to reduce 256D embeddings to [224, 192, 128, 64]D
- Only runs scenarios 1 (parent→children) and 2 (child→parent)
- Fixed configuration: 256D, 64 max features, 32 parent features
- Noise levels: [0.01, 0.05, 0.1]
- Tracks projection dimension as metadata for comparison

Purpose:
- Test whether topological features are preserved under dimensionality reduction
- Analyze sensitivity of homology metrics to embedding dimension
- Study scenarios with controlled, focused configuration
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from core.connections import get_boto_session
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm

# Import reusable functions from original script
from .run_homology_simulation import (
    generate_all_datapoints,
    load_datapoints_from_s3,
    upload_dataframe_to_s3,
    upload_datapoints_to_s3,
)
from .simulation_scenarios import ScenarioFactory

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.common_tools import EC2Logger

from config import AWS_PROFILE_NAME, S3_EXPERIMENT_RETRIEVAL_AMBIGUITY, pre

# Initialize S3 client and logger
s3_client = get_boto_session(profile_name=AWS_PROFILE_NAME).client("s3")
logger = EC2Logger(
    pre("HomologySimulationsBoostResearchMachine") + "Logs",
    "homology_projection_simulation_logstream"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fixed configuration for focused experiment
EMBEDDING_DIMENSIONS = [256]  # Original dimension
MAX_FEATURES = 64
PARENT_FEATURES = 32
NOISE_LEVELS = [0.01, 0.05, 0.1]
PROJECTION_DIMENSIONS = [224, 192, 128, 64]  # Target dimensions for random projection
RANDOM_SEED = 42
MAX_CORPUS_SIZE = 50
DEPSILON_VALUES = [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0]

# Scenarios to run (only 1 and 2 for focused analysis)
SCENARIOS_TO_RUN = [1, 2]


# ============================================================================
# RANDOM PROJECTION FUNCTIONS
# ============================================================================


def apply_random_projection(datapoints, target_dim, random_seed):
    """
    Apply Gaussian random projection to reduce embedding dimensionality.

    Uses sklearn's GaussianRandomProjection to project embeddings from original
    dimension (256D) to target dimension while approximately preserving distances.

    Args:
        datapoints: List of datapoint dicts with "X" field (numpy arrays)
        target_dim: Target embedding dimension after projection
        random_seed: Random seed for reproducible projections

    Returns:
        List of datapoint dicts with projected "X" field and projection metadata
    """
    logger.info(f"  Applying random projection to {target_dim}D...")

    # Extract all embeddings into a matrix (n_points, original_dim)
    embeddings = np.array([dp["X"] for dp in datapoints])
    original_dim = embeddings.shape[1]

    logger.info(f"    Original shape: {embeddings.shape}")

    # Apply Gaussian random projection
    projector = GaussianRandomProjection(
        n_components=target_dim,
        random_state=random_seed
    )
    projected_embeddings = projector.fit_transform(embeddings)

    logger.info(f"    Projected shape: {projected_embeddings.shape}")

    # Reconstruct datapoint dicts with projected embeddings
    projected_datapoints = []
    for i, dp in enumerate(datapoints):
        # Create new datapoint dict with projected embedding
        projected_dp = dp.copy()
        projected_dp["X"] = projected_embeddings[i]
        projected_dp["original_embedding_dim"] = original_dim
        projected_dp["projection_dim"] = target_dim
        projected_datapoints.append(projected_dp)

    return projected_datapoints


def project_and_save_datapoints(datapoints_keys, projection_dims, random_seed, s3_available):
    """
    Batch process datapoint files through random projection pipeline.

    For each original datapoint file and each projection dimension:
    - Check if projected version already exists (skip if yes)
    - Load original datapoints
    - Apply random projection
    - Save to S3 with projection metadata in key

    Args:
        datapoints_keys: List of S3 keys pointing to original datapoint files
        projection_dims: List of target dimensions for projection
        random_seed: Random seed for reproducible projections
        s3_available: Whether S3 is accessible for saving

    Returns:
        Dictionary mapping projection_dim -> list of projected datapoint S3 keys
    """
    logger.info("=" * 80)
    logger.info("PROJECTING DATAPOINTS TO LOWER DIMENSIONS")
    logger.info("=" * 80)

    projected_keys = {dim: [] for dim in projection_dims}
    projected_keys["original"] = datapoints_keys.copy()  # Track original keys too

    for datapoints_key in tqdm(datapoints_keys, desc="Processing datapoint files"):
        # Load original datapoints
        logger.info(f"\nLoading original datapoints: {datapoints_key}")
        datapoints_package = load_datapoints_from_s3(datapoints_key)

        if datapoints_package is None:
            logger.error(f"Failed to load datapoints from {datapoints_key}")
            continue

        # Extract metadata and datapoints
        metadata = datapoints_package["metadata"]
        parent_datapoints = datapoints_package["parent_datapoints"]
        all_children_datapoints = datapoints_package["all_children_datapoints"]

        logger.info(
            f"Loaded: d={metadata['embedding_dim']}, seed={metadata['random_seed']}, "
            f"noise={metadata['noise_scale']:.2f}, {len(parent_datapoints)} parents"
        )

        # Project to each target dimension
        for target_dim in tqdm(projection_dims, desc="Projection dimensions", leave=False):
            # Build projected filename
            original_filename = datapoints_key.split("/")[-1].replace(".json", "")
            projected_filename = original_filename.replace(
                f"d{metadata['embedding_dim']}",
                f"d{metadata['embedding_dim']}_proj{target_dim}"
            )
            projected_s3_key = f"homology_simulations/projected_datapoints/proj{target_dim}/{projected_filename}.json"

            # Check if already exists
            if s3_available:
                try:
                    s3_client.head_object(
                        Bucket=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
                        Key=projected_s3_key
                    )
                    logger.info(f"  ⊙ Projected datapoints already exist, skipping: {projected_s3_key}")
                    projected_keys[target_dim].append(projected_s3_key)
                    continue
                except:
                    pass  # Doesn't exist, proceed with projection

            # Apply projection to parent datapoints
            projected_parents = apply_random_projection(
                parent_datapoints,
                target_dim,
                random_seed
            )

            # Apply projection to all children groups
            projected_children_groups = []
            for children_group in all_children_datapoints:
                projected_children = apply_random_projection(
                    children_group,
                    target_dim,
                    random_seed
                )
                projected_children_groups.append(projected_children)

            # Package projected datapoints with updated metadata
            projected_package = {
                "metadata": metadata.copy(),
                "parent_datapoints": projected_parents,
                "all_children_datapoints": projected_children_groups,
                "root_feature_ids": datapoints_package["root_feature_ids"],
            }

            # Add projection metadata
            projected_package["metadata"]["original_embedding_dim"] = metadata["embedding_dim"]
            projected_package["metadata"]["projection_dim"] = target_dim
            projected_package["metadata"]["projection_method"] = "GaussianRandomProjection"
            projected_package["metadata"]["projection_random_seed"] = random_seed

            # Save to S3
            if s3_available:
                upload_success = upload_datapoints_to_s3(projected_package, projected_s3_key)
                if upload_success:
                    logger.info(f"  ✓ Projected datapoints saved to S3: {projected_s3_key}")
                    projected_keys[target_dim].append(projected_s3_key)

    logger.info("\n" + "=" * 80)
    logger.info("PROJECTION COMPLETE")
    logger.info("=" * 80)

    return projected_keys


def run_scenarios_on_projections(
    original_keys,
    projection_dims,
    scenarios,
    depsilon_values,
    s3_available
):
    """
    Run scenarios on original and all projected datapoint variants.

    Executes specified scenarios on:
    - Original 256D datapoints (projection_dim = "original")
    - Each projected dimension variant (224D, 192D, 128D, 64D)

    Adds "projection_dim" column to results for comparison analysis.

    Args:
        original_keys: Dictionary mapping projection_dim -> list of datapoint S3 keys
                      Must include "original" key for non-projected data
        projection_dims: List of projection dimensions (e.g., [224, 192, 128, 64])
        scenarios: List of scenario numbers to run (e.g., [1, 2])
        depsilon_values: List of epsilon thresholds for homology computation
        s3_available: Whether S3 is accessible for saving results

    Returns:
        Dictionary mapping scenario numbers to combined result DataFrames
    """
    logger.info("=" * 80)
    logger.info("RUNNING SCENARIOS ON ORIGINAL AND PROJECTED DATAPOINTS")
    logger.info("=" * 80)

    all_results = {scenario_num: [] for scenario_num in scenarios}

    # Process all projection variants (including original)
    variants_to_process = ["original"] + projection_dims

    for variant in variants_to_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing variant: {variant}")
        logger.info(f"{'='*80}")

        # Get datapoint keys for this variant
        datapoints_keys = original_keys.get(variant, [])

        if not datapoints_keys:
            logger.warning(f"No datapoints found for variant {variant}")
            continue

        # Process each datapoint file
        for datapoints_key in tqdm(datapoints_keys, desc=f"Datapoints ({variant})"):
            # Load datapoints
            datapoints_package = load_datapoints_from_s3(datapoints_key)

            if datapoints_package is None:
                logger.error(f"Failed to load datapoints from {datapoints_key}")
                continue

            # Extract metadata and datapoints
            metadata = datapoints_package["metadata"]
            parent_datapoints = datapoints_package["parent_datapoints"]
            all_children_dp = datapoints_package["all_children_datapoints"]

            logger.info(
                f"\nProcessing: variant={variant}, "
                f"d={metadata.get('original_embedding_dim', metadata['embedding_dim'])}, "
                f"seed={metadata['random_seed']}, noise={metadata['noise_scale']:.2f}"
            )

            # Run scenarios
            scenario_results = {scenario_num: [] for scenario_num in scenarios}

            for i, p_dp in enumerate(parent_datapoints):
                # Scenario 1: Parent outliers to children
                if 1 in scenarios:
                    res_s1 = ScenarioFactory.scenario_map[1](
                        [p_dp] + all_children_dp[i],
                        noise_scale=metadata["noise_scale"],
                        depsilon=depsilon_values,
                    )
                    scenario_results[1].append(res_s1)

                # Scenario 2: Children to parent outliers
                if 2 in scenarios:
                    res_s2 = ScenarioFactory.scenario_map[2](
                        parent_datapoints + all_children_dp[i],
                        noise_scale=metadata["noise_scale"],
                        depsilon=depsilon_values,
                    )
                    scenario_results[2].append(res_s2)

            # Combine results for each scenario
            for scenario_num in scenarios:
                if scenario_results[scenario_num]:
                    results_df = pd.concat(scenario_results[scenario_num])

                    # Add metadata columns
                    for key, value in metadata.items():
                        if key not in ["n_parent_datapoints", "n_children_groups", "max_corpus_size"]:
                            results_df[key] = value

                    # Add projection dimension column (key for comparison)
                    results_df["projection_dim"] = variant

                    # Create filename for saving
                    noise_str = f"noise_{metadata['noise_scale']:.2f}".replace(".", "_")
                    original_dim = metadata.get('original_embedding_dim', metadata['embedding_dim'])

                    if variant == "original":
                        filename = (
                            f"scenario{scenario_num}_d{original_dim}_seed{metadata['random_seed']}_"
                            f"maxfeat{metadata['max_features']}_parentfeat{metadata['n_parent_features']}_{noise_str}"
                        )
                    else:
                        filename = (
                            f"scenario{scenario_num}_proj{variant}_d{original_dim}_seed{metadata['random_seed']}_"
                            f"maxfeat{metadata['max_features']}_parentfeat{metadata['n_parent_features']}_{noise_str}"
                        )

                    s3_key = f"homology_simulations/{filename}.json"

                    # Save to S3
                    if s3_available:
                        upload_success = upload_dataframe_to_s3(results_df, s3_key)
                        if upload_success:
                            logger.info(f"  ✓ Scenario {scenario_num} ({variant}) saved to S3")

                    # Collect for aggregation
                    all_results[scenario_num].append(results_df)

    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO EXECUTION COMPLETE")
    logger.info("=" * 80)

    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("RANDOM PROJECTION HOMOLOGY ANALYSIS")
    logger.info("=" * 80)

    # Check S3 bucket accessibility
    logger.info(f"Checking S3 bucket: {S3_EXPERIMENT_RETRIEVAL_AMBIGUITY}")
    try:
        s3_client.head_bucket(Bucket=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY)
        logger.info("✓ S3 bucket accessible - results will be uploaded to S3")
        s3_available = True
    except Exception as e:
        logger.warning(f"S3 bucket not accessible ({e})")
        logger.warning("Results will be saved locally only")
        s3_available = False

    # ========================================================================
    # STEP 1: Generate or load original 256D datapoints
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: GENERATE/LOAD ORIGINAL 256D DATAPOINTS")
    logger.info("=" * 80)

    # Calculate fractions from fixed values
    max_features_fraction = MAX_FEATURES / EMBEDDING_DIMENSIONS[0]  # 64/256 = 0.25
    parent_features_fraction = PARENT_FEATURES / MAX_FEATURES  # 32/64 = 0.5

    logger.info(f"Configuration:")
    logger.info(f"  Embedding dimension: {EMBEDDING_DIMENSIONS[0]}")
    logger.info(f"  Max features: {MAX_FEATURES} ({max_features_fraction:.1%} of d)")
    logger.info(f"  Parent features: {PARENT_FEATURES} ({parent_features_fraction:.1%} of max_features)")
    logger.info(f"  Noise levels: {NOISE_LEVELS}")
    logger.info(f"  Random seed: {RANDOM_SEED}")

    original_datapoints_keys = generate_all_datapoints(
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        random_seeds=[RANDOM_SEED],
        max_features_fractions=[max_features_fraction],
        parent_features_fractions=[parent_features_fraction],
        noise_levels=NOISE_LEVELS,
        max_corpus_size=MAX_CORPUS_SIZE,
        s3_available=s3_available,
        skip_existing=True,
    )

    logger.info(f"\n✓ Generated/loaded {len(original_datapoints_keys)} original datapoint files")

    # ========================================================================
    # STEP 2: Apply random projections
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: APPLY RANDOM PROJECTIONS")
    logger.info("=" * 80)
    logger.info(f"Projection dimensions: {PROJECTION_DIMENSIONS}")

    all_datapoints_keys = project_and_save_datapoints(
        datapoints_keys=original_datapoints_keys,
        projection_dims=PROJECTION_DIMENSIONS,
        random_seed=RANDOM_SEED,
        s3_available=s3_available,
    )

    # ========================================================================
    # STEP 3: Run scenarios on all variants
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: RUN SCENARIOS ON ALL PROJECTION VARIANTS")
    logger.info("=" * 80)
    logger.info(f"Scenarios: {SCENARIOS_TO_RUN}")
    logger.info(f"Variants: original + {PROJECTION_DIMENSIONS}")

    scenario_results = run_scenarios_on_projections(
        original_keys=all_datapoints_keys,
        projection_dims=PROJECTION_DIMENSIONS,
        scenarios=SCENARIOS_TO_RUN,
        depsilon_values=DEPSILON_VALUES,
        s3_available=s3_available,
    )

    # ========================================================================
    # STEP 4: Aggregate results for comparison
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: AGGREGATE RESULTS FOR COMPARISON")
    logger.info("=" * 80)

    for scenario_num in SCENARIOS_TO_RUN:
        if scenario_results[scenario_num]:
            # Combine all results for this scenario
            combined_df = pd.concat(scenario_results[scenario_num], ignore_index=True)

            logger.info(f"\nScenario {scenario_num}:")
            logger.info(f"  Total rows: {len(combined_df)}")
            logger.info(f"  Projection variants: {combined_df['projection_dim'].unique()}")

            # Save combined results
            if s3_available:
                combined_s3_key = f"homology_simulations/combined_scenario{scenario_num}_with_projections.json"
                upload_success = upload_dataframe_to_s3(combined_df, combined_s3_key)
                if upload_success:
                    logger.info(f"  ✓ Combined results saved: {combined_s3_key}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE - ALL SCENARIOS AND PROJECTIONS PROCESSED")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to S3 bucket: {S3_EXPERIMENT_RETRIEVAL_AMBIGUITY}")
    logger.info(f"  - Original datapoints: homology_simulations/datapoints/")
    logger.info(f"  - Projected datapoints: homology_simulations/projected_datapoints/proj{{dim}}/")
    logger.info(f"  - Scenario results: homology_simulations/scenario{{num}}_*.json")
    logger.info(f"  - Combined results: homology_simulations/combined_scenario{{num}}_with_projections.json")

# %%
