"""
Utility functions for homology experiments.

This module contains common utilities for:
- Loading data from S3
- S3 file operations
- Data processing helpers
- Histogram creation
- Metadata parsing
"""

import itertools
import json
import math
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from common_tools.connections import get_boto_session

from config import AWS_PROFILE_NAME, S3_EXPERIMENT_RETRIEVAL_AMBIGUITY

# ============================================================================
# S3 CLIENT INITIALIZATION
# ============================================================================

def get_s3_client():
    """Get initialized S3 client."""
    return get_boto_session(profile_name=AWS_PROFILE_NAME).client("s3")


# ============================================================================
# S3 DATA LOADING FUNCTIONS
# ============================================================================

def list_available_s3_files(
    scenario_num=None,
    embedding_dim=None,
    random_seed=None,
    max_features=None,
    n_parent_features=None,
    noise_scale=None,
    bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
):
    """
    List available homology results files in S3 with optional filtering.

    Args:
        scenario_num: Scenario number (1, 2, or 3) to filter by
        embedding_dim: Embedding dimension to filter by
        random_seed: Random seed to filter by
        max_features: Max features to filter by
        n_parent_features: Number of parent features to filter by
        noise_scale: Noise scale to filter by
        bucket_name: S3 bucket name

    Returns:
        List of matching S3 keys
    """
    s3_client = get_s3_client()

    try:
        prefix = "homology_simulations/"

        # Build filter pattern
        pattern_parts = []
        if scenario_num is not None:
            pattern_parts.append(f"scenario{scenario_num}_")
        else:
            pattern_parts.append("scenario")

        if embedding_dim is not None:
            pattern_parts.append(f"d{embedding_dim}_")
        if random_seed is not None:
            pattern_parts.append(f"seed{random_seed}_")
        if max_features is not None:
            pattern_parts.append(f"maxfeat{max_features}_")
        if n_parent_features is not None:
            pattern_parts.append(f"parentfeat{n_parent_features}_")
        if noise_scale is not None:
            noise_str = f"noise_{noise_scale:.2f}".replace(".", "_")
            pattern_parts.append(noise_str)

        # List objects
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        matching_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]

                # Skip datapoints directory
                if "datapoints/" in key:
                    continue

                # Check if all pattern parts match
                matches_all = all(part in filename for part in pattern_parts)

                if matches_all and filename.endswith(".json"):
                    matching_keys.append(key)

        return sorted(matching_keys)
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []


def list_available_s3_files_projection(
    scenario_num=None,
    bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
    prefix="homology_simulations/",
):
    """
    List available projection scenario results files in S3.

    Args:
        scenario_num: Scenario number (1, 2, or 3) to filter by
        bucket_name: S3 bucket name
        prefix: S3 prefix for projection files (default: "homology_simulations/")

    Returns:
        List of matching S3 keys
    """
    s3_client = get_s3_client()

    try:
        # List objects
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        matching_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]

                # Check if it's a projection file
                if not filename.startswith("combined_scenario") or not filename.endswith("_with_projections.json"):
                    continue

                # If scenario_num specified, filter by it
                if scenario_num is not None:
                    expected_filename = f"combined_scenario{scenario_num}_with_projections.json"
                    if filename != expected_filename:
                        continue

                matching_keys.append(key)

        return sorted(matching_keys)
    except Exception as e:
        print(f"Error listing S3 projection files: {e}")
        return []


def parse_metadata_from_filename(filename):
    """
    Extract metadata from filename.

    Format: scenario{X}_d{dim}_seed{seed}_maxfeat{N}_parentfeat{M}_noise_{X_XX}.json

    Args:
        filename: S3 key or filename

    Returns:
        Dictionary with metadata
    """
    basename = filename.split("/")[-1].replace(".json", "")
    metadata = {}

    # Extract scenario
    if "scenario" in basename:
        scenario_part = basename.split("scenario")[1].split("_")[0]
        metadata["scenario"] = int(scenario_part)

    # Extract embedding dimension
    if "_d" in basename:
        dim_part = basename.split("_d")[1].split("_")[0]
        metadata["embedding_dim"] = int(dim_part)

    # Extract seed
    if "seed" in basename:
        seed_part = basename.split("seed")[1].split("_")[0]
        metadata["random_seed"] = int(seed_part)

    # Extract max_features
    if "maxfeat" in basename:
        maxfeat_part = basename.split("maxfeat")[1].split("_")[0]
        metadata["max_features"] = int(maxfeat_part)

    # Extract n_parent_features
    if "parentfeat" in basename:
        parentfeat_part = basename.split("parentfeat")[1].split("_")[0]
        metadata["n_parent_features"] = int(parentfeat_part)

    # Extract noise_scale
    if "noise_" in basename:
        noise_part = basename.split("noise_")[1]
        noise_str = noise_part.replace("_", ".")
        metadata["noise_scale"] = float(noise_str)

    return metadata


def load_scenario_data_from_s3(
    scenario_num,
    n_parent_features=None,
    embedding_dim=None,
    random_seed=None,
    max_features=None,
    noise_scale=None,
    bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
    with_dist=False
):
    """
    Load scenario data from S3 JSON files across different noise levels.
    Can filter by specific metadata values if provided.

    Args:
        scenario_num: Scenario number (1, 2, or 3)
        n_parent_features: Number of parent features (optional, for filtering)
        embedding_dim: Embedding dimension (optional, for filtering)
        random_seed: Random seed (optional, for filtering)
        max_features: Max features (optional, for filtering)
        noise_scale: Specific noise scale (optional, for filtering)
        bucket_name: S3 bucket name
        with_dist: If True, keep h0_dist and h1_dist columns; if False, drop them

    Returns:
        Combined DataFrame with all matching files
    """
    from tqdm import tqdm

    s3_client = get_s3_client()

    # List matching files
    matching_keys = list_available_s3_files(
        scenario_num=scenario_num,
        embedding_dim=embedding_dim,
        random_seed=random_seed,
        max_features=max_features,
        n_parent_features=n_parent_features,
        noise_scale=noise_scale,
        bucket_name=bucket_name,
    )

    if not matching_keys:
        print(f"Warning: No S3 files found for scenario {scenario_num} with given filters")
        return None

    print(f"Found {len(matching_keys)} S3 files for scenario {scenario_num}")

    # Download and parse each file
    dfs = []
    for key in tqdm(matching_keys):
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            json_data = response["Body"].read().decode("utf-8")
            data = json.loads(json_data)
            df = pd.DataFrame(data)
            if with_dist is False and "h0_dist" in df.columns and "h1_dist" in df.columns:
                df = df.drop(columns=["h0_dist", "h1_dist"])

            dfs.append(df)
        except Exception as e:
            print(f"Error loading {key}: {e}")
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    print(
        f"Scenario {scenario_num}: Loaded {len(combined)} rows "
        f"with noise levels: {sorted(combined['noise_scale'].unique())}"
    )

    # Print available metadata
    metadata_cols = [
        "embedding_dim",
        "random_seed",
        "max_features",
        "n_parent_features",
        "noise_scale",
    ]
    for col in metadata_cols:
        if col in combined.columns and len(combined[col].unique()) > 1:
            print(f"  {col}: {sorted(combined[col].unique())}")

    return combined


def load_scenario_data_from_s3_projection(
    scenario_num,
    bucket_name=S3_EXPERIMENT_RETRIEVAL_AMBIGUITY,
    prefix="homology_simulations/",
    with_dist=False
):
    """
    Load projection scenario data from S3 JSON files.

    Args:
        scenario_num: Scenario number (1, 2, or 3)
        bucket_name: S3 bucket name
        prefix: S3 prefix for projection files (default: "homology_simulations/")
        with_dist: If True, keep h0_dist and h1_dist columns; if False, drop them

    Returns:
        DataFrame with projection data, or None if file not found
    """
    s3_client = get_s3_client()

    # List matching files
    matching_keys = list_available_s3_files_projection(
        scenario_num=scenario_num,
        bucket_name=bucket_name,
        prefix=prefix,
    )

    if not matching_keys:
        print(f"Warning: No S3 projection file found for scenario {scenario_num}")
        return None

    if len(matching_keys) > 1:
        print(f"Warning: Multiple projection files found for scenario {scenario_num}, using first one")

    key = matching_keys[0]
    print(f"Loading projection data from: {key}")

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        json_data = response["Body"].read().decode("utf-8")
        data = json.loads(json_data)
        df = pd.DataFrame(data)

        if with_dist is False and "h0_dist" in df.columns and "h1_dist" in df.columns:
            df = df.drop(columns=["h0_dist", "h1_dist"])

        print(f"Scenario {scenario_num} (projection): Loaded {len(df)} rows")

        # Print available columns
        print(f"  Available columns: {list(df.columns)}")

        # Print some statistics about the data
        if "noise_scale" in df.columns:
            print(f"  Noise levels: {sorted(df['noise_scale'].unique())}")
        if "depsilon" in df.columns:
            print(f"  Depsilon values: {sorted(df['depsilon'].unique())}")
        if "dim_ratio" in df.columns:
            print(f"  Dim ratio range: [{df['dim_ratio'].min():.2f}, {df['dim_ratio'].max():.2f}]")

        return df

    except Exception as e:
        print(f"Error loading projection data from {key}: {e}")
        return None


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
    s3_client = get_s3_client()

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
        print(f"Error uploading to S3: {e}")
        return False


# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def create_histogram(df, bins=None):
    """
    Create normalized histogram from h0_dist data.

    Args:
        df: DataFrame with h0_dist column
        bins: Histogram bins (default: np.arange(0, 2, 0.05))

    Returns:
        Normalized histogram array
    """
    if bins is None:
        bins = np.arange(0, 2, 0.05)

    h0s = list(itertools.chain.from_iterable([a for a in df["h0_dist"].dropna()]))
    hist, _ = np.histogram(h0s, bins=bins)
    return hist / max(hist) if max(hist) > 0 else hist


def create_normalized_histogram(df, bins=None):
    """
    Create normalized histogram from h0_dist data (sum = 1).

    Args:
        df: DataFrame with h0_dist column
        bins: Histogram bins (default: np.arange(0, 2, 0.05))

    Returns:
        Normalized histogram array (sum = 1)
    """
    if bins is None:
        bins = np.arange(0, 2, 0.05)

    h0s = list(itertools.chain.from_iterable([a for a in df["h0_dist"].dropna()]))
    hist, _ = np.histogram(h0s, bins=bins)
    return hist / np.sum(hist) if np.sum(hist) > 0 else hist


def zero_overlap_probability(n, k):
    """
    Calculate probability of zero overlap between two k-sized subsets of n items.

    Args:
        n: Total number of items
        k: Subset size

    Returns:
        Probability of zero overlap
    """
    subset = n - k
    if subset < 2 * k:
        return 0.0

    proba = (math.factorial(subset) / (math.factorial(subset - k) * math.factorial(k))) / (
        math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
    )
    return proba


def combinations(n, k):
    """
    Calculate binomial coefficient (n choose k).

    Args:
        n: Total number of items
        k: Number of items to choose

    Returns:
        Number of combinations
    """
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
