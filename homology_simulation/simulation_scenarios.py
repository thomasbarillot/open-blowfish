# %%
"""
Cluster Scenario Implementations for Persistent Homology Analysis

This module implements scenarios for testing persistent homology features
on hierarchical clusters. Each scenario takes pre-generated cluster data (X, labels_df)
and samples query and corpus sets for homology calculation.

Scenarios:
1. Query from parent outliers, corpus from children datapoints
2. Query from node datapoints, corpus from parent outliers
3. Query from child datapoint, corpus from children with varying feature overlap
4. Query from node datapoints, corpus from uncorrelated node (same depth)
5. Query from node datapoints, corpus from children + uncorrelated nodes
6. Query from node datapoints, corpus from parent + same-depth uncorrelated parents
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_bank_generator import ClusterNode, FeaturesGenerator
from .homology_calculation import calculate_homology_for_query_corpus


def scenario_1_parent_to_children(datapoints, noise_scale: float, depsilon: list):
    """
    Scenario 1: Query from parent outliers, corpus from children datapoints.

    For each parent node that has outliers and children:
    - Sample queries from parent's outliers
    - Corpus = all datapoints from direct children (non-outliers)
    - Calculate homology and track children's active dimensions

    Args:
        depsilon_values: List of epsilon thresholds to scan
        min_depth: Minimum depth for parent nodes to analyze (excludes shallow levels)

    Returns:
        results_df: DataFrame with homology features and n_active_dims
    """

    labels_df = pd.DataFrame(datapoints)

    # Find all parent nodes that have both outliers and children, at depth >= min_depth
    parent_candidates = labels_df[labels_df["type"] == "parent"]

    results = []
    children_nfeature_groups = set(labels_df.n_features)
    for _, parent in tqdm(parent_candidates.iterrows(), desc="Processing parent nodes"):
        # Get parent outliers
        for nfeat in children_nfeature_groups:

            # Find children of this parent
            children_df = labels_df[(labels_df["type"] != "parent") & (labels_df["n_features"] == nfeat)]

            if len(children_df) == 0:
                continue

            corpus_embeds = np.array([list(x) for x in children_df.X])
            query_embed = np.array(parent.X)
            print(f"{query_embed.shape=}", f"{corpus_embeds.shape=}")

            for de in depsilon:
                homology_features = calculate_homology_for_query_corpus(query_embed, corpus_embeds, depsilon=de)

                # Add metadata
                homology_features["query_active_dims"] = parent.n_features
                homology_features["corpus_active_dims"] = nfeat
                homology_features["dim_ratio"] = parent.n_features / nfeat
                homology_features["n_corpus_points"] = len(corpus_embeds)
                homology_features["depsilon"] = de
                homology_features["scenario"] = "parent_to_children"

                results.append(homology_features)

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df)} query-corpus pairs")
    # print(f"Parent nodes analyzed: {len(parent_ids)}")

    return results_df


def scenario_2_child_to_parent(datapoints, noise_scale: float, depsilon: list):
    """
    Scenario 2: Query from child node datapoints, corpus from parent outliers.

    For each child node:
    - Sample queries from child's datapoints (non-outliers)
    - Corpus = outliers from parent node
    - Calculate homology and track child's active dimensions

    Args:
        depsilon_values: List of epsilon thresholds to scan
        min_depth: Minimum depth for child nodes to analyze (excludes shallow levels)

    Returns:
        results_df: DataFrame with homology features and n_active_dims
    """

    results = []
    labels_df = pd.DataFrame(datapoints)

    # Find all parent nodes that have both outliers and children, at depth >= min_depth
    parents = labels_df[labels_df["type"] == "parent"]
    children = labels_df[labels_df["type"] != "parent"]

    children_nfeature_groups = set(children.n_features)
    for nfeat in children_nfeature_groups:

        if len(parents) == 0:
            continue

        children_df = children[labels_df["n_features"] == nfeat]
        query_embed = np.array(children_df.iloc[0].X)

        corpus_embeds = np.array([list(x) for x in parents.X])

        for de in depsilon:
            homology_features = calculate_homology_for_query_corpus(query_embed, corpus_embeds, depsilon=de)
            print(homology_features)
            # Add metadata
            homology_features["query_active_dims"] = nfeat
            homology_features["corpus_active_dims"] = parents.iloc[0].n_features
            homology_features["dim_ratio"] = nfeat / parents.iloc[0].n_features
            homology_features["n_corpus_points"] = len(corpus_embeds)
            homology_features["depsilon"] = de
            homology_features["scenario"] = "children_to_parent"

            results.append(homology_features)

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df)} query-corpus pairs")
    # print(f"Parent nodes analyzed: {len(parent_ids)}")

    return results_df


def scenario_3_parent_to_child_no_overlap(
    datapoints,
    noise_scale: float,
    depsilon: list,
):
    """
    Scenario 3: Query from child datapoint, corpus from children with
    varying feature overlap.

    For each child node:
    - Use one child datapoint as query
    - Build corpus subsets of other children grouped by number of shared features
    - Calculate homology for each overlap level
    - Track feature overlap count as key metadata

    This scenario tests how homology features correlate with feature similarity.

    Args:
        datapoints: List of generated datapoints with features_ids
        noise_scale: Noise scale used in generation (for metadata)
        depsilon: List of epsilon thresholds to scan
        min_corpus_size: Minimum number of points required for a corpus subset

    Returns:
        results_df: DataFrame with homology features and feature overlap metadata
    """

    labels_df = pd.DataFrame(datapoints)

    # Find all parent nodes that have both outliers and children, at depth >= min_depth
    parent_candidates = labels_df[labels_df["type"] == "parent"]

    results = []
    children_nfeature_groups = set(labels_df.n_features)
    for _, parent in tqdm(parent_candidates.iterrows(), desc="Processing parent nodes"):
        # Get parent outliers
        for nfeat in children_nfeature_groups:

            # Find children of this parent
            children_df = labels_df[(labels_df["type"] != "parent") & (labels_df["n_features"] == nfeat)]

            random_direct_children = np.random.randint(10, 40)
            

            direct_children = children_df.loc[
                children_df["features_ids"].apply(lambda x: len(set(x).difference(parent.features_ids)) == 0)
            ]
            if len(direct_children) == 0:
                continue

            print(f"{nfeat=}; {len(direct_children)=},{random_direct_children=}")
            direct_children = direct_children.sample(n=min(len(direct_children), random_direct_children))

            uncorrelated_children = children_df.loc[
                children_df["features_ids"].apply(lambda x: len(set(x).difference(parent.features_ids)) == len(x))
            ]
            if len(uncorrelated_children) == 0:
                continue
            if len(uncorrelated_children) < (50 - min(len(direct_children), random_direct_children)):
                continue
            uncorrelated_children = uncorrelated_children.sample(
                n=50 - min(len(direct_children), random_direct_children)
            )

            sampled_children = pd.concat([direct_children, uncorrelated_children])

            if len(children_df) == 0:
                continue

            corpus_embeds = np.array([list(x) for x in sampled_children.X])
            query_embed = np.array(parent.X)
            print(f"{query_embed.shape=}", f"{corpus_embeds.shape=}")

            for de in depsilon:
                homology_features = calculate_homology_for_query_corpus(query_embed, corpus_embeds, depsilon=de)
                # Add metadata
                homology_features["query_active_dims"] = parent.n_features
                homology_features["corpus_active_dims"] = nfeat
                homology_features["dim_ratio"] = parent.n_features / nfeat
                homology_features["n_corpus_points"] = len(corpus_embeds)
                homology_features["depsilon"] = de
                homology_features["scenario"] = "parent_to_children"
                homology_features["direct_children_count"] = min(len(direct_children), random_direct_children)

                results.append(homology_features)

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df)} query-corpus pairs")
    # print(f"Parent nodes analyzed: {len(parent_ids)}")

    return results_df


def scenario_4_parent_to_child_var_overlap(
    datapoints,
    noise_scale: float,
    depsilon: list,
):
    """
    Scenario 3: Query from child datapoint, corpus from children with
    varying feature overlap.

    For each child node:
    - Use one child datapoint as query
    - Build corpus subsets of other children grouped by number of shared features
    - Calculate homology for each overlap level
    - Track feature overlap count as key metadata

    This scenario tests how homology features correlate with feature similarity.

    Args:
        datapoints: List of generated datapoints with features_ids
        noise_scale: Noise scale used in generation (for metadata)
        depsilon: List of epsilon thresholds to scan
        min_corpus_size: Minimum number of points required for a corpus subset

    Returns:
        results_df: DataFrame with homology features and feature overlap metadata
    """

    labels_df = pd.DataFrame(datapoints)

    # Find all parent nodes that have both outliers and children, at depth >= min_depth
    parent_candidates = labels_df[labels_df["type"] == "parent"]

    results = []
    children_nfeature_groups = set(labels_df.n_features)
    for _, parent in tqdm(parent_candidates.iterrows(), desc="Processing parent nodes"):
        # Get parent outliers
        for nfeat in children_nfeature_groups:

            # Find children of this parent
            children_df = labels_df[(labels_df["type"] != "parent") & (labels_df["n_features"] == nfeat)]

            random_direct_children = np.random.randint(10, 40)

            direct_children = children_df.loc[
                children_df["features_ids"].apply(lambda x: len(set(x).difference(parent.features_ids)) == 0)
            ]
            if len(direct_children) == 0:
                continue
            direct_children = direct_children.sample(n=min(len(direct_children), random_direct_children))

            uncorrelated_children = children_df.loc[
                children_df["features_ids"].apply(lambda x: 0 < len(set(x).difference(parent.features_ids)) < len(x))
            ]
            if len(uncorrelated_children) == 0:
                continue
            if len(uncorrelated_children) < (50 - min(len(direct_children), random_direct_children)):
                continue
            uncorrelated_children = uncorrelated_children.sample(
                n=50 - min(len(direct_children), random_direct_children)
            )

            sampled_children = pd.concat([direct_children, uncorrelated_children])

            if len(children_df) == 0:
                continue

            corpus_embeds = np.array([list(x) for x in sampled_children.X])
            query_embed = np.array(parent.X)
            print(f"{query_embed.shape=}", f"{corpus_embeds.shape=}")

            for de in depsilon:
                homology_features = calculate_homology_for_query_corpus(query_embed, corpus_embeds, depsilon=de)
                # Add metadata
                homology_features["query_active_dims"] = parent.n_features
                homology_features["corpus_active_dims"] = nfeat
                homology_features["dim_ratio"] = parent.n_features / nfeat
                homology_features["n_corpus_points"] = len(corpus_embeds)
                homology_features["depsilon"] = de
                homology_features["scenario"] = "parent_to_children"
                homology_features["direct_children_count"] = min(len(direct_children), random_direct_children)

                results.append(homology_features)

    results_df = pd.DataFrame(results)
    print(f"\nProcessed {len(results_df)} query-corpus pairs")
    # print(f"Parent nodes analyzed: {len(parent_ids)}")

    return results_df


class ScenarioFactory:

    scenario_map = {
        1: scenario_1_parent_to_children,
        2: scenario_2_child_to_parent,
        3: scenario_3_parent_to_child_no_overlap,
        4: scenario_4_parent_to_child_var_overlap,
    }
