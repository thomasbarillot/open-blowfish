# %%
"""
Persistent Homology Calculation Functions

This module provides functions to calculate persistent homology features
for query-corpus pairs using Ripser persistent homology library.

Adapted from sparse_embedding_homology_experiment.ipynb and autotagging_experiment_RKYC.py
"""

import traceback

import numpy as np
from ripser import Rips
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances


def calculate_first_order_homology_distribution(
    q_embed: np.ndarray, chunks_embed: np.ndarray, sorted_scores: np.ndarray, pred_types: np.ndarray, depsilon: float, reverse=False, *args
) -> dict[str, float]:
    """
    Calculates the topological features for the embeddings of both the query and retrieved chunks.
    Adapted from autotagging_experiment_RKYC.py

    Args:
        q_embed: (d,) array - query embedding
        chunks_embed: (n, d) array - corpus embeddings sorted by distance
        scores: (n,) array - distances from query to corpus points
        pred_types: (n,) array - indices for tracking neighbors
        depsilon: threshold for filtering neighbors by distance ratio
        reverse: pick the tail of neigbours rather than the head

    Returns:
        dict with homology features:
            - max_homology_birth: max 0th homology birth time
            - mean_homology_birth: mean 0th homology birth time
            - w1_h0: Wasserstein-1 distance for 0th homology
            - std_homology_birth: std of 0th homology birth times
            - mean_homology1st_birth: mean 1st homology birth time
            - mean_homology1st_lifetime: mean 1st homology hole lifetime
            - ltmax_h1: maximum 1st homology hole lifetime
            - nneighbours: number of neighbors after filtering
            - all_nneighbours: total number of neighbors before filtering
            - which_neighbours: set of neighbor indices after filtering
    """
    rips = Rips(verbose=False)
    if reverse:
        mask = (np.sort(sorted_scores) / np.min(np.sort(sorted_scores))) - 1 >= depsilon
    else:
        mask = (np.sort(sorted_scores) / np.min(np.sort(sorted_scores))) - 1 <= depsilon
    all_neighbours = set(pred_types)
    neighbours = set(pred_types[mask])
    try:
        
        if sum(mask) < 3:
            raise ValueError("Not enough neighbors after filtering")
        chunks_embed = chunks_embed[mask, :]
        query_embed = q_embed

        renormalised_embeddings = chunks_embed - query_embed[np.newaxis, :] + np.finfo(float).eps
        renormalised_embeddings = (
            renormalised_embeddings / np.linalg.norm(renormalised_embeddings, axis=-1)[:, np.newaxis]
        )

        diagrams = rips.fit_transform(renormalised_embeddings[:, :])

        neighbour_0th_homology = diagrams[0][:-1, 1]
        neighbour_1st_homology = diagrams[1][:, :2]

        holes_lifetimes = neighbour_1st_homology[:, 1] - neighbour_1st_homology[:, 0]
        if len(holes_lifetimes) == 0:
            ltmax_h1 = np.nan
        else:
            ltmax_h1 = np.max(holes_lifetimes)

        entropy_range = np.arange(0.0, 1.1, 0.01)
        h, _ = np.histogram(neighbour_0th_homology, bins=entropy_range, density=True)
        h = h / sum(h)
        S0 = entropy(h)

        h, _ = np.histogram(holes_lifetimes, bins=entropy_range, density=True)
        h = h / sum(h)
        S1 = entropy(h)

        homology_distribution = {
            "max_homology_birth": np.max(neighbour_0th_homology),
            "mean_homology_birth": np.mean(neighbour_0th_homology),
            "w1_h0": np.sum(np.abs(neighbour_0th_homology - neighbour_0th_homology / 2.0))
            / (len(neighbour_0th_homology) - 1),
            "std_homology_birth": np.std(neighbour_0th_homology),
            "mean_homology1st_birth": np.mean(neighbour_1st_homology, axis=0)[0],
            "mean_homology1st_lifetime": np.mean(holes_lifetimes),
            "h0_dist": neighbour_0th_homology,
            "h0_entropy": S0,
            "h1_dist": holes_lifetimes,
            "h1_entropy": S1,
            "ltmax_h1": ltmax_h1,
            "nneighbours": len(neighbours),
            "all_nneighbours": len(all_neighbours),
            "which_neighbours": neighbours,
            "scores": list(sorted_scores),
        }
        return homology_distribution
    except Exception:
        #print(traceback.format_exc())
        homology_distribution = {
            "max_homology_birth": np.nan,
            "mean_homology_birth": np.nan,
            "w1_h0": np.nan,
            "std_homology_birth": np.nan,
            "mean_homology1st_birth": np.nan,
            "mean_homology1st_lifetime": np.nan,
            "h0_dist": np.nan,
            "h0_entropy": np.nan,
            "h1_dist": np.nan,
            "h1_entropy": np.nan,
            "ltmax_h1": np.nan,
            "nneighbours": len(neighbours),
            "all_nneighbours": len(all_neighbours),
            "which_neighbours": neighbours,
            "scores": list(sorted_scores),
        }
        return homology_distribution


def calculate_homology_for_query_corpus(query_embed, corpus_embeds, depsilon=1.2, reverse=False, **kwargs):
    """
    Calculate homology features for a single query against a corpus.

    Args:
        query_embed: (d,) array - query embedding
        corpus_embeds: (n_corpus, d) array - corpus embeddings
        depsilon: threshold for filtering neighbors by distance

    Returns:
        dict with homology features (see calculate_first_order_homology_distribution for details)
    """
    # Compute euclidean distances
    all_embeds = np.vstack([query_embed[np.newaxis, :], corpus_embeds])
    distances = euclidean_distances(all_embeds)

    # Get distances from query to all corpus points
    dist_scores = distances[1:, 0]

    # Sort by distance
    sorting_indices = np.argsort(dist_scores)
    sorted_scores = dist_scores[sorting_indices]
    sorted_embeddings = corpus_embeds[sorting_indices]
    # Create dummy pred_types (just indices for tracking)
    pred_types = np.arange(len(sorted_scores))

    # Calculate homology
    homology_features = calculate_first_order_homology_distribution(
        query_embed, sorted_embeddings, sorted_scores, pred_types, depsilon=depsilon, reverse=reverse
    )

    return homology_features


def calculate_homology0_map(query_embed, corpus_embeds, reverse=False, **kwargs):
    """
    Calculate homology features for a single query against a corpus.

    Args:
        query_embed: (d,) array - query embedding
        corpus_embeds: (n_corpus, d) array - corpus embeddings
        depsilon: threshold for filtering neighbors by distance

    Returns:
        dict with homology features (see calculate_first_order_homology_distribution for details)
    """
    # Compute euclidean distances
    all_embeds = np.vstack([query_embed[np.newaxis, :], corpus_embeds])
    distances = euclidean_distances(all_embeds)

    # Get distances from query to all corpus points
    dist_scores = distances[1:, 0]

    # Sort by distance
    sorting_indices = np.argsort(dist_scores)
    sorted_scores = dist_scores[sorting_indices]
    sorted_embeddings = corpus_embeds[sorting_indices]
    sorted_depsilon = (np.sort(sorted_scores) / np.min(np.sort(sorted_scores)))
    # Create dummy pred_types (just indices for tracking)
    pred_types = np.arange(len(sorted_scores))
    print(f"{sorted_depsilon=}")
    # Calculate homology
    homology_map = []
    for neighbour_idx in range(len(corpus_embeds)):
        depsilon = 1e9#sorted_depsilon[neighbour_idx]
        #print(f"current {depsilon=}")
        homology_features = calculate_first_order_homology_distribution(
            query_embed, sorted_embeddings[:neighbour_idx+1, :], sorted_scores[:neighbour_idx+1], pred_types[:neighbour_idx+1], depsilon=depsilon, reverse=reverse
        )
        #print(f"current {homology_features["h0_dist"]=}")
        homology_map.append(homology_features["h0_dist"])

    return homology_map, sorting_indices
