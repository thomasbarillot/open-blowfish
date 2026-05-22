# %%
import random
import traceback
from hashlib import sha256

from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ripser import Rips
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# %%
# ============================================================================
# DIMENSIONAL CLUSTERING
# ============================================================================


def _generate_orthogonal_features(max_features, d, seed) -> dict[str, np.ndarray]:
    """
    Generate a dictionary of orthogonal binary feature vectors.

    Each feature is a normalized binary vector in d-dimensional space.
    Vectors are approximately orthogonal (binary constraint introduces small error).

    Args:
        max_features: Number of feature vectors to generate
        d: Dimension of embedding space
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping feature_id -> normalized binary vector (d,)
    """
    rng = np.random.default_rng(seed)

    # Generate random matrix and get orthonormal basis via QR decomposition
    random_matrix = rng.standard_normal((d, max_features))
    Q, _ = np.linalg.qr(random_matrix)

    # Q now contains orthonormal vectors in columns
    # Threshold at median to create binary patterns
    feature_bank = {}
    for i in range(max_features):
        vector = Q[:, i]
        # Threshold at median to get roughly half 1s and half 0s
        binary_vector = vector  # (vector >= threshold).astype(float)

        # Normalize to unit length
        norm = np.linalg.norm(binary_vector)
        if norm > 0:
            binary_vector = binary_vector / norm
        else:
            # Fallback: if all zeros, set first element to 1
            binary_vector[0] = 1.0
        # binary_vector /= binary_vector.sum()
        feature_bank[i] = binary_vector

    return feature_bank


def _generate_single_dim_orthogonal_features(max_features: int, d: int, seed: int) -> dict[int, np.ndarray]:
    """Generate a dictionary of orthogonal binary feature vectors.

    Parameters
    ----------
    max_features:
        Number of feature vectors to generate.
    d:
        Dimension of the embedding space.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[int, np.ndarray]
        A mapping from feature identifier to a normalized binary vector.
    """

    feature_bank: dict[int, np.ndarray] = {}
    for i in range(max_features):
        binary_vector = np.zeros(d)
        binary_vector[i] = 1.0
        feature_bank[i] = binary_vector

    return feature_bank


def _build_covariance_matrices(feature_bank, d) -> np.ndarray:
    """
    Build covariance matrix from features.

    Each feature contributes covariance between its non-zero dimensions.
    The covariance map for multiple features is the sum of individual maps,
    with diagonal normalized to 1.

    Args:
        feature_ids: List of feature IDs for this datapoint
        feature_bank: Dictionary mapping feature_id -> binary vector
        d: Embedding dimension

    Returns:
        (d, d) covariance matrix with diagonal normalized to 1
    """
    cov_matrices = np.zeros((len(feature_bank), d, d))

    # Sum covariance contributions from all features
    for i in tqdm(feature_bank.keys()):
        mask = feature_bank[i]
        # Add outer product: covariance between all pairs of non-zero dims
        cov_matrix = np.zeros((d, d)) + np.outer(mask, mask)
        #diag = np.diag(cov_matrix).copy()
        #diag[diag == 0] = 1  # Avoid division by zero for unused dimensions
        #D_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
        #cov_matrix = cov_matrix + D_inv_sqrt
        #cov_matrix[cov_matrix > 0] = 1
        cov_matrices[i] = cov_matrix

    # Normalize diagonal to 1

    return cov_matrices


class ClusterNode:
    """
    Node in the hierarchical cluster tree.

    Each node represents a cluster at a specific depth, with a set of features
    that include inherited features (subset from parent) and optionally new features.
    """

    def __init__(self, depth: int, n_features: int, n_new_features: int = 0, parent: ClusterNode | None = None) -> None:
        """Create a new :class:`ClusterNode`.

        Parameters
        ----------
        depth:
            Tree depth of the node (root is ``0``).
        n_features:
            Number of inherited features for this node.
        n_new_features:
            Number of new features added at this node.
        parent:
            Reference to the parent :class:`ClusterNode` if any.
        """
        self.id = hex(random.getrandbits(128))[2:]
        self.depth = depth  # Tree depth (0 = root)
        self.n_features = n_features
        self.n_new_features = n_new_features
        self.children: list[ClusterNode] = []  # List of child ClusterNode objects
        self.parent: ClusterNode | None = parent  # Reference to the parent cluster

    def format_datapoint_instance(self, X: np.ndarray, features_ids: np.ndarray, type: str = "parent", **kwargs) -> dict:
        """Return a dictionary representation of a datapoint.

        Parameters
        ----------
        X:
            The embedding vector for the datapoint.
        features_ids:
            The feature identifiers that contributed to ``X``.
        type:
            The type of the node (e.g. ``"parent"`` or ``"child"``).
        **kwargs:
            Additional keyword arguments that are merged into the dictionary.

        Returns
        -------
        dict
            A dictionary containing the node metadata and the datapoint.
        """
        return {
            "depth": self.depth,
            "n_features": self.n_features + self.n_new_features,
            "parent_id": getattr(self.parent, "id", None),
            "children_ids": [c.id for c in self.children],
            "X": X,
            "features_ids": features_ids,
            "type": type,
        } | kwargs

    @property
    def has_new_features(self):
        """Returns True if this node added new features beyond inherited ones."""
        return self.n_new_features > 0

    def __repr__(self):
        return (
            f"ClusterNode(id={self.id}, depth={self.depth}, "
            f"n_features={self.n_features}, has_new={self.has_new_features})"
        )


class FeaturesGenerator:

    def __init__(self, max_features, d, seed):
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.n_max_features = max_features
        self.features_bank = _generate_orthogonal_features(max_features, self.d, seed)
        self.covariance_matrices = _build_covariance_matrices(self.features_bank, d)
        # Pre-convert feature bank to array for faster indexing
        self.feature_vectors = np.array([self.features_bank[fid] for fid in range(self.n_max_features)])
        self.cov_noise_storage = {}

    def generate_datapoints_from_features(
        self, node: ClusterNode, feature_ids=None, n_points=1, noise_scale=0.0, type="parent"
    ):
        """
        Generate n_points embeddings from features with covariance-based noise.

        Args:
            feature_ids: List of feature IDs for this cluster
            feature_bank: Dictionary mapping feature_id -> normalized binary vector
            n_points: Number of points to generate
            noise_scale: Standard deviation scaling for noise
            d: Full embedding dimension
            rng: Numpy random generator

        Returns:
            (n_points, d) array of embeddings
        """

        if feature_ids is None:
            feature_ids = self.rng.choice(np.arange(0, self.n_max_features, 1), size=node.n_features, replace=False)

        # Sample features for all points at once (n_points, n_features)
        # Constrain to uniform feature distribution
        all_point_features = []
        for i in range(n_points):
            all_point_features.append(self.rng.choice(feature_ids, size=node.n_features, replace=False))
        
        all_point_features = np.vstack(all_point_features)
        
        # Allow feature over-representation 
        # TODO: Can pass a pmf to modulate features representations here
        # all_point_features = self.rng.choice(feature_ids, size=(n_points, node.n_features), replace=True)
        
        # Vectorized feature summation: gather all feature vectors and sum
        # Shape: (n_points, n_features, d) -> (n_points, d)
        X = self.feature_vectors[all_point_features].sum(axis=1)

        # Generate noise for all points if noise_scale > 0
        if noise_scale > 0:
            # Process covariance matrices and noise in batches for memory efficiency
            batch_size = min(1000, n_points)
            for batch_start in tqdm(range(0, n_points, batch_size), desc="generate noise"):
                batch_end = min(batch_start + batch_size, n_points)
                batch_features = all_point_features[batch_start:batch_end]

                # Compute covariance matrices for batch
                #print(f"{all_point_features.shape=}", f"{self.covariance_matrices.shape=}")
                
                #batch_cov = self.covariance_matrices[batch_features].sum(axis=1)
                #print(f"{batch_cov.shape=}")
                #print(f"{all_point_features=}", f"{batch_cov=}")
                #batch_cov = np.where(batch_cov >= 1.0, 1.0, 0.0)

                # Generate noise for each point in batch (must loop as each has different cov)
                for i in range(batch_size):
                    #noise = self.cov_noise_storage.get(str(all_point_features[i]))
                    # if noise is None:
                    #     noise = np.abs(self.rng.multivariate_normal(mean=np.zeros(self.d), cov=cov_matrix))
                    #     self.cov_noise_storage[str(all_point_features[i])] = noise

                    X[batch_start + i] += np.random.normal(scale=noise_scale, size=self.d)

        # Vectorized normalization
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms

        parent_id = f"{getattr(node.parent, "id", "None")}_{sha256(str(feature_ids).encode("utf-8")).hexdigest()}"

        # Create datapoints list
        datapoints = [
            node.format_datapoint_instance(X[i], all_point_features[i], type=type, parent_id=parent_id)
            for i in range(n_points)
        ]

        return datapoints

