"""
Copyright 2024 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence

#: Minimum surviving neighbors required after ε filtering before VR runs. Below
#: this the filter is ignored and the full neighborhood is used (degenerate H₁
#: diagrams below ~4 points are not informative).
DEFAULT_EPSILON_N_MIN = 4


def calculate_scaled_distance_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    """
        Calculates the minimum value, the mean and the interquartile range for the similarity
        metric (e.g. L2/Cosine/Inner Product)
    """
    sorted_scores = np.sort(np.asarray(sub_df.score, dtype=np.float64))
    if sorted_scores.size < 2:
        scale = np.array([0.0], dtype=np.float64)
    else:
        tail = sorted_scores[1:]
        denominator = sorted_scores[0]
        if abs(denominator) <= 1e-12:
            nonzero_tail = tail[np.abs(tail) > 1e-12]
            denominator = nonzero_tail[0] if nonzero_tail.size else 1.0
        scale = tail / denominator
    
    scale_distribution = {
                        "scale_mean": np.mean(scale),
                        "scale_min": np.min(scale),
                        "iq25-75_scale": np.quantile(scale, 0.75) - np.quantile(scale, 0.25),
                        }
    return scale_distribution


def _single_sample_diagram(diagrams: np.ndarray) -> np.ndarray:
    """Flatten giotto batch output to (n_points, 3) with columns birth, death, dimension."""
    arr = np.asarray(diagrams)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected persistence diagram shape {arr.shape}")
    return arr


def _finite_intervals(birth: np.ndarray, death: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Keep finite strict intervals (death > birth)."""
    finite = np.isfinite(birth) & np.isfinite(death) & (death > birth + 1e-15)
    return birth[finite], death[finite]


def _rows_for_dim(dgm: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    dimcol = dgm[:, 2]
    mask = np.isclose(dimcol.astype(float), float(dim))
    return _finite_intervals(dgm[mask, 0].astype(np.float64), dgm[mask, 1].astype(np.float64))


def paper_eq1_scaled_neighbor_distances(
    chunks_embed: np.ndarray,
    query_embed: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Paper arXiv:2406.07990 Eq. (1): ``d(i,q) = ||Δv_iq|| / ||Δv_0q||`` where
    ``Δv_iq = chunks[i] - query`` and ``Δv_0q`` is the nearest-neighbor offset
    (smallest ``||Δv_iq||``). The nearest neighbor has d=1 by construction;
    farther neighbors have d≥1.

    Note: paper §3.6 retrieves the top-k by **cosine** similarity, then Eq. (1)
    re-scales by **Euclidean** offset norms within that retrieved set. For
    L2-normalized embeddings (the four models in Table 2) cosine top-1 and
    L2-nearest coincide; if a future retriever uses unnormalized embeddings
    these can diverge — ``v_0`` here is always the geometrically nearest
    chunk inside whatever set was passed in.

    Safe handling when the nearest-neighbor offset has zero norm (the query
    coincides numerically with a retrieved chunk): fall back to the next-
    smallest strictly positive offset norm. If every offset has zero norm,
    return zeros — the neighborhood is degenerate and ε filtering downstream
    will fall back to the unfiltered path.
    """
    chunks = np.asarray(chunks_embed, dtype=np.float64)
    if chunks.ndim != 2 or chunks.shape[0] == 0:
        return np.array([], dtype=np.float64)
    query = np.asarray(query_embed, dtype=np.float64)
    if query.ndim == 2:
        query = query[0]
    offsets = chunks - query
    norms = np.linalg.norm(offsets, axis=1)
    nonzero = norms[norms > zero_tol]
    if nonzero.size == 0:
        return np.zeros_like(norms)
    denom = float(np.min(nonzero))
    return norms / denom


def paper_w1_h0_from_birth_death(birth: np.ndarray, death: np.ndarray) -> float:
    """
    Paper arXiv:2406.07990 Eq. (2): ``W1(H0) = (1/(N-1)) * sum |y - gamma_perp(y)|``
    where N is the total number of H0 features (including the infinite component) and the
    sum runs over the **finite** features. With K input points, N = K and the number of
    finite features is K-1, so the divisor (N-1) equals the number of summands. In practice:

        W1(H0) = mean over finite H0 bars of |death - (birth+death)/2| = mean of |d-b|/2.
    """
    b, d = _finite_intervals(np.asarray(birth, dtype=np.float64), np.asarray(death, dtype=np.float64))
    n = b.size
    if n == 0:
        return 0.0
    mid = (b + d) / 2.0
    contrib = np.abs(d - mid)
    return float(np.sum(contrib) / n)


def paper_lt_max_h1(birth: np.ndarray, death: np.ndarray) -> float:
    """
    Paper arXiv:2406.07990 Eq. (3): ``LT_max(H1) = sup |y - gamma_perp(y)|``.
    For an H1 point (b, d), |d - (b+d)/2| = (d-b)/2, so this returns max((d-b)/2) over
    finite H1 bars (half-life convention used in the paper).
    """
    b, d = _finite_intervals(np.asarray(birth, dtype=np.float64), np.asarray(death, dtype=np.float64))
    if b.size == 0:
        return 0.0
    return float(np.max(d - b) / 2.0)


def calculate_vr_persistence_features(
    sub_df: pd.DataFrame,
    *args,
    epsilon: Optional[float] = None,
    n_min: int = DEFAULT_EPSILON_N_MIN,
) -> Dict[str, Any]:
    """
        Vietoris–Rips persistence on query-centered, unit-normalized chunk offsets.
        Exports paper-aligned ``w1_h0`` and ``lt_max_h1`` (arXiv:2406.07990) plus legacy keys
        for backward-compatible KDE configs loaded from older pickles.

        TASK-004: when ``epsilon`` is provided, neighbors are first filtered to those
        satisfying paper Eq. (1) ``d(i,q) = ||Δv_iq|| / ||Δv_0q|| <= epsilon`` before
        VR runs (approach A: subsample by ε). If fewer than ``n_min`` neighbors
        survive the filter, the full neighborhood is used (H₁ diagrams below ~4
        points are not informative). ``epsilon=None`` preserves prior behavior.

        Axis convention: the paper writes ``ε = d(i,q) − 1`` (ε ∈ [0, …]), while
        this kwarg is ``d(i,q)`` directly (ε ∈ [1, …]). To compare with paper
        figure captions like "ε = 0.4", pass ``epsilon=1.4`` here.
    """
    vietorisRipsGenerator = VietorisRipsPersistence(homology_dimensions=(0, 1))

    chunks_embed = np.array(sub_df["chunk_embeddings"].to_list())
    query_embed = np.array(sub_df["query_embedding"].to_list())

    if epsilon is not None:
        scaled = paper_eq1_scaled_neighbor_distances(chunks_embed, query_embed)
        if scaled.size:
            mask = scaled <= float(epsilon)
            if int(mask.sum()) >= n_min:
                chunks_embed = chunks_embed[mask]
                query_embed = query_embed[mask] if query_embed.ndim == 2 else query_embed

    renormalised_embeddings = chunks_embed - query_embed
    norms = np.linalg.norm(renormalised_embeddings, axis=1)[:, np.newaxis]
    norms = np.maximum(norms, 1e-15)
    renormalised_embeddings = renormalised_embeddings / norms

    diagrams = vietorisRipsGenerator.fit_transform(renormalised_embeddings[np.newaxis, :, :])
    dgm = _single_sample_diagram(diagrams)

    b0, d0 = _rows_for_dim(dgm, 0)
    b1, d1 = _rows_for_dim(dgm, 1)

    w1_h0 = paper_w1_h0_from_birth_death(b0, d0)
    lt_max_h1 = paper_lt_max_h1(b1, d1)

    holes_lifetimes = (d1 - b1) if b1.size else np.array([], dtype=np.float64)

    legacy_max_death = float(np.max(d0)) if d0.size else 0.0
    legacy_mean_death = float(np.mean(d0)) if d0.size else 0.0
    legacy_std_death = float(np.std(d0)) if d0.size else 0.0
    legacy_mean_h1_birth = float(np.mean(b1)) if b1.size else 0.0
    legacy_mean_h1_life = float(np.mean(holes_lifetimes)) if holes_lifetimes.size else 0.0

    return {
        "w1_h0": w1_h0,
        "lt_max_h1": lt_max_h1,
        "max_homology_birth": legacy_max_death,
        "mean_homology_birth": legacy_mean_death,
        "std_homology_birth": legacy_std_death,
        "mean_homology1st_birth": legacy_mean_h1_birth,
        "mean_homology1st_lifetime": legacy_mean_h1_life,
    }


calculate_first_order_homology_distribution = calculate_vr_persistence_features


def calculate_silhouette_score_distribution(sub_df: pd.DataFrame, *args) -> Dict[str, Any]:
    """
        Calculates the mean of the standard deviation of the silhouette score
    """
    silhouette_score_distribution = {
                                    "silhouette_score_mean": sub_df["silhouette_score"].mean(),
                                    "silhouette_score_std": sub_df["silhouette_score"].std()
                                    }
    return silhouette_score_distribution


def calculate_doc_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    """
        Calculates the number of number of distinct documents of the retrieved chunks over
        the total number of total chunks retrieved
    """
    return {"top_k_doc_spread": len(set(sub_df["docname"])) / len(sub_df)}


def calculate_topic_spread(sub_df: pd.DataFrame) -> Dict[str, Any]:
    """
        Calculates the number of number of distinct topics contained in the the retrieved 
        chunks over the total number of total chunks retrieved
    """
    return {"top_k_topic_spread": len(set(sub_df["topic_label"])) / len(sub_df)}


def calculate_relevant_features(
    sub_df: pd.DataFrame,
    kde_features_order: List[str],
    *,
    epsilon: Optional[float] = None,
) -> Dict[str, Any]:
    """
        Compute all features and return only the ones in ``kde_features_order``,
        with the **dict iteration order following ``kde_features_order``** so that
        downstream consumers (training matrix construction, inference vectorization)
        see a single canonical column order.

        ``epsilon`` (TASK-004) is forwarded to the VR persistence step. ``None``
        preserves the pre-TASK-004 single-scale behavior.
    """
    feature_set = set(kde_features_order)

    features: Dict[str, Any] = {}
    features.update(calculate_scaled_distance_distribution(sub_df))
    features.update(calculate_vr_persistence_features(sub_df, epsilon=epsilon))
    features.update(calculate_silhouette_score_distribution(sub_df))
    features.update(calculate_doc_spread(sub_df))
    features.update(calculate_topic_spread(sub_df))

    missing = [k for k in kde_features_order if k not in features]
    if missing:
        raise KeyError(
            f"Requested KDE features not produced by feature pipeline: {missing}. "
            "Update kde_features_order or extend the feature calculators."
        )
    return {k: features[k] for k in kde_features_order if k in feature_set}
