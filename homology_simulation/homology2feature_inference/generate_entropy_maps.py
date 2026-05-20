#!/usr/bin/env python3
"""
Generate simulated entropy maps and label matrices for training the feature predictor.

Each map is an 80x80 2D histogram of (depsilon, h0_birth_value) accumulated across
incremental persistent homology computations on a Markov-ordered neighbor sequence.
"""

import argparse
import csv
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
from ripser import Rips
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.homology_simulation.feature_bank_generator import FeaturesGenerator

EMBEDDING_DIM = 256
MAX_FEATURES = 64
QUERY_N_FEATURES = 10
NOISE = 0.1
MAX_NEIGHBORS = 50
N_BINS_H0 = 20
DEFAULT_N_BINS_NN = 50  # neighbour-axis bin count (was 20)
H0_BIRTH_BINS = np.linspace(0.0, 1.5, N_BINS_H0 + 1)

POOL_PER_N = 10




def _batch_embeddings(feature_vectors, selections, noise_scale, rng, d):
    """
    Generate normalized embeddings from pre-selected feature indices.

    Args:
        feature_vectors: (max_features, d) array from FeaturesGenerator
        selections: list of 1-D arrays, each containing feature indices for one point
        noise_scale: std of additive gaussian noise (0 = no noise)
        rng: numpy Generator
        d: embedding dimension

    Returns:
        (n_points, d) normalized embeddings
    """
    n = len(selections)
    X = np.zeros((n, d), dtype=np.float64)
    for i, sel in enumerate(selections):
        X[i] = feature_vectors[sel].sum(axis=0)
    if noise_scale > 0:
        X += rng.normal(scale=noise_scale, size=(n, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X /= norms
    return X.astype(np.float64)


MODES = [
    "subset",           # only subset pool
    "superset",         # only superset pool
    "mixed",            # only mixed pool
    "subset+superset",  # 50/50 subset and superset
    "subset+mixed",     # 50/50 subset and mixed
    "superset+mixed",   # 50/50 superset and mixed
    "all",              # uniform across all three pools
]


def _build_pool(selections, fv, query_embed, rng):
    """Embed selections, compute distances to query, sort by distance."""
    embeds = _batch_embeddings(fv, selections, NOISE, rng, EMBEDDING_DIM)
    dists = np.linalg.norm(embeds - query_embed[np.newaxis, :], axis=1)
    order = np.argsort(dists)
    return (
        dists[order],
        embeds[order],
        [selections[i] for i in order],
    )


def generate_single_map(
    fg: FeaturesGenerator,
    rng: np.random.Generator,
    mode: str = "all",
    n_bins_nn: int = DEFAULT_N_BINS_NN,
) -> dict:
    """
    Generate one entropy map + label matrix using distance-stepping selection.

    Builds three pools of candidate neighbours:
      - Subset pool: n features in [1, QUERY_N_FEATURES-1], all from query features
      - Superset pool: n features in [QUERY_N_FEATURES+1, MAX_FEATURES-1],
        using all query features plus extras
      - Mixed pool: x features from query (x in [1, QUERY_N_FEATURES-1]) plus
        y features from non-query (y in [1, MAX_FEATURES-x])

    mode controls which pools are used during neighbour selection:
      - "subset": only subset pool
      - "superset": only superset pool
      - "mixed": only mixed pool
      - "subset+superset": 50/50 subset and superset
      - "subset+mixed": 50/50 subset and mixed
      - "superset+mixed": 50/50 superset and mixed
      - "all": uniform across all three pools

    Returns dict with keys: entropy_map, label_matrix, p_feature_scale, query_feature_ids, mode
    """
    query_feature_ids = rng.choice(MAX_FEATURES, size=QUERY_N_FEATURES, replace=False)
    query_feature_ids_set = set(query_feature_ids.tolist())
    non_query_features = np.array([f for f in range(MAX_FEATURES) if f not in query_feature_ids_set])

    fv = fg.feature_vectors

    # Query embedding (no noise)
    query_embed = fv[query_feature_ids].sum(axis=0)
    query_embed = query_embed / np.linalg.norm(query_embed)

    p_feature_scale = rng.uniform(0.5, 2.0)

    # --- Build subset pool ---
    subset_selections = []
    for n in range(1, QUERY_N_FEATURES):
        for _ in range(POOL_PER_N):
            subset_selections.append(rng.choice(query_feature_ids, size=n, replace=False))

    subset_pool = _build_pool(subset_selections, fv, query_embed, rng)

    # --- Build superset pool ---
    superset_selections = []
    for n in range(QUERY_N_FEATURES + 1, MAX_FEATURES):
        n_extra = n - QUERY_N_FEATURES
        if n_extra > len(non_query_features):
            continue
        for _ in range(POOL_PER_N):
            extra = rng.choice(non_query_features, size=n_extra, replace=False)
            superset_selections.append(np.concatenate([query_feature_ids, extra]))

    superset_pool = _build_pool(superset_selections, fv, query_embed, rng)

    # --- Build mixed pool: x from query + y from non-query ---
    mixed_selections = []
    for x in range(1, QUERY_N_FEATURES):
        max_y = min(MAX_FEATURES - x, len(non_query_features))
        for y in range(1, max_y + 1):
            for _ in range(max(1, POOL_PER_N // (QUERY_N_FEATURES - 1))):
                q_part = rng.choice(query_feature_ids, size=x, replace=False)
                nq_part = rng.choice(non_query_features, size=y, replace=False)
                mixed_selections.append(np.concatenate([q_part, nq_part]))

    mixed_pool = _build_pool(mixed_selections, fv, query_embed, rng)

    # --- Select active pools based on mode ---
    pool_map = {
        "subset": subset_pool,
        "superset": superset_pool,
        "mixed": mixed_pool,
    }
    mode_to_pools = {
        "subset": ["subset"],
        "superset": ["superset"],
        "mixed": ["mixed"],
        "subset+superset": ["subset", "superset"],
        "subset+mixed": ["subset", "mixed"],
        "superset+mixed": ["superset", "mixed"],
        "all": ["subset", "superset", "mixed"],
    }
    pool_keys = mode_to_pools[mode]
    pools = [pool_map[k] for k in pool_keys]

    # --- Iteratively select MAX_NEIGHBORS neighbours with increasing distance ---
    neighbor_embeds = np.empty((MAX_NEIGHBORS, EMBEDDING_DIM), dtype=np.float64)
    neighbor_feature_lists = [None] * MAX_NEIGHBORS
    distances = np.empty(MAX_NEIGHBORS, dtype=np.float64)

    # Pre-compute feature sets for each pool element (for clustering overlap check)
    pool_feature_sets = []
    for pool_dists, pool_embeds, pool_sels in pools:
        pool_feature_sets.append([set(s.tolist()) for s in pool_sels])

    # Clustering probability: with p_cluster, prefer neighbours sharing features
    # with the previous one
    p_cluster = rng.uniform(0.0, 0.8)

    # First neighbour: random pick from a random available pool
    first_pool_idx = rng.integers(0, len(pools))
    pd, pe, ps = pools[first_pool_idx]
    first_idx = rng.integers(0, len(pd))
    neighbor_embeds[0] = pe[first_idx]
    neighbor_feature_lists[0] = ps[first_idx]
    distances[0] = pd[first_idx]

    n_pools = len(pools)
    for t in range(1, MAX_NEIGHBORS):
        step = rng.uniform(0.0, 0.02)
        threshold = distances[t - 1] * (1.0 + step)
        prev_features = set(neighbor_feature_lists[t - 1].tolist())
        clustering = rng.random() < p_cluster

        # Randomise pool visit order for multi-pool modes
        order = rng.permutation(n_pools).tolist() if n_pools > 1 else [0]

        chosen_idx = None
        for pi in order:
            pool_dists, pool_embeds, pool_sels = pools[pi]
            start = bisect_left(pool_dists, threshold)
            if start >= len(pool_dists):
                continue

            if clustering:
                # Scan from threshold onwards for a candidate sharing features
                for idx in range(start, len(pool_dists)):
                    if pool_feature_sets[pi][idx] & prev_features:
                        neighbor_embeds[t] = pool_embeds[idx]
                        neighbor_feature_lists[t] = pool_sels[idx]
                        distances[t] = pool_dists[idx]
                        chosen_idx = idx
                        break
                if chosen_idx is not None:
                    break
            else:
                neighbor_embeds[t] = pool_embeds[start]
                neighbor_feature_lists[t] = pool_sels[start]
                distances[t] = pool_dists[start]
                chosen_idx = start
                break

        # Fallback: ignore clustering constraint
        if chosen_idx is None:
            for pi in order:
                pool_dists, pool_embeds, pool_sels = pools[pi]
                start = bisect_left(pool_dists, threshold)
                if start < len(pool_dists):
                    neighbor_embeds[t] = pool_embeds[start]
                    neighbor_feature_lists[t] = pool_sels[start]
                    distances[t] = pool_dists[start]
                    chosen_idx = start
                    break

        # Last resort: farthest element across available pools
        if chosen_idx is None:
            best = None
            for pd, pe, ps in pools:
                if len(pd) > 0 and (best is None or pd[-1] > best[2]):
                    best = (pe[-1], ps[-1], pd[-1])
            neighbor_embeds[t] = best[0]
            neighbor_feature_lists[t] = best[1]
            distances[t] = best[2]

    # Build entropy map by incremental homology: row index = t - 1 (one row per neighbor count)
    rips = Rips(verbose=False)
    entropy_map = np.zeros((n_bins_nn, N_BINS_H0), dtype=np.float32)

    for t in range(3, MAX_NEIGHBORS + 1):
        row_idx = t - 1
        if row_idx >= n_bins_nn:
            break

        subset = neighbor_embeds[:t] - query_embed[np.newaxis, :] + np.finfo(float).eps
        subset = subset / np.linalg.norm(subset, axis=-1, keepdims=True)

        try:
            diagrams = rips.fit_transform(subset)
            h0_births = diagrams[0][:-1, 1]
        except Exception:
            continue

        if len(h0_births) == 0:
            continue

        h, _ = np.histogram(h0_births, bins=H0_BIRTH_BINS)
        entropy_map[row_idx] += h.astype(np.float32)

    # Build label matrix (50 neighbours x 64 features)
    label_matrix = np.zeros((MAX_NEIGHBORS, MAX_FEATURES), dtype=np.int8)
    for t, features in enumerate(neighbor_feature_lists):
        for f in features:
            if f in query_feature_ids_set:
                label_matrix[t, f] = 1
            else:
                label_matrix[t, f] = -1

    return {
        "entropy_map": entropy_map,
        "label_matrix": label_matrix,
        "p_feature_scale": p_feature_scale,
        "query_feature_ids": query_feature_ids,
        "mode": mode,
        "p_cluster": p_cluster,
    }


S3_BUCKET = "homology-experiment"
S3_PREFIX = "simulation_dataset"


def upload_to_s3(local_dir: Path, bucket: str, prefix: str):
    """Upload dataset files from local_dir to s3://bucket/prefix/."""
    import boto3

    from config import AWS_PROFILE_NAME

    session = boto3.Session(profile_name=AWS_PROFILE_NAME)
    s3 = session.client("s3")

    files = ["entropy_maps.npy", "label_matrices.npy", "metadata.csv"]
    for fname in files:
        local_path = local_dir / fname
        s3_key = f"{prefix}/{fname}"
        print(f"Uploading {local_path} -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)
    print(f"Upload complete: s3://{bucket}/{prefix}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate entropy maps for feature predictor training")
    parser.add_argument("--n-maps", type=int, default=15000, help="Number of maps to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "simulation_dataset_v2"),
        help="Output directory",
    )
    parser.add_argument(
        "--n-neighbour-bins",
        type=int,
        default=DEFAULT_N_BINS_NN,
        help="Number of n_neighbour bins in the entropy map (default 50 = one per neighbor)",
    )
    parser.add_argument("--upload-s3", action="store_true", help="Upload to S3 after generation")
    parser.add_argument("--s3-bucket", type=str, default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default=S3_PREFIX, help="S3 key prefix")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.n_maps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fg = FeaturesGenerator(max_features=MAX_FEATURES, d=EMBEDDING_DIM, seed=args.seed)

    # Pre-allocate memory-mapped files to avoid accumulating everything in RAM
    maps_path = output_dir / "entropy_maps.npy"
    labels_path = output_dir / "label_matrices.npy"
    maps_mmap = np.lib.format.open_memmap(
        str(maps_path),
        mode="w+",
        dtype=np.float32,
        shape=(n, args.n_neighbour_bins, N_BINS_H0),
    )
    labels_mmap = np.lib.format.open_memmap(
        str(labels_path), mode="w+", dtype=np.int8, shape=(n, MAX_NEIGHBORS, MAX_FEATURES),
    )

    meta_fields = ["map_id", "p_feature_scale", "p_cluster", "seed", "query_feature_ids", "mode"]
    meta_file = open(output_dir / "metadata.csv", "w", newline="")
    meta_writer = csv.DictWriter(meta_file, fieldnames=meta_fields)
    meta_writer.writeheader()

    for i in tqdm(range(n), desc="Generating maps"):
        rng = np.random.default_rng(args.seed + i)
        mode = MODES[i % len(MODES)]
        result = generate_single_map(fg, rng, mode=mode, n_bins_nn=args.n_neighbour_bins)

        maps_mmap[i] = result["entropy_map"]
        labels_mmap[i] = result["label_matrix"]
        meta_writer.writerow({
            "map_id": i,
            "p_feature_scale": result["p_feature_scale"],
            "p_cluster": result["p_cluster"],
            "seed": args.seed + i,
            "query_feature_ids": ",".join(map(str, result["query_feature_ids"])),
            "mode": mode,
        })

        # Flush mmap periodically to avoid dirty-page buildup
        if (i + 1) % 10000 == 0:
            maps_mmap.flush()
            labels_mmap.flush()
            meta_file.flush()

    maps_mmap.flush()
    labels_mmap.flush()
    meta_file.close()
    del maps_mmap, labels_mmap

    print(f"Saved {n} maps to {output_dir}")
    print(f"  entropy_maps.npy: ({n}, {args.n_neighbour_bins}, {N_BINS_H0})")
    print(f"  label_matrices.npy: ({n}, {MAX_NEIGHBORS}, {MAX_FEATURES})")
    print(f"  metadata.csv: {n} rows")

    if args.upload_s3:
        upload_to_s3(output_dir, args.s3_bucket, args.s3_prefix)
