"""Interactive UMAP/HDBSCAN explorer for entropy maps.

Projects flattened entropy maps to 2D, clusters them with HDBSCAN, and writes a
self-contained HTML where hovering a highlighted point displays its label
matrix and a button reshuffles the 50/cluster highlighted set.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MAPS = Path.home() / "Downloads" / "entropy_maps.npy"
DEFAULT_LABELS = Path.home() / "Downloads" / "label_matrices.npy"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "entropy_umap_clusters.html"

NEIGHBOR_AXIS = 50  # rows of label matrix
FEATURE_AXIS = 64   # cols of label matrix
NUM_POS_FEATURES = 10   # divisor for the faithfulness summary (count of +1 / 10)


@dataclasses.dataclass
class PoolData:
    """Per-cluster candidate pool used by the JS resample handler."""
    cluster_ids: np.ndarray            # (P,) int — cluster id per pool entry
    xy: np.ndarray                     # (P, 2) float32 — UMAP coords
    labels: np.ndarray                 # (P, 50, 64) int8 — label matrices
    cluster_to_pool_indices: dict[int, list[int]]  # cluster id -> indices in pool


def load_and_subsample(
    maps_path: Path,
    labels_path: Path,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load entropy maps and label matrices as mmaps and draw a random subset.

    Returns (subset_idx, entropy_subset, labels_mmap):
      - subset_idx: (M,) int64 indices into the original arrays
      - entropy_subset: (M, 50, 20) float32 — copied into RAM
      - labels_mmap: full (N, 50, 64) int8 mmap — only pool rows are read later
    """
    maps_mmap = np.load(maps_path, mmap_mode="r")
    labels_mmap = np.load(labels_path, mmap_mode="r")
    assert maps_mmap.shape[0] == labels_mmap.shape[0], (
        f"shape mismatch: maps {maps_mmap.shape[0]} vs labels {labels_mmap.shape[0]}"
    )

    n_total = maps_mmap.shape[0]
    m = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    subset_idx = np.sort(rng.choice(n_total, size=m, replace=False)).astype(np.int64)
    entropy_subset = np.asarray(maps_mmap[subset_idx], dtype=np.float32)
    return subset_idx, entropy_subset, labels_mmap


def fit_umap(
    entropy_subset: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    """Flatten entropy maps to (M, 1000) and project to 2D with UMAP."""
    import umap

    m = entropy_subset.shape[0]
    flat = entropy_subset.reshape(m, -1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
    )
    embedding = reducer.fit_transform(flat).astype(np.float32)
    return embedding


def cluster_embedding(
    embedding_2d: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
) -> np.ndarray:
    """Run HDBSCAN. Returns (M,) cluster ids; -1 marks noise."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(embedding_2d).astype(np.int32)


def build_pool(
    cluster_ids: np.ndarray,
    embedding_2d: np.ndarray,
    labels_mmap: np.ndarray,
    subset_idx: np.ndarray,
    pool_per_cluster: int,
    seed: int,
) -> PoolData:
    """Pick up to `pool_per_cluster` indices per non-noise cluster, materialize
    their label matrices into a single (P, 50, 64) array, and return a PoolData."""
    rng = np.random.default_rng(seed)

    pool_subset_positions: list[int] = []
    pool_cluster_ids: list[int] = []
    cluster_to_pool: dict[int, list[int]] = {}

    unique = sorted(int(c) for c in np.unique(cluster_ids) if c != -1)
    for cid in unique:
        positions = np.where(cluster_ids == cid)[0]
        take = min(pool_per_cluster, len(positions))
        chosen = rng.choice(positions, size=take, replace=False)
        start = len(pool_subset_positions)
        pool_subset_positions.extend(int(p) for p in chosen)
        pool_cluster_ids.extend([cid] * take)
        cluster_to_pool[cid] = list(range(start, start + take))

    pool_subset_positions_arr = np.asarray(pool_subset_positions, dtype=np.int64)
    pool_xy = embedding_2d[pool_subset_positions_arr].astype(np.float32)
    original_indices = subset_idx[pool_subset_positions_arr]
    pool_labels = np.asarray(labels_mmap[original_indices], dtype=np.int8)

    return PoolData(
        cluster_ids=np.asarray(pool_cluster_ids, dtype=np.int32),
        xy=pool_xy,
        labels=pool_labels,
        cluster_to_pool_indices=cluster_to_pool,
    )


_QUALITATIVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]


def _initial_highlight_indices(pool: PoolData, sample_per_cluster: int, seed: int) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    out: dict[int, list[int]] = {}
    for cid, idxs in pool.cluster_to_pool_indices.items():
        take = min(sample_per_cluster, len(idxs))
        out[cid] = list(rng.choice(idxs, size=take, replace=False).tolist())
    return out


def summary_per_neighbor(label_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a (50, 64) label matrix to per-neighbor noise and faithfulness.

    noise[t]  = (count of label == -1 across the 64 features at neighbor t) / 64
    faith[t]  = (count of label == +1 across the 64 features at neighbor t) / 10

    Faithfulness uses NUM_POS_FEATURES (= 10) as the divisor — the assumed
    total number of positive features in the underlying label space — so a
    value of 1.0 means "all 10 positive features are firing at this neighbor".
    """
    neg = (label_matrix == -1).sum(axis=-1).astype(np.float32) / float(FEATURE_AXIS)
    pos = (label_matrix == 1).sum(axis=-1).astype(np.float32) / float(NUM_POS_FEATURES)
    return neg, pos


def compute_cluster_references(pool: PoolData) -> dict[int, dict[str, np.ndarray]]:
    """For each cluster, compute the per-neighbor mean of noise and
    faithfulness across the cluster's pool points.

    Returns {cid: {"noise": (50,), "shared": (50,)}}.
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    for cid, idxs in pool.cluster_to_pool_indices.items():
        sub = pool.labels[idxs]  # (k, 50, 64)
        neg = (sub == -1).sum(axis=-1).astype(np.float32) / float(FEATURE_AXIS)
        pos = (sub == 1).sum(axis=-1).astype(np.float32) / float(NUM_POS_FEATURES)
        out[int(cid)] = {"noise": neg.mean(axis=0), "shared": pos.mean(axis=0)}
    return out


def make_figure(
    embedding_2d: np.ndarray,
    cluster_ids: np.ndarray,
    pool: PoolData,
    sample_per_cluster: int,
    seed: int,
) -> tuple["plotly.graph_objects.Figure", dict]:
    """Build the explorer Plotly figure and return (fig, trace_map).

    Layout (2 rows x 3 cols, UMAP spans both rows on the left):
      col 1, rowspan=2: UMAP scatter
      row 1 col 2: raw 50x64 label-matrix heatmap (live)
      row 1 col 3: 50x2 [noise, faithfulness] summary heatmap (live)
      row 2 col 2: noise[t] over 50 neighbors (live + cluster mean)
      row 2 col 3: faithfulness[t] over 50 neighbors (live + cluster mean)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.46, 0.30, 0.24],
        row_heights=[0.55, 0.45],
        specs=[
            [{"rowspan": 2}, {}, {}],
            [None, {}, {}],
        ],
        subplot_titles=(
            "UMAP projection",
            "Label matrix (50 x 64)",
            "Summary (50 x 2)",
            "noise[t] = #(-1) / 64",
            "faithfulness[t] = #(+1) / 10",
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.13,
    )

    trace_map: dict = {"clusters": {}}

    fig.add_trace(
        go.Scattergl(
            x=embedding_2d[:, 0], y=embedding_2d[:, 1],
            mode="markers",
            marker=dict(size=3, color="#cccccc", opacity=0.15),
            hoverinfo="skip", name="Background", showlegend=True,
        ),
        row=1, col=1,
    )
    trace_map["background"] = len(fig.data) - 1

    noise_mask = cluster_ids == -1
    fig.add_trace(
        go.Scattergl(
            x=embedding_2d[noise_mask, 0], y=embedding_2d[noise_mask, 1],
            mode="markers",
            marker=dict(size=4, color="#888888", opacity=0.4),
            hoverinfo="skip", name="Noise", showlegend=True,
        ),
        row=1, col=1,
    )
    trace_map["noise"] = len(fig.data) - 1

    initial = _initial_highlight_indices(pool, sample_per_cluster, seed)
    sorted_cids = sorted(initial.keys())
    for i, cid in enumerate(sorted_cids):
        idxs = initial[cid]
        x = pool.xy[idxs, 0]
        y = pool.xy[idxs, 1]
        color = _QUALITATIVE_COLORS[i % len(_QUALITATIVE_COLORS)]
        cd = np.column_stack([
            np.asarray(idxs, dtype=np.int32),
            np.full(len(idxs), cid, dtype=np.int32),
        ])
        fig.add_trace(
            go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=8, color=color, line=dict(width=0.5, color="#222")),
                customdata=cd,
                hovertemplate=(
                    "cluster %d<br>pool idx=%%{customdata[0]}<extra></extra>" % cid
                ),
                name=f"Cluster {cid}", showlegend=True,
            ),
            row=1, col=1,
        )
        trace_map["clusters"][cid] = len(fig.data) - 1

    fig.add_trace(
        go.Heatmap(
            z=np.zeros((NEIGHBOR_AXIS, FEATURE_AXIS), dtype=np.int8),
            zmin=-1, zmax=1, colorscale="RdBu",
            colorbar=dict(title="value", thickness=10, len=0.4, y=0.80),
            name="Label matrix", showlegend=False,
            hovertemplate="t=%{y} feat=%{x}<br>v=%{z}<extra></extra>",
        ),
        row=1, col=2,
    )
    trace_map["heatmap"] = len(fig.data) - 1

    fig.add_trace(
        go.Heatmap(
            z=np.zeros((NEIGHBOR_AXIS, 2), dtype=np.float32),
            zmin=0, zmax=1, colorscale="Viridis",
            x=["noise", "faith"],
            colorbar=dict(title="value", thickness=10, len=0.4, y=0.80, x=1.02),
            name="Summary", showlegend=False,
            hovertemplate="t=%{y} %{x}=%{z:.3f}<extra></extra>",
        ),
        row=1, col=3,
    )
    trace_map["summary_heatmap"] = len(fig.data) - 1

    neighbor_axis = np.arange(NEIGHBOR_AXIS, dtype=np.int32)
    zeros = np.zeros(NEIGHBOR_AXIS, dtype=np.float32)

    # row 2 col 2: noise per neighbor
    fig.add_trace(
        go.Scatter(
            x=neighbor_axis, y=zeros, mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            name="noise (this point)", showlegend=True,
            hovertemplate="t=%{x}<br>noise=%{y:.3f}<extra>this point</extra>",
        ),
        row=2, col=2,
    )
    trace_map["noise_live"] = len(fig.data) - 1

    fig.add_trace(
        go.Scatter(
            x=neighbor_axis, y=zeros, mode="lines",
            line=dict(color="#d62728", width=2),
            opacity=0.4,
            name="noise (cluster mean)", showlegend=True,
            hovertemplate="t=%{x}<br>noise=%{y:.3f}<extra>cluster mean</extra>",
        ),
        row=2, col=2,
    )
    trace_map["noise_ref"] = len(fig.data) - 1

    # row 2 col 3: faithfulness per neighbor
    fig.add_trace(
        go.Scatter(
            x=neighbor_axis, y=zeros, mode="lines+markers",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=4),
            name="faith (this point)", showlegend=True,
            hovertemplate="t=%{x}<br>faith=%{y:.3f}<extra>this point</extra>",
        ),
        row=2, col=3,
    )
    trace_map["shared_live"] = len(fig.data) - 1

    fig.add_trace(
        go.Scatter(
            x=neighbor_axis, y=zeros, mode="lines",
            line=dict(color="#d62728", width=2),
            opacity=0.4,
            name="faith (cluster mean)", showlegend=True,
            hovertemplate="t=%{x}<br>faith=%{y:.3f}<extra>cluster mean</extra>",
        ),
        row=2, col=3,
    )
    trace_map["shared_ref"] = len(fig.data) - 1

    fig.update_xaxes(title_text="UMAP-1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP-2", row=1, col=1)
    fig.update_xaxes(title_text="feature", row=1, col=2)
    fig.update_yaxes(title_text="neighbor (t)", row=1, col=2, autorange="reversed")
    fig.update_xaxes(title_text="summary", row=1, col=3)
    fig.update_yaxes(title_text="neighbor (t)", row=1, col=3, autorange="reversed")
    fig.update_xaxes(title_text="neighbor (t)", row=2, col=2,
                     range=[-0.5, NEIGHBOR_AXIS - 0.5])
    fig.update_yaxes(title_text="noise", row=2, col=2, range=[0, 1])
    fig.update_xaxes(title_text="neighbor (t)", row=2, col=3,
                     range=[-0.5, NEIGHBOR_AXIS - 0.5])
    fig.update_yaxes(title_text="faithfulness", row=2, col=3, range=[0, 1])
    fig.update_layout(
        title="Entropy-map UMAP cluster explorer",
        width=1600, height=1100,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig, trace_map


_POST_SCRIPT_TEMPLATE = r"""
(function() {
    var gd = document.getElementById('{plot_id}');

    var POOL_LABELS_B64 = "__LABELS_B64__";
    var POOL_CLUSTER_TO_INDICES = __CLUSTER_TO_INDICES__;
    var POOL_XY = __POOL_XY__;
    var TRACE_MAP = __TRACE_MAP__;
    var CLUSTER_REFS = __CLUSTER_REFS__;  // {cid: {noise: [..50..], shared: [..50..]}} per neighbor
    var SAMPLE_PER_CLUSTER = __SAMPLE_PER_CLUSTER__;
    var ROWS = 50;
    var COLS = 64;
    var NUM_POS = 10;

    function _b64ToInt8Array(b64) {
        var bin = atob(b64);
        var len = bin.length;
        var buf = new Int8Array(len);
        for (var i = 0; i < len; i++) {
            var c = bin.charCodeAt(i);
            buf[i] = c > 127 ? c - 256 : c;
        }
        return buf;
    }
    var POOL_LABELS = _b64ToInt8Array(POOL_LABELS_B64);

    function labelMatrixForPoolIdx(poolIdx) {
        var start = poolIdx * ROWS * COLS;
        var z = new Array(ROWS);
        for (var r = 0; r < ROWS; r++) {
            var row = new Array(COLS);
            var rowStart = start + r * COLS;
            for (var c = 0; c < COLS; c++) row[c] = POOL_LABELS[rowStart + c];
            z[r] = row;
        }
        return z;
    }

    function summaryForPoolIdx(poolIdx) {
        // Per-neighbor noise and faithfulness for one (50, 64) label matrix.
        // noise[t]  = (#-1 over the 64 features at neighbor t) / 64
        // faith[t]  = (#+1 over the 64 features at neighbor t) / 10
        var start = poolIdx * ROWS * COLS;
        var noise = new Array(ROWS);
        var faith = new Array(ROWS);
        for (var r = 0; r < ROWS; r++) {
            var negCount = 0;
            var posCount = 0;
            var rowStart = start + r * COLS;
            for (var c = 0; c < COLS; c++) {
                var v = POOL_LABELS[rowStart + c];
                if (v === -1) negCount += 1;
                else if (v === 1) posCount += 1;
            }
            noise[r] = negCount / COLS;
            faith[r] = posCount / NUM_POS;
        }
        return {noise: noise, faith: faith};
    }

    function summaryHeatmapZ(summary) {
        // Build a 50x2 z array; cols are [noise, faith].
        var z = new Array(ROWS);
        for (var r = 0; r < ROWS; r++) z[r] = [summary.noise[r], summary.faith[r]];
        return z;
    }

    gd.on('plotly_hover', function(ev) {
        var pt = ev.points[0];
        if (!pt) return;
        var cd = pt.customdata;
        if (!cd) return;
        var poolIdx = Array.isArray(cd) ? cd[0] : cd;
        var cid = Array.isArray(cd) && cd.length > 1 ? cd[1] : null;
        if (poolIdx === undefined || poolIdx === null) return;

        var z = labelMatrixForPoolIdx(poolIdx);
        var summary = summaryForPoolIdx(poolIdx);
        var summaryZ = summaryHeatmapZ(summary);

        var traceIdx = [
            TRACE_MAP.heatmap,
            TRACE_MAP.summary_heatmap,
            TRACE_MAP.noise_live,
            TRACE_MAP.shared_live,
        ];
        var update = {
            z: [z, summaryZ, undefined, undefined],
            y: [undefined, undefined, summary.noise, summary.faith],
        };
        Plotly.restyle(gd, update, traceIdx);

        if (cid !== null && CLUSTER_REFS[cid] !== undefined) {
            var ref = CLUSTER_REFS[cid];
            Plotly.restyle(
                gd,
                {y: [ref.noise, ref.shared]},
                [TRACE_MAP.noise_ref, TRACE_MAP.shared_ref]
            );
        }
    });

    function shuffleAndTake(arr, n, rngState) {
        var copy = arr.slice();
        var out = [];
        var len = copy.length;
        var k = Math.min(n, len);
        for (var i = 0; i < k; i++) {
            rngState.s = (rngState.s * 1103515245 + 12345) & 0x7fffffff;
            var j = i + (rngState.s % (len - i));
            var tmp = copy[i]; copy[i] = copy[j]; copy[j] = tmp;
            out.push(copy[i]);
        }
        return out;
    }

    function resample() {
        var rngState = {s: (Date.now() & 0x7fffffff) || 1};
        var traceIndices = [];
        var update = {x: [], y: [], customdata: []};
        Object.keys(TRACE_MAP.clusters).forEach(function(cidStr) {
            var cid = parseInt(cidStr, 10);
            var pool = POOL_CLUSTER_TO_INDICES[cidStr];
            var picks = shuffleAndTake(pool, SAMPLE_PER_CLUSTER, rngState);
            var xs = picks.map(function(i) { return POOL_XY[i][0]; });
            var ys = picks.map(function(i) { return POOL_XY[i][1]; });
            var cd = picks.map(function(i) { return [i, cid]; });
            update.x.push(xs);
            update.y.push(ys);
            update.customdata.push(cd);
            traceIndices.push(TRACE_MAP.clusters[cidStr]);
        });
        Plotly.restyle(gd, update, traceIndices);
    }

    var btn = document.getElementById('resample-btn');
    if (btn) btn.addEventListener('click', resample);
})();
"""


def render_html(
    fig,
    pool: PoolData,
    trace_map: dict,
    sample_per_cluster: int,
    output_path: Path,
    include_plotlyjs: str,
) -> None:
    """Write the Plotly figure plus button + JS into a self-contained HTML file."""
    import plotly.io as pio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_b64 = base64.b64encode(pool.labels.tobytes()).decode("ascii")
    cluster_to_indices_str = json.dumps(
        {str(k): v for k, v in pool.cluster_to_pool_indices.items()}
    )
    pool_xy_str = json.dumps(pool.xy.tolist())
    trace_map_payload = {
        "clusters": {str(k): v for k, v in trace_map["clusters"].items()},
        "heatmap": trace_map["heatmap"],
        "summary_heatmap": trace_map["summary_heatmap"],
        "noise_live": trace_map["noise_live"],
        "noise_ref": trace_map["noise_ref"],
        "shared_live": trace_map["shared_live"],
        "shared_ref": trace_map["shared_ref"],
    }
    trace_map_str = json.dumps(trace_map_payload)

    cluster_refs = compute_cluster_references(pool)
    cluster_refs_payload = {
        str(cid): {
            "noise": refs["noise"].astype(float).round(6).tolist(),
            "shared": refs["shared"].astype(float).round(6).tolist(),
        }
        for cid, refs in cluster_refs.items()
    }
    cluster_refs_str = json.dumps(cluster_refs_payload)

    plot_id = "umap-explorer-plot"
    fig_html = pio.to_html(
        fig,
        include_plotlyjs=(True if include_plotlyjs == "inline" else "cdn"),
        full_html=False,
        div_id=plot_id,
    )

    post_script = (
        _POST_SCRIPT_TEMPLATE
        .replace("{plot_id}", plot_id)
        .replace('"__LABELS_B64__"', json.dumps(labels_b64))
        .replace("__CLUSTER_TO_INDICES__", cluster_to_indices_str)
        .replace("__POOL_XY__", pool_xy_str)
        .replace("__TRACE_MAP__", trace_map_str)
        .replace("__CLUSTER_REFS__", cluster_refs_str)
        .replace("__SAMPLE_PER_CLUSTER__", str(sample_per_cluster))
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Entropy-map UMAP cluster explorer</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    #controls {{ margin-bottom: 8px; }}
    button {{ font-size: 14px; padding: 6px 12px; }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="resample-btn">Resample 50 / cluster</button>
    <span style="margin-left:8px;color:#666;">Hover a colored point to see its label matrix.</span>
  </div>
  {fig_html}
  <script>{post_script}</script>
</body>
</html>
"""
    output_path.write_text(html)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entropy-maps", type=Path, default=DEFAULT_MAPS)
    p.add_argument("--label-matrices", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-fit-samples", type=int, default=50_000)
    p.add_argument("--pool-per-cluster", type=int, default=300)
    p.add_argument("--sample-per-cluster", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=200)
    p.add_argument("--hdbscan-min-samples", type=int, default=None)
    p.add_argument(
        "--include-plotlyjs",
        choices=["inline", "cdn"],
        default="inline",
        help="inline = self-contained ~3MB HTML; cdn = small file, needs internet",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading {args.entropy_maps} and {args.label_matrices} ...", flush=True)
    subset_idx, entropy_subset, labels_mmap = load_and_subsample(
        args.entropy_maps, args.label_matrices,
        n_samples=args.max_fit_samples, seed=args.seed,
    )
    print(f"  using {entropy_subset.shape[0]} / {labels_mmap.shape[0]} samples")

    print("Fitting UMAP (this can take several minutes) ...", flush=True)
    embedding = fit_umap(
        entropy_subset,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    print("Clustering with HDBSCAN ...", flush=True)
    cluster_ids = cluster_embedding(
        embedding,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
    )
    n_clusters = len({int(c) for c in cluster_ids if c != -1})
    n_noise = int((cluster_ids == -1).sum())
    print(f"  found {n_clusters} clusters, {n_noise} noise points")

    if n_clusters < 1:
        print("WARNING: HDBSCAN found no clusters. The plot will only show noise.")

    print("Building per-cluster pool ...", flush=True)
    pool = build_pool(
        cluster_ids=cluster_ids,
        embedding_2d=embedding,
        labels_mmap=labels_mmap,
        subset_idx=subset_idx,
        pool_per_cluster=args.pool_per_cluster,
        seed=args.seed + 1,
    )

    print("Building figure ...", flush=True)
    fig, trace_map = make_figure(
        embedding_2d=embedding,
        cluster_ids=cluster_ids,
        pool=pool,
        sample_per_cluster=args.sample_per_cluster,
        seed=args.seed + 2,
    )

    print(f"Writing HTML to {args.output_html} ...", flush=True)
    render_html(
        fig=fig, pool=pool, trace_map=trace_map,
        sample_per_cluster=args.sample_per_cluster,
        output_path=args.output_html,
        include_plotlyjs=args.include_plotlyjs,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
