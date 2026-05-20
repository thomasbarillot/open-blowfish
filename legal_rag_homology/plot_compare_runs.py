from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _load_metrics(run_names: list[str]) -> list[dict]:
    metrics = []
    for name in run_names:
        path = config.RUNS_DIR / name / "metrics.json"
        if not path.exists():
            logger.warning("No metrics.json for run '%s', skipping", name)
            continue
        with open(path) as f:
            m = json.load(f)
        metrics.append(m)
    return metrics


def _stacked_bar(ax, run_names: list[str], categories: list[str],
                 data: list[dict], title: str) -> None:
    x = np.arange(len(run_names))
    width = 0.6
    bottom = np.zeros(len(run_names))

    colors = {
        "accurate": "#2ca02c",
        "incomplete": "#ff7f0e",
        "hallucinated": "#d62728",
        "correct": "#2ca02c",
        "partially_correct": "#ff7f0e",
        "incorrect": "#d62728",
        "error": "#7f7f7f",
        "grounded": "#2ca02c",
        "partially_grounded": "#ff7f0e",
        "ungrounded": "#d62728",
    }

    for cat in categories:
        values = []
        for d in data:
            values.append(d.get(cat, 0))
        values = np.array(values, dtype=float)
        color = colors.get(cat, "#999999")
        ax.bar(x, values, width, bottom=bottom, label=cat, color=color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right")


def _retrieved_vs_gold_scatter(
    ax, run_names: list[str], eval_dfs: list[pd.DataFrame]
) -> None:
    """Per-question scatter of n_retrieved vs retrieval_n_gold, colored by run.

    Adds y=x reference line. Uses small jitter so coincident integer points
    don't collapse into one dot.
    """
    rng = np.random.default_rng(0)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(run_names), 1)))
    max_val = 0
    for color, name, df in zip(colors, run_names, eval_dfs):
        x = df["retrieval_n_gold"].dropna().values.astype(float)
        y = df["retrieval_n_retrieved"].dropna().values.astype(float)
        if len(x) == 0:
            continue
        jitter_x = rng.uniform(-0.15, 0.15, size=len(x))
        jitter_y = rng.uniform(-0.15, 0.15, size=len(y))
        ax.scatter(
            x + jitter_x, y + jitter_y, s=14, alpha=0.55,
            color=color, edgecolors="none", label=name,
        )
        max_val = max(max_val, float(x.max()), float(y.max()))
    if max_val > 0:
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.4, label="y = x")
    ax.set_xlabel("retrieval_n_gold")
    ax.set_ylabel("n_retrieved")
    ax.set_title("Per-question n_retrieved vs n_gold")
    ax.legend(loc="upper left", fontsize=7)


def _chunk_distribution_paired(
    ax, run_names: list[str], eval_dfs: list[pd.DataFrame]
) -> None:
    """Per-run paired boxes: n_retrieved (left) vs retrieval_n_gold (right).

    Both quantities filtered to questions with n_gold > 0 and any active
    gold_source / require_adjacent_quote selection — same subset already
    applied upstream by compare_runs.
    """
    pair_width = 0.36
    inner = 0.18
    positions_left, positions_right = [], []
    data_retrieved, data_gold = [], []
    for i, df in enumerate(eval_dfs):
        positions_left.append(i - inner)
        positions_right.append(i + inner)
        data_retrieved.append(df["retrieval_n_retrieved"].dropna().values)
        data_gold.append(df["retrieval_n_gold"].dropna().values)

    bp_r = ax.boxplot(
        data_retrieved, positions=positions_left, widths=pair_width,
        patch_artist=True, showfliers=False, medianprops={"color": "black"},
    )
    bp_g = ax.boxplot(
        data_gold, positions=positions_right, widths=pair_width,
        patch_artist=True, showfliers=False, medianprops={"color": "black"},
    )
    for patch in bp_r["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)
    for patch in bp_g["boxes"]:
        patch.set_facecolor("#2ca02c")
        patch.set_alpha(0.7)

    ax.set_xticks(np.arange(len(run_names)))
    ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("Chunks per question")
    ax.set_title("Retrieved vs gold chunks per question (n_gold > 0)")
    ax.legend(
        [bp_r["boxes"][0], bp_g["boxes"][0]],
        ["n_retrieved", "retrieval_n_gold"],
        loc="upper right",
    )


def _chunk_distribution_box(ax, run_names: list[str], eval_dfs: list[pd.DataFrame]) -> None:
    data_retrieved = []
    for df in eval_dfs:
        has_gold = df["retrieval_n_gold"] > 0
        data_retrieved.append(df.loc[has_gold, "retrieval_n_retrieved"].dropna().values)

    gold_chunks = eval_dfs[0].loc[eval_dfs[0]["retrieval_n_gold"] > 0, "retrieval_n_gold"].values

    x = np.arange(len(run_names))
    bp = ax.boxplot(
        data_retrieved,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)

    gold_median = np.median(gold_chunks)
    gold_q1, gold_q3 = np.percentile(gold_chunks, [25, 75])
    gold_line = ax.axhline(gold_median, color="#2ca02c", linewidth=2, linestyle="-")
    ax.axhspan(gold_q1, gold_q3, color="#2ca02c", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("Chunks per question")
    ax.set_title("Retrieved chunks vs available gold chunks")
    ax.legend(
        [bp["boxes"][0], gold_line],
        ["Retrieved", "Gold (median + IQR)"],
        loc="upper right",
    )


def _retrieval_grouped_bar(ax, run_names: list[str], metrics: list[dict]) -> None:
    """Stacked P/R/F1 bars: strict (solid) + +/-1 neighbor delta (hatched).

    Neighbor-tolerant metrics are >= strict (same denominator, larger TP),
    so the stack heights correspond to the relaxed score.
    """
    x = np.arange(len(run_names))
    width = 0.25

    p_strict, r_strict, f_strict = [], [], []
    p_neigh, r_neigh, f_neigh = [], [], []
    for m in metrics:
        rm = m.get("retrieval_chunk_metrics", {})
        p_s = rm.get("macro_precision", 0.0)
        r_s = rm.get("macro_recall", 0.0)
        f_s = rm.get("macro_f1", 0.0)
        p_n = rm.get("macro_precision_at_neighbor_1", p_s)
        r_n = rm.get("macro_recall_at_neighbor_1", r_s)
        f_n = rm.get("macro_f1_at_neighbor_1", f_s)
        p_strict.append(p_s)
        r_strict.append(r_s)
        f_strict.append(f_s)
        p_neigh.append(max(0.0, p_n - p_s))
        r_neigh.append(max(0.0, r_n - r_s))
        f_neigh.append(max(0.0, f_n - f_s))

    ax.bar(x - width, p_strict, width, label="Precision (strict)", color="#1f77b4")
    ax.bar(
        x - width, p_neigh, width, bottom=p_strict, color="#1f77b4",
        alpha=0.45, hatch="//", edgecolor="white", label="+/-1 neighbor delta",
    )
    ax.bar(x, r_strict, width, label="Recall (strict)", color="#ff7f0e")
    ax.bar(
        x, r_neigh, width, bottom=r_strict, color="#ff7f0e",
        alpha=0.45, hatch="//", edgecolor="white",
    )
    ax.bar(x + width, f_strict, width, label="F1 (strict)", color="#2ca02c")
    ax.bar(
        x + width, f_neigh, width, bottom=f_strict, color="#2ca02c",
        alpha=0.45, hatch="//", edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval P/R/F1 (strict + +/-1 neighbor)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1)


def _retrieval_by_category(
    ax,
    run_names: list[str],
    metrics: list[dict],
    metric_key: str,
    title: str,
    neighbor_key: str | None = None,
) -> None:
    """Per-category bars for a strict retrieval metric, with an optional
    +/-1 neighbor delta stacked on top (hatched).
    """
    all_categories = set()
    for m in metrics:
        for entry in m.get("by_category", []):
            if metric_key in entry:
                all_categories.add(entry["category"])
    categories = sorted(all_categories)
    if not categories:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return

    n_runs = len(run_names)
    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.8 / max(n_runs, 1)

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_runs, 1)))

    has_neighbor_overlay = False
    for i, (name, m) in enumerate(zip(run_names, metrics)):
        by_cat = {e["category"]: e for e in m.get("by_category", [])}
        strict_values = [by_cat.get(cat, {}).get(metric_key, 0.0) for cat in categories]
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, strict_values, width, label=name, color=colors[i])

        if neighbor_key is None:
            continue
        deltas = []
        for cat, strict_v in zip(categories, strict_values):
            neighbor_v = by_cat.get(cat, {}).get(neighbor_key, strict_v)
            deltas.append(max(0.0, neighbor_v - strict_v))
        if any(d > 0 for d in deltas):
            has_neighbor_overlay = True
            ax.bar(
                x + offset, deltas, width, bottom=strict_values,
                color=colors[i], alpha=0.45, hatch="//", edgecolor="white",
            )

    handles, labels = ax.get_legend_handles_labels()
    if has_neighbor_overlay:
        from matplotlib.patches import Patch
        handles = list(handles) + [
            Patch(facecolor="white", edgecolor="black", hatch="//", label="+/-1 neighbor delta")
        ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)


def _category_grouped_bar(ax, run_names: list[str], metrics: list[dict]) -> None:
    all_categories = set()
    for m in metrics:
        for entry in m.get("by_category", []):
            all_categories.add(entry["category"])
    categories = sorted(all_categories)

    n_runs = len(run_names)
    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.8 / max(n_runs, 1)

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_runs, 1)))

    for i, (name, m) in enumerate(zip(run_names, metrics)):
        by_cat = {e["category"]: e for e in m.get("by_category", [])}
        halluc_rates = []
        for cat in categories:
            entry = by_cat.get(cat, {})
            halluc_rates.append(entry.get("hallucinated", 0.0))
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, halluc_rates, width, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Hallucination rate by category")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)


def _verdict_correlation(fig, axes_row, run_names: list[str], eval_dfs: list[pd.DataFrame]) -> None:
    verdict_map = {"accurate": 2, "incomplete": 1, "hallucinated": 0}
    n_runs = len(run_names)

    merged = eval_dfs[0][["question_id"]].copy()
    for name, df in zip(run_names, eval_dfs):
        merged = merged.merge(
            df[["question_id", "verdict"]].rename(columns={"verdict": name}),
            on="question_id", how="inner",
        )

    n_q = len(merged)
    if n_q == 0:
        for ax in axes_row:
            ax.axis("off")
        return

    # Agreement matrix: fraction of questions where both runs give same verdict
    agreement = np.zeros((n_runs, n_runs))
    for i, ri in enumerate(run_names):
        for j, rj in enumerate(run_names):
            agreement[i, j] = (merged[ri] == merged[rj]).mean()

    im = axes_row[0].imshow(agreement, vmin=0, vmax=1, cmap="RdYlGn")
    axes_row[0].set_xticks(range(n_runs))
    axes_row[0].set_yticks(range(n_runs))
    axes_row[0].set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
    axes_row[0].set_yticklabels(run_names, fontsize=8)
    axes_row[0].set_title(f"Verdict agreement ({n_q} questions)")
    for i in range(n_runs):
        for j in range(n_runs):
            axes_row[0].text(j, i, f"{agreement[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes_row[0], fraction=0.046)

    # Per-question verdict matrix: rows=questions, cols=runs, color=verdict
    score_matrix = np.zeros((n_q, n_runs))
    for i, name in enumerate(run_names):
        score_matrix[:, i] = merged[name].map(verdict_map).values

    sort_idx = np.lexsort(score_matrix.T)
    score_matrix = score_matrix[sort_idx]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d62728", "#ff7f0e", "#2ca02c"])
    im2 = axes_row[1].imshow(score_matrix.T, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes_row[1].set_yticks(range(n_runs))
    axes_row[1].set_yticklabels(run_names, fontsize=8)
    axes_row[1].set_xlabel("Questions (sorted)")
    axes_row[1].set_title("Per-question verdicts")
    cbar = fig.colorbar(im2, ax=axes_row[1], ticks=[0, 1, 2], fraction=0.046)
    cbar.ax.set_yticklabels(["hallucinated", "incomplete", "accurate"])

    # Complementarity: for each pair, fraction of questions where one is accurate and the other is not
    complementarity = np.zeros((n_runs, n_runs))
    for i, ri in enumerate(run_names):
        for j, rj in enumerate(run_names):
            if i == j:
                continue
            i_good = merged[ri] == "accurate"
            j_bad = merged[rj] != "accurate"
            complementarity[i, j] = (i_good & j_bad).mean()

    im3 = axes_row[2].imshow(complementarity, vmin=0, cmap="Blues")
    axes_row[2].set_xticks(range(n_runs))
    axes_row[2].set_yticks(range(n_runs))
    axes_row[2].set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
    axes_row[2].set_yticklabels(run_names, fontsize=8)
    axes_row[2].set_title("Complementarity (row accurate, col not)")
    for i in range(n_runs):
        for j in range(n_runs):
            axes_row[2].text(j, i, f"{complementarity[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im3, ax=axes_row[2], fraction=0.046)


def _compute_ranking_comparison(questions_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run retrieval (no LLM) for each question and collect per-chunk ranks from both methods.

    Returns DataFrame with one row per chunk:
      - question_id, chunk_id
      - hybrid_rank: position in BM25+dense RRF ranking (0-indexed)
      - homology_rank: position in homology distance ordering (0-indexed)
      - shared_features: predicted shared feature count for that position
      - extra_features: predicted non-query feature count for that position
    """
    import torch

    from .retrieval.bm25_retriever import BM25Retriever
    from .retrieval.chunk_store import ChunkStore
    from .retrieval.dense_retriever import DenseRetriever
    from .retrieval.homology_retriever import _homology_map_to_entropy_map, _load_feature_predictor
    from .retrieval.hybrid_retriever import HybridRetriever

    from scripts.homology_simulation.homology_calculation import calculate_homology0_map

    bm25 = BM25Retriever.load(config.INDEX_DIR / "bm25", config.CORPUS_DIR / "chunks.parquet")
    dense = DenseRetriever.load(config.INDEX_DIR / "dense.faiss", config.CORPUS_DIR / "chunks.parquet")
    hybrid = HybridRetriever(bm25, dense, k_candidates=config.BM25_TOPK, rrf_k=config.RRF_K)
    chunk_store = ChunkStore()
    device = torch.device("cpu")
    predictor = _load_feature_predictor(config.FEATURE_PREDICTOR_MODEL, device)

    rows = []
    for _, q in questions_df.iterrows():
        query = q["question_text"]
        qid = q["question_id"]

        ranked_scores, ranked_chunks, _ = hybrid._get_and_rerank(query, 50)
        if not ranked_chunks:
            continue

        hybrid_rank_map = {c.chunk_id: i for i, (_, c) in enumerate(zip(ranked_scores, ranked_chunks))}

        chunk_ids = [c.chunk_id for c in ranked_chunks]
        chunk_embeds = chunk_store.get_embeddings(chunk_ids)
        q_embed = dense._encode(query).squeeze(0)

        homology_map, sorting_indices = calculate_homology0_map(
            query_embed=q_embed, corpus_embeds=chunk_embeds
        )
        entropy_map = _homology_map_to_entropy_map(homology_map)

        if entropy_map is None:
            continue

        from scripts.ottp_topology_analysis.train_feature_predictor_v2 import MAX_FEATURES
        from scripts.ottp_topology_analysis.train_feature_predictor_v4 import NUM_POS_FEATURES

        summary = predictor.predict(entropy_map)                  # (T, 2)
        t = min(summary.shape[0], len(ranked_chunks))
        shared_counts = summary[:t, 1] * float(NUM_POS_FEATURES)
        extra_counts = summary[:t, 0] * float(MAX_FEATURES)

        for homology_pos in range(t):
            original_idx = int(sorting_indices[homology_pos])
            chunk_id = chunk_ids[original_idx]
            rows.append({
                "question_id": qid,
                "chunk_id": chunk_id,
                "hybrid_rank": hybrid_rank_map[chunk_id],
                "homology_rank": homology_pos,
                "shared_features": float(shared_counts[homology_pos]),
                "extra_features": float(extra_counts[homology_pos]),
            })

    return pd.DataFrame(rows)


def _ranking_correlation_scatter(fig, axes_row, run_names: list[str]) -> None:
    """Plot per-chunk rank correlation between hybrid and homology orderings.

    axes_row should have 2 axes:
      [0]: scatter of hybrid_rank vs homology_rank, colored by chunk quality
      [1]: per-question Spearman correlation histogram
    """
    from scipy.stats import spearmanr

    cache_path = config.PLOTS_DIR / "ranking_comparison.parquet"
    if cache_path.exists():
        comp_df = pd.read_parquet(cache_path)
    else:
        questions_path = config.DATASET_DIR / "questions.parquet"
        if not questions_path.exists():
            for ax in axes_row:
                ax.set_title("Ranking correlation (no questions.parquet)")
                ax.axis("off")
            return
        questions_df = pd.read_parquet(questions_path)
        comp_df = _compute_ranking_comparison(questions_df)
        comp_df.to_parquet(cache_path, index=False)
        logger.info("Cached ranking comparison to %s", cache_path)

    if comp_df.empty:
        for ax in axes_row:
            ax.set_title("Ranking correlation (no data)")
            ax.axis("off")
        return

    # Classify each chunk: noisy if extra_features > median, incomplete if shared < median
    med_shared = comp_df["shared_features"].median()
    med_extra = comp_df["extra_features"].median()

    conditions = [
        (comp_df["extra_features"] > med_extra) & (comp_df["shared_features"] >= med_shared),
        (comp_df["shared_features"] < med_shared) & (comp_df["extra_features"] <= med_extra),
        (comp_df["extra_features"] > med_extra) & (comp_df["shared_features"] < med_shared),
    ]
    colors = np.where(
        conditions[0], "#d62728",   # noisy
        np.where(conditions[1], "#1f77b4",   # incomplete
                 np.where(conditions[2], "#9467bd",  # both
                          "#7f7f7f"))  # balanced
    )

    ax = axes_row[0]
    ax.scatter(
        comp_df["hybrid_rank"], comp_df["homology_rank"],
        c=colors, s=12, alpha=0.5, edgecolors="none",
    )
    max_rank = max(comp_df["hybrid_rank"].max(), comp_df["homology_rank"].max())
    ax.plot([0, max_rank], [0, max_rank], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Hybrid rank (BM25 + dense RRF)")
    ax.set_ylabel("Homology rank (distance order)")
    ax.set_title("Per-chunk rank: Hybrid vs Homology")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="Noisy (high extra features)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Incomplete (low shared features)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467bd", markersize=8, label="Both"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=8, label="Balanced"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7)

    # Per-question Spearman correlation
    ax2 = axes_row[1]
    per_q_rho = []
    for qid, grp in comp_df.groupby("question_id"):
        if len(grp) < 3:
            continue
        rho, _ = spearmanr(grp["hybrid_rank"], grp["homology_rank"])
        per_q_rho.append(rho)

    if per_q_rho:
        per_q_rho = np.array(per_q_rho)
        ax2.hist(per_q_rho, bins=20, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax2.axvline(np.mean(per_q_rho), color="#d62728", linewidth=2, linestyle="--",
                    label=f"Mean: {np.mean(per_q_rho):.3f}")
        ax2.axvline(np.median(per_q_rho), color="#ff7f0e", linewidth=2, linestyle="-",
                    label=f"Median: {np.median(per_q_rho):.3f}")
        ax2.set_xlabel("Spearman rho (per question)")
        ax2.set_ylabel("Count")
        ax2.set_title("Rank correlation distribution")
        ax2.legend(loc="upper left", fontsize=8)
    else:
        ax2.set_title("Rank correlation (insufficient data)")
        ax2.axis("off")


def _load_evaluations(run_names: list[str]) -> tuple[list[str], list[pd.DataFrame]]:
    valid = []
    dfs = []
    for name in run_names:
        path = config.RUNS_DIR / name / "evaluation.parquet"
        if path.exists():
            valid.append(name)
            dfs.append(pd.read_parquet(path))
    return valid, dfs


def _recompute_metrics_from_eval(metric: dict, eval_df: pd.DataFrame) -> dict:
    """Rebuild metrics.json-shaped aggregates from a (filtered) eval_df.

    Used after dropping rows with retrieval_n_gold == 0 so the verdict /
    correctness / groundedness / category panels show only questions that
    actually had gold chunks available.
    """
    out = {**metric, "n_questions": len(eval_df)}
    if len(eval_df) == 0:
        out["verdict"] = {"accurate": 0, "incomplete": 0, "hallucinated": 0}
        out["correctness"] = {}
        out["groundedness"] = {}
        out["by_category"] = []
        out["retrieval_chunk_metrics"] = {}
        return out

    if "retrieval_precision" in eval_df.columns:
        scored = eval_df.dropna(subset=["retrieval_precision"])
        if len(scored) > 0:
            chunk_metrics = {
                "macro_precision": float(scored["retrieval_precision"].mean()),
                "macro_recall": float(scored["retrieval_recall"].mean()),
                "macro_f1": float(scored["retrieval_f1"].mean()),
                "mean_n_retrieved": float(scored["retrieval_n_retrieved"].mean()),
                "mean_n_gold": float(scored["retrieval_n_gold"].mean()),
                "n_questions_with_gold": len(scored),
            }
            if "retrieval_precision_at_neighbor_1" in scored.columns:
                chunk_metrics["macro_precision_at_neighbor_1"] = float(
                    scored["retrieval_precision_at_neighbor_1"].mean()
                )
                chunk_metrics["macro_recall_at_neighbor_1"] = float(
                    scored["retrieval_recall_at_neighbor_1"].mean()
                )
                chunk_metrics["macro_f1_at_neighbor_1"] = float(
                    scored["retrieval_f1_at_neighbor_1"].mean()
                )
            if "retrieval_tp" in scored.columns:
                chunk_metrics["mean_tp"] = float(scored["retrieval_tp"].mean())
            if "retrieval_tp_neighbor" in scored.columns:
                chunk_metrics["mean_tp_neighbor"] = float(
                    scored["retrieval_tp_neighbor"].mean()
                )
            out["retrieval_chunk_metrics"] = chunk_metrics

    out["verdict"] = {
        v: int((eval_df["verdict"] == v).sum())
        for v in ("accurate", "incomplete", "hallucinated")
    }
    out["correctness"] = {k: int(v) for k, v in eval_df["correctness"].value_counts().items()}
    out["groundedness"] = {k: int(v) for k, v in eval_df["groundedness"].value_counts().items()}

    by_cat = []
    for cat, grp in eval_df.groupby("category"):
        n = len(grp)
        entry = {
            "category": cat,
            "n": n,
            "accurate": float((grp["verdict"] == "accurate").mean()),
            "incomplete": float((grp["verdict"] == "incomplete").mean()),
            "hallucinated": float((grp["verdict"] == "hallucinated").mean()),
            "precision": float(grp["precision"].mean()),
            "recall": float(grp["recall"].mean()),
            "f1": float(grp["f1"].mean()),
        }
        if "retrieval_precision" in grp.columns:
            entry["retrieval_precision"] = float(grp["retrieval_precision"].mean())
            entry["retrieval_recall"] = float(grp["retrieval_recall"].mean())
            entry["retrieval_f1"] = float(grp["retrieval_f1"].mean())
            if "retrieval_precision_at_neighbor_1" in grp.columns:
                entry["retrieval_precision_at_neighbor_1"] = float(
                    grp["retrieval_precision_at_neighbor_1"].mean()
                )
                entry["retrieval_recall_at_neighbor_1"] = float(
                    grp["retrieval_recall_at_neighbor_1"].mean()
                )
                entry["retrieval_f1_at_neighbor_1"] = float(
                    grp["retrieval_f1_at_neighbor_1"].mean()
                )
        by_cat.append(entry)
    out["by_category"] = by_cat
    return out


_GOLD_SOURCE_CHOICES = ("all", "primary-only", "paraphrase-only")


def _plot_features_by_gold_source(suffix: str = "") -> None:
    """Emit two PNGs of shared/extra features along homology_rank, split by
    whether the qid was grounded via primary (LLM-validated quote -> chunk) or
    only via the paraphrase fallback. Per-qid lines drawn at low alpha; bold
    mean overlaid.

    Reads the cached ranking_comparison.parquet (produced by the existing
    _compute_ranking_comparison path); skips silently if the cache is absent.
    No-op for qids that lack a gold_source entry.
    """
    cache_path = config.PLOTS_DIR / "ranking_comparison.parquet"
    if not cache_path.exists():
        logger.info(
            "ranking_comparison.parquet not found; skipping features-by-gold-source plot"
        )
        return

    from .run_evaluate import _build_gold_chunk_sets

    _, _, gold_source_by_q, _ = _build_gold_chunk_sets()
    comp = pd.read_parquet(cache_path)
    comp = comp[comp["question_id"].isin(gold_source_by_q.keys())]
    if comp.empty:
        logger.info("No overlap between ranking cache and gold_source qids; skipping")
        return

    comp = comp.assign(gold_source=comp["question_id"].map(gold_source_by_q))

    for source_label, source_value in [("primary", "primary"), ("paraphrase", "paraphrase_only")]:
        sub = comp[comp["gold_source"] == source_value]
        if sub.empty:
            logger.info("No qids in %s bucket; skipping", source_label)
            continue
        n_qids = sub["question_id"].nunique()

        fig, ax = plt.subplots(figsize=(10, 6))
        for color, feature_col, label in [
            ("#1f77b4", "shared_features", "shared features"),
            ("#d62728", "extra_features", "extra (noise) features"),
        ]:
            for _, grp in sub.groupby("question_id"):
                grp = grp.sort_values("homology_rank")
                ax.plot(
                    grp["homology_rank"].values, grp[feature_col].values,
                    color=color, alpha=0.10, linewidth=0.8,
                )
            mean_curve = (
                sub.groupby("homology_rank")[feature_col].mean().sort_index()
            )
            ax.plot(
                mean_curve.index.values, mean_curve.values,
                color=color, linewidth=2.5, label=f"mean {label}",
            )

        ax.set_xlabel("homology_rank")
        ax.set_ylabel("predicted feature count")
        ax.set_title(
            f"Shared vs noise features by homology rank — "
            f"{source_label} qids (n={n_qids})"
        )
        ax.legend(loc="upper right")
        out_path = config.PLOTS_DIR / f"features_by_gold_source_{source_label}{suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved features plot to {out_path}")


def _filter_by_gold_source(
    eval_dfs: list[pd.DataFrame], run_names: list[str], gold_source: str
) -> list[pd.DataFrame]:
    if gold_source == "all":
        return eval_dfs
    out = []
    for name, df in zip(run_names, eval_dfs):
        if "gold_source" not in df.columns:
            logger.warning(
                "%s: evaluation.parquet has no 'gold_source' column; "
                "re-run with --patch-retrieval to enable --gold-source filtering. "
                "Falling back to unfiltered.",
                name,
            )
            out.append(df)
            continue
        keep = "primary" if gold_source == "primary-only" else "paraphrase_only"
        before = len(df)
        df = df[df["gold_source"] == keep].reset_index(drop=True)
        logger.info("%s: gold_source=%s filtered %d -> %d", name, gold_source, before, len(df))
        out.append(df)
    return out


_CITE_CLASS_CHOICES = ("federal", "state", "lexis", "specialty", "unknown")


def _filter_by_cite_class(
    eval_dfs: list[pd.DataFrame],
    run_names: list[str],
    require: tuple[str, ...],
    exclude: tuple[str, ...],
) -> list[pd.DataFrame]:
    """Filter qids by membership of their gold cite_classes.

    require: keep qids that have at least one cite_class in this set (empty -> no constraint).
    exclude: drop qids whose ANY cite_class is in this set.
    """
    if not require and not exclude:
        return eval_dfs
    out = []
    for name, df in zip(run_names, eval_dfs):
        if "cite_classes" not in df.columns:
            logger.warning(
                "%s: evaluation.parquet has no 'cite_classes' column; "
                "re-run --patch-retrieval to enable --cite-class filtering. "
                "Falling back to unfiltered.",
                name,
            )
            out.append(df)
            continue
        before = len(df)
        keep = []
        for classes in df["cite_classes"].values:
            cs = set(classes) if classes is not None else set()
            if require and not (cs & set(require)):
                keep.append(False)
                continue
            if exclude and (cs & set(exclude)):
                keep.append(False)
                continue
            keep.append(True)
        df = df[keep].reset_index(drop=True)
        logger.info(
            "%s: cite_class require=%s exclude=%s filtered %d -> %d",
            name, require, exclude, before, len(df),
        )
        out.append(df)
    return out


def _filter_by_adjacent_quote(
    eval_dfs: list[pd.DataFrame], run_names: list[str]
) -> list[pd.DataFrame]:
    out = []
    for name, df in zip(run_names, eval_dfs):
        if "has_adjacent_quote" not in df.columns:
            logger.warning(
                "%s: evaluation.parquet has no 'has_adjacent_quote' column; "
                "re-run with --patch-retrieval to enable --require-adjacent-quote. "
                "Falling back to unfiltered.",
                name,
            )
            out.append(df)
            continue
        before = len(df)
        df = df[df["has_adjacent_quote"]].reset_index(drop=True)
        logger.info(
            "%s: require_adjacent_quote filtered %d -> %d", name, before, len(df)
        )
        out.append(df)
    return out


def compare_runs(
    run_names: list[str],
    output_path: Path | None = None,
    gold_source: str = "all",
    require_adjacent_quote: bool = False,
    require_cite_classes: tuple[str, ...] = (),
    exclude_cite_classes: tuple[str, ...] = (),
) -> None:
    metrics = _load_metrics(run_names)
    if not metrics:
        print("No metrics found for any of the specified runs.")
        return

    valid_names = [m["run_name"] for m in metrics]
    _, eval_dfs = _load_evaluations(valid_names)

    if eval_dfs:
        before = [len(df) for df in eval_dfs]
        eval_dfs = [df[df["retrieval_n_gold"] > 0].reset_index(drop=True) for df in eval_dfs]
        after = [len(df) for df in eval_dfs]
        for name, b, a in zip(valid_names, before, after):
            logger.info("%s: filtered %d -> %d questions (dropped n_gold==0)", name, b, a)
        eval_dfs = _filter_by_gold_source(eval_dfs, valid_names, gold_source)
        if require_adjacent_quote:
            eval_dfs = _filter_by_adjacent_quote(eval_dfs, valid_names)
        if require_cite_classes or exclude_cite_classes:
            eval_dfs = _filter_by_cite_class(
                eval_dfs, valid_names, require_cite_classes, exclude_cite_classes
            )
        metrics = [_recompute_metrics_from_eval(m, df) for m, df in zip(metrics, eval_dfs)]

    fig, axes = plt.subplots(5, 3, figsize=(24, 30))

    _stacked_bar(
        axes[0, 0], valid_names,
        ["accurate", "incomplete", "hallucinated"],
        [m["verdict"] for m in metrics],
        "Verdict distribution (n_gold>0)",
    )

    _stacked_bar(
        axes[0, 1], valid_names,
        ["correct", "partially_correct", "incorrect", "error"],
        [m["correctness"] for m in metrics],
        "Correctness (LLM judge, n_gold>0)",
    )

    _stacked_bar(
        axes[0, 2], valid_names,
        ["grounded", "partially_grounded", "ungrounded"],
        [m["groundedness"] for m in metrics],
        "Groundedness (citation matching, n_gold>0)",
    )

    _retrieval_grouped_bar(axes[1, 0], valid_names, metrics)

    _category_grouped_bar(axes[1, 1], valid_names, metrics)

    if eval_dfs:
        _chunk_distribution_box(axes[1, 2], valid_names, eval_dfs)
    else:
        axes[1, 2].axis("off")

    _retrieval_by_category(
        axes[2, 0], valid_names, metrics,
        "retrieval_precision", "Retrieval precision by category",
        neighbor_key="retrieval_precision_at_neighbor_1",
    )
    _retrieval_by_category(
        axes[2, 1], valid_names, metrics,
        "retrieval_recall", "Retrieval recall by category",
        neighbor_key="retrieval_recall_at_neighbor_1",
    )
    _retrieval_by_category(
        axes[2, 2], valid_names, metrics,
        "retrieval_f1", "Retrieval F1 by category",
        neighbor_key="retrieval_f1_at_neighbor_1",
    )

    if eval_dfs:
        _verdict_correlation(fig, axes[3], valid_names, eval_dfs)
    else:
        for ax in axes[3]:
            ax.axis("off")

    _ranking_correlation_scatter(fig, axes[4, :2], valid_names)
    axes[4, 2].axis("off")

    title_tags = []
    if gold_source != "all":
        title_tags.append(gold_source)
    if require_adjacent_quote:
        title_tags.append("has adjacent_quote")
    if require_cite_classes:
        title_tags.append("require " + "+".join(require_cite_classes))
    if exclude_cite_classes:
        title_tags.append("exclude " + "+".join(exclude_cite_classes))
    suptitle = "Run comparison" + (f" ({', '.join(title_tags)})" if title_tags else "")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()

    suffix_parts = []
    if gold_source != "all":
        suffix_parts.append(gold_source.replace("-", "_"))
    if require_adjacent_quote:
        suffix_parts.append("has_quote")
    if require_cite_classes:
        suffix_parts.append("req_" + "_".join(require_cite_classes))
    if exclude_cite_classes:
        suffix_parts.append("excl_" + "_".join(exclude_cite_classes))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    if output_path is None:
        output_path = config.PLOTS_DIR / f"run_comparison{suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")

    if eval_dfs:
        fig_pair, axes_pair = plt.subplots(
            1, 2, figsize=(max(14, 1.6 * len(valid_names) + 8), 5)
        )
        _chunk_distribution_paired(axes_pair[0], valid_names, eval_dfs)
        _retrieved_vs_gold_scatter(axes_pair[1], valid_names, eval_dfs)
        pair_title = "Retrieved vs gold chunks"
        if title_tags:
            pair_title = f"{pair_title} ({', '.join(title_tags)})"
        fig_pair.suptitle(pair_title, fontsize=12)
        fig_pair.tight_layout()
        pair_path = config.PLOTS_DIR / f"retrieved_vs_gold{suffix}.png"
        fig_pair.savefig(pair_path, dpi=150, bbox_inches="tight")
        plt.close(fig_pair)
        print(f"Saved retrieved-vs-gold plot to {pair_path}")

    if gold_source == "all" and not require_adjacent_quote and not require_cite_classes and not exclude_cite_classes:
        # Features-by-gold-source plot is a property of the gold set, not a run
        # filter; only emit on the unfiltered pass to avoid duplicate writes.
        _plot_features_by_gold_source()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Compare evaluation metrics across runs")
    p.add_argument("runs", nargs="+", help="Run names to compare")
    p.add_argument("--output", type=str, default=None, help="Output path for plot")
    p.add_argument(
        "--gold-source",
        choices=_GOLD_SOURCE_CHOICES,
        default="all",
        help=(
            "Filter questions by how their gold chunks were obtained: 'primary-only' "
            "keeps qids matched via adjacent_quote, 'paraphrase-only' keeps qids that "
            "only matched via the paraphrase fallback. Default 'all' keeps both."
        ),
    )
    p.add_argument(
        "--all-gold-sources",
        action="store_true",
        help="Generate three plots in one run: all, primary-only, paraphrase-only.",
    )
    p.add_argument(
        "--require-adjacent-quote",
        action="store_true",
        help=(
            "Drop questions whose source gold rows had no adjacent_quote at all "
            "(i.e. they only got grounded via the paraphrase fallback because the "
            "released answer didn't include a verbatim quote)."
        ),
    )
    p.add_argument(
        "--require-cite-class",
        action="append",
        choices=_CITE_CLASS_CHOICES,
        default=[],
        help=(
            "Keep only qids whose gold cite_classes contain at least one of the "
            "given values. Repeat to allow multiple. E.g. --require-cite-class state."
        ),
    )
    p.add_argument(
        "--exclude-cite-class",
        action="append",
        choices=_CITE_CLASS_CHOICES,
        default=[],
        help=(
            "Drop qids whose gold cite_classes contain ANY of the given values. "
            "E.g. --exclude-cite-class lexis to exclude vendor-only cites."
        ),
    )
    args = p.parse_args()

    output = Path(args.output) if args.output else None
    require_cc = tuple(args.require_cite_class)
    exclude_cc = tuple(args.exclude_cite_class)
    if args.all_gold_sources:
        if output is not None:
            logger.warning("--output ignored when --all-gold-sources is set")
        for choice in _GOLD_SOURCE_CHOICES:
            compare_runs(
                args.runs,
                output_path=None,
                gold_source=choice,
                require_adjacent_quote=args.require_adjacent_quote,
                require_cite_classes=require_cc,
                exclude_cite_classes=exclude_cc,
            )
    else:
        compare_runs(
            args.runs,
            output_path=output,
            gold_source=args.gold_source,
            require_adjacent_quote=args.require_adjacent_quote,
            require_cite_classes=require_cc,
            exclude_cite_classes=exclude_cc,
        )


if __name__ == "__main__":
    main()
