from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.homology_simulation.homology_calculation import calculate_homology0_map
from scripts.ottp_topology_analysis.train_feature_predictor_v4 import (
    NUM_POS_FEATURES,
    EntropyEncoder,
    SummaryHead,
)
from scripts.ottp_topology_analysis.train_feature_predictor_v2 import (
    MAX_FEATURES,
    N_H0_BINS,
)

from .. import config
from .base import RetrievedChunk, Retriever
from .chunk_store import ChunkStore
from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_H0_BINS = np.linspace(0.0, 1.5, N_H0_BINS + 1)


def _homology_map_to_entropy_map(homology_map: list) -> np.ndarray | None:
    """Convert list of h0_dist arrays into a row-normalized (T, 20) entropy map."""
    n = len(homology_map)
    if n == 0:
        return None
    raw = np.zeros((n, N_H0_BINS), dtype=np.float32)
    for i, h0_dist in enumerate(homology_map):
        if not isinstance(h0_dist, (list, np.ndarray)):
            continue
        arr = np.asarray(h0_dist, dtype=np.float64)
        if arr.size == 0:
            continue
        h, _ = np.histogram(arr, bins=_H0_BINS)
        raw[i] = h.astype(np.float32)

    if raw.sum() < 1.0:
        return None

    last_nonzero = (raw.sum(axis=1) > 0).nonzero()[0]
    if len(last_nonzero) == 0:
        return None
    trimmed = raw[: int(last_nonzero[-1]) + 1]
    row_sums = trimmed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return (trimmed / row_sums).astype(np.float32)


def _build_raw_histogram(homology_map: list) -> np.ndarray:
    """Build unnormalized (T, 20) histogram from raw h0_dist arrays."""
    n = len(homology_map)
    raw = np.zeros((n, N_H0_BINS), dtype=np.float32)
    for i, h0_dist in enumerate(homology_map):
        if not isinstance(h0_dist, (list, np.ndarray)):
            continue
        arr = np.asarray(h0_dist, dtype=np.float64)
        if arr.size == 0:
            continue
        h, _ = np.histogram(arr, bins=_H0_BINS)
        raw[i] = h.astype(np.float32)
    return raw


def _plot_entropy_and_summary(
    homology_map: list,
    entropy_map: np.ndarray,
    summary: np.ndarray,
    question_id: str,
    query_short: str,
) -> None:
    """Per-query diagnostic plot. summary is (T, 2) in [0, 1]: [noise, faith]."""
    raw_hist = _build_raw_histogram(homology_map)
    t = entropy_map.shape[0]
    summary = summary[:t]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(28, 6))

    tick_pos = np.arange(0, N_H0_BINS, 4)

    im0 = ax0.imshow(raw_hist[:t].T, aspect="auto", origin="lower", cmap="hot")
    ax0.set_xlabel("Neighbor position")
    ax0.set_ylabel("H0 birth bin")
    ax0.set_title("Homology map (unnormalized counts)")
    ax0.set_yticks(tick_pos)
    ax0.set_yticklabels([f"{_H0_BINS[i]:.2f}" for i in tick_pos])
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    im1 = ax1.imshow(entropy_map.T, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_xlabel("Neighbor position")
    ax1.set_ylabel("H0 birth bin")
    ax1.set_title("Entropy map (row-normalized)")
    ax1.set_yticks(tick_pos)
    ax1.set_yticklabels([f"{_H0_BINS[i]:.2f}" for i in tick_pos])
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(
        summary, aspect="auto", origin="lower",
        cmap="viridis", vmin=0, vmax=1,
    )
    ax2.set_xlabel("Summary channel")
    ax2.set_ylabel("Neighbor position")
    ax2.set_title("Predicted summary (noise / faithfulness)")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["noise", "faith"])
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    faithfulness = summary[:, 1] * float(NUM_POS_FEATURES)
    noisiness = summary[:, 0] * float(MAX_FEATURES)
    x_pos = range(t)

    ln1 = ax3.plot(x_pos, faithfulness, color="#d6604d", linewidth=1.8, label="Faithfulness")
    ax3.set_xlabel("Neighbor position")
    ax3.set_ylabel("Faithfulness (shared features)", color="#d6604d")
    ax3.tick_params(axis="y", labelcolor="#d6604d")

    ax3_twin = ax3.twinx()
    ln2 = ax3_twin.plot(x_pos, noisiness, color="#4393c3", linewidth=1.8, label="Noisiness")
    ax3_twin.set_ylabel("Noisiness (non-query features)", color="#4393c3")
    ax3_twin.tick_params(axis="y", labelcolor="#4393c3")

    lines = ln1 + ln2
    ax3.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax3.set_title("Faithfulness vs Noisiness")

    title = f"Q: {query_short}" if len(query_short) <= 80 else f"Q: {query_short[:77]}..."
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout()

    path = config.PLOTS_DIR / f"{question_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved plot to %s", path)


class SummaryPredictor:
    """Adapter around the v4 summary model so call sites stay simple.

    Loads the encoder + head from a single checkpoint and exposes a
    .predict(entropy_map) -> (T, 2) numpy array, where channel 0 is per-neighbor
    noise (#-1 / 64) and channel 1 is per-neighbor faithfulness (#+1 / 10),
    both already in [0, 1].
    """

    def __init__(self, encoder: EntropyEncoder, head: SummaryHead, device):
        self.encoder = encoder
        self.head = head
        self.device = device

    @torch.no_grad()
    def predict(self, entropy_map: np.ndarray) -> np.ndarray:
        """entropy_map: (T, n_h0_bins) numpy. Returns (T, 2) in [0, 1]."""
        x = torch.from_numpy(entropy_map[np.newaxis]).to(self.device)
        t = x.shape[1]
        mask = torch.ones(1, t, dtype=torch.bool, device=self.device)
        z = self.encoder(x, mask)
        out = self.head(z)[0]                  # (T_full=50, 2); summary head is fixed-length
        return out[:t].cpu().numpy()


def _load_feature_predictor(model_path, device) -> SummaryPredictor:
    """Load the v4 summary checkpoint (encoder + head)."""
    ckpt = torch.load(model_path, weights_only=True, map_location=device)
    geom = ckpt["geom"]
    encoder = EntropyEncoder(**geom).to(device).eval()
    head = SummaryHead(**geom).to(device).eval()
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    return SummaryPredictor(encoder=encoder, head=head, device=device)


def _filter_positions(
    shared_counts: np.ndarray,
    extra_counts: np.ndarray,
    shared_tolerance: float,
    noisiness_tolerance: float,
) -> np.ndarray:
    """Filter candidate positions by shared feature count and noisiness.

    Both tolerances are fractions in [0, 1]:
      - 0.0 = strictest (only best candidates pass)
      - 1.0 = most permissive (all candidates pass)

    shared_tolerance: fraction of the shared-count range below the max to accept.
    noisiness_tolerance: fraction of the extra-count range above the min to accept.
    """
    all_positions = np.arange(len(shared_counts))

    max_shared = shared_counts.max()
    if max_shared == 0:
        return np.array([], dtype=int)

    if shared_tolerance >= 1.0:
        selected_positions = all_positions
    else:
        min_shared = shared_counts.min()
        shared_threshold = max_shared - shared_tolerance * (max_shared - min_shared)
        selected_positions = all_positions[shared_counts >= shared_threshold]

    if len(selected_positions) == 0:
        return np.array([], dtype=int)

    if noisiness_tolerance >= 1.0:
        return selected_positions

    selected_extra = extra_counts[selected_positions]
    min_extra = selected_extra.min()
    max_extra = selected_extra.max()
    tolerance_upper = min_extra + noisiness_tolerance * (max_extra - min_extra)
    return selected_positions[selected_extra <= tolerance_upper]


class HomologyRetriever(Retriever):

    def __init__(
        self,
        base_retriever: HybridRetriever,
        chunk_store: ChunkStore,
        n_candidates: int = 50,
        homology_weight: float = 0.5,
    ):
        self.base_retriever = base_retriever
        self.chunk_store = chunk_store
        self.n_candidates = n_candidates
        self.homology_weight = homology_weight
        self._device = torch.device("cpu")
        self._predictor = _load_feature_predictor(
            config.FEATURE_PREDICTOR_MODEL, self._device
        )
        self._plot_idx = 0

    _MAX_RESULTS = 20

    def retrieve(
        self,
        query: str,
        k: int,
        shared_tolerance: float = 0.0,
        noisiness_tolerance: float = 0.2,
    ) -> list[RetrievedChunk]:
        ranked_scores, ranked_chunks, _ = self.base_retriever._get_and_rerank(
            query, self.n_candidates
        )
        if not ranked_chunks:
            return []

        chunk_ids = [c.chunk_id for c in ranked_chunks]
        chunk_embeds = self.chunk_store.get_embeddings(chunk_ids)
        q_embed = self.base_retriever._dense._encode(query).squeeze(0)
        homology_map, sorting_indices = calculate_homology0_map(
            query_embed=q_embed, corpus_embeds=chunk_embeds
        )
        entropy_map = _homology_map_to_entropy_map(homology_map)

        if entropy_map is None:
            logger.info("Fallback: entropy_map is None, returning top %d", self._MAX_RESULTS)
            return self._fallback(ranked_scores, ranked_chunks, self._MAX_RESULTS)

        summary = self._predictor.predict(entropy_map)            # (T, 2): [noise, faith]
        shared_counts = summary[:, 1] * float(NUM_POS_FEATURES)
        extra_counts = summary[:, 0] * float(MAX_FEATURES)

        qid = f"{self._plot_idx:04d}"
        self._plot_idx += 1
        _plot_entropy_and_summary(homology_map, entropy_map, summary, qid, query)

        n_cand = len(ranked_chunks)
        t = min(summary.shape[0], n_cand)
        shared_counts = shared_counts[:t]
        extra_counts = extra_counts[:t]

        forced_positions = np.arange(min(2, t), dtype=int)

        if t > 2:
            extra_positions = _filter_positions(
                shared_counts[2:], extra_counts[2:], shared_tolerance, noisiness_tolerance,
            ) + 2
            extra_positions = extra_positions[np.argsort(extra_counts[extra_positions])]
        else:
            extra_positions = np.array([], dtype=int)

        final_positions = np.concatenate([forced_positions, extra_positions])[: self._MAX_RESULTS]
        logger.info(
            "Retrieved %d chunks (forced %d + %d extras passed filter / %d candidates, "
            "shared=[%.2f,%.2f], extra=[%.2f,%.2f])",
            len(final_positions), len(forced_positions), len(extra_positions), t,
            shared_counts.min(), shared_counts.max(), extra_counts.min(), extra_counts.max(),
        )

        out_chunks = []
        for rank, pos in enumerate(final_positions):
            chunk_idx = int(sorting_indices[pos])
            base = ranked_chunks[chunk_idx]
            out_chunks.append(RetrievedChunk(
                chunk_id=base.chunk_id,
                opinion_id=base.opinion_id,
                normalized_citation=base.normalized_citation,
                text=base.text,
                score=float(shared_counts[pos]),
                source="homology",
                metadata={
                    **base.metadata,
                    "shared_features": float(shared_counts[pos]),
                    "extra_features": float(extra_counts[pos]),
                    "homology_position": int(pos),
                    "rank": rank,
                },
            ))
        return out_chunks

    def _fallback(self, ranked_scores, ranked_chunks, k):
        out = []
        for (_, s), chunk in zip(ranked_scores[:k], ranked_chunks[:k]):
            out.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                opinion_id=chunk.opinion_id,
                normalized_citation=chunk.normalized_citation,
                text=chunk.text,
                score=float(s),
                source="homology_fallback",
                metadata={**chunk.metadata, "fused_score": float(s)},
            ))
        return out


class RandomHomologyRetriever(HomologyRetriever):
    """Baseline that uses the real homology pipeline to determine N (how many
    chunks to retrieve) but then randomly samples N chunks from the candidate
    pool. noise_std adds Gaussian noise to shared/noisiness counts before
    thresholding, perturbing N itself.
    """

    def __init__(
        self,
        base_retriever: HybridRetriever,
        chunk_store: ChunkStore,
        n_candidates: int = 50,
        homology_weight: float = 0.5,
        noise_std: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(base_retriever, chunk_store, n_candidates, homology_weight)
        self._noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def retrieve(
        self,
        query: str,
        k: int,
        shared_tolerance: float = 0.0,
        noisiness_tolerance: float = 0.2,
    ) -> list[RetrievedChunk]:
        ranked_scores, ranked_chunks, _ = self.base_retriever._get_and_rerank(
            query, self.n_candidates
        )
        if not ranked_chunks:
            return []

        chunk_ids = [c.chunk_id for c in ranked_chunks]
        chunk_embeds = self.chunk_store.get_embeddings(chunk_ids)
        q_embed = self.base_retriever._dense._encode(query).squeeze(0)

        homology_map, _ = calculate_homology0_map(
            query_embed=q_embed, corpus_embeds=chunk_embeds
        )
        entropy_map = _homology_map_to_entropy_map(homology_map)

        if entropy_map is None:
            logger.info("RandomHomology fallback: entropy_map is None")
            return self._fallback(ranked_scores, ranked_chunks, self._MAX_RESULTS)

        summary = self._predictor.predict(entropy_map)
        n_cand = len(ranked_chunks)
        t = min(summary.shape[0], n_cand)

        shared_counts = (summary[:t, 1] * float(NUM_POS_FEATURES)).astype(np.float64)
        extra_counts = (summary[:t, 0] * float(MAX_FEATURES)).astype(np.float64)

        if self._noise_std > 0:
            shared_counts += self._rng.normal(0, self._noise_std, size=shared_counts.shape)
            extra_counts += self._rng.normal(0, self._noise_std, size=extra_counts.shape)
            shared_counts = np.clip(shared_counts, 0, None)
            extra_counts = np.clip(extra_counts, 0, None)

        final_positions = _filter_positions(
            shared_counts, extra_counts, shared_tolerance, noisiness_tolerance
        )
        n_to_pick = min(len(final_positions), self._MAX_RESULTS)
        if n_to_pick == 0:
            logger.info("RandomHomology fallback: 0/%d passed filter", t)
            return self._fallback(ranked_scores, ranked_chunks, self._MAX_RESULTS)

        indices = self._rng.choice(n_cand, size=min(n_to_pick, n_cand), replace=False)
        logger.info("RandomHomology: picking %d random chunks (filter passed %d/%d)", n_to_pick, len(final_positions), t)

        out_chunks = []
        for rank, idx in enumerate(indices):
            base = ranked_chunks[idx]
            out_chunks.append(RetrievedChunk(
                chunk_id=base.chunk_id,
                opinion_id=base.opinion_id,
                normalized_citation=base.normalized_citation,
                text=base.text,
                score=0.0,
                source="random_homology",
                metadata={**base.metadata, "rank": rank},
            ))
        return out_chunks


class TailHomologyRetriever(HomologyRetriever):
    """Baseline that picks the WORST N chunks according to homology: lowest
    shared features and highest noisiness. If homology ranking is meaningful,
    this should perform significantly worse than the head retriever.
    """

    def retrieve(
        self,
        query: str,
        k: int,
        shared_tolerance: float = 0.0,
        noisiness_tolerance: float = 0.2,
    ) -> list[RetrievedChunk]:
        ranked_scores, ranked_chunks, _ = self.base_retriever._get_and_rerank(
            query, self.n_candidates
        )
        if not ranked_chunks:
            return []

        chunk_ids = [c.chunk_id for c in ranked_chunks]
        chunk_embeds = self.chunk_store.get_embeddings(chunk_ids)
        q_embed = self.base_retriever._dense._encode(query).squeeze(0)

        homology_map, sorting_indices = calculate_homology0_map(
            query_embed=q_embed, corpus_embeds=chunk_embeds
        )
        entropy_map = _homology_map_to_entropy_map(homology_map)

        if entropy_map is None:
            logger.info("TailHomology fallback: entropy_map is None")
            return self._fallback(ranked_scores, ranked_chunks, self._MAX_RESULTS)

        summary = self._predictor.predict(entropy_map)
        n_cand = len(ranked_chunks)
        t = min(summary.shape[0], n_cand)

        shared_counts = summary[:t, 1] * float(NUM_POS_FEATURES)
        extra_counts = summary[:t, 0] * float(MAX_FEATURES)

        head_positions = _filter_positions(
            shared_counts, extra_counts, shared_tolerance, noisiness_tolerance
        )
        n_to_pick = min(len(head_positions), self._MAX_RESULTS)
        if n_to_pick == 0:
            logger.info("TailHomology fallback: 0/%d passed filter", t)
            return self._fallback(ranked_scores, ranked_chunks, self._MAX_RESULTS)

        tail_order = np.lexsort((-extra_counts, shared_counts))
        tail_positions = tail_order[:n_to_pick]
        logger.info(
            "TailHomology: picking %d tail chunks (head filter passed %d/%d)",
            len(tail_positions), len(head_positions), t,
        )

        out_chunks = []
        for rank, pos in enumerate(tail_positions):
            chunk_idx = int(sorting_indices[pos])
            base = ranked_chunks[chunk_idx]
            out_chunks.append(RetrievedChunk(
                chunk_id=base.chunk_id,
                opinion_id=base.opinion_id,
                normalized_citation=base.normalized_citation,
                text=base.text,
                score=float(shared_counts[pos]),
                source="tail_homology",
                metadata={
                    **base.metadata,
                    "shared_features": float(shared_counts[pos]),
                    "extra_features": float(extra_counts[pos]),
                    "homology_position": int(pos),
                    "rank": rank,
                },
            ))
        return out_chunks
