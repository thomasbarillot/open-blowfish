from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from . import config
from .corpus.chunking import chunk_text
from .retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_embeddings(texts, model, tokenizer, max_length, device):
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        return_tensors="pt", max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


_EMBED_CHECKPOINT = config.INDEX_DIR / "embeddings_checkpoint.npy"
_EMBED_PROGRESS = config.INDEX_DIR / "embeddings_progress.txt"
_CHECKPOINT_EVERY = 500


def _encode_batched(
    texts: list[str],
    max_length: int,
    batch_size: int,
    model_name: str = config.EMBEDDING_MODEL,
) -> np.ndarray:
    device = _get_device()
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=model_config)
    model.to(device).eval()

    start_batch = 0
    all_embeddings = []

    if _EMBED_CHECKPOINT.exists() and _EMBED_PROGRESS.exists():
        start_batch = int(_EMBED_PROGRESS.read_text().strip())
        cached = np.load(_EMBED_CHECKPOINT)
        all_embeddings = [cached]
        logger.info("Resuming from batch %d (%d vectors cached)", start_batch, cached.shape[0])

    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("Encoding %d texts on %s (batch_size=%d, batches %d-%d)",
                len(texts), device, batch_size, start_batch, total_batches)

    new_embeddings = []

    for batch_idx in tqdm(range(start_batch, total_batches), desc="encoding",
                          initial=start_batch, total=total_batches):
        i = batch_idx * batch_size
        batch = texts[i : i + batch_size]
        embeddings = _get_embeddings(batch, model, tokenizer, max_length, device)
        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
        new_embeddings.append(embeddings)

        if (batch_idx + 1) % _CHECKPOINT_EVERY == 0:
            chunk = np.concatenate(new_embeddings, axis=0)
            new_embeddings = []
            all_embeddings.append(chunk)
            merged = np.concatenate(all_embeddings, axis=0)
            np.save(_EMBED_CHECKPOINT, merged)
            _EMBED_PROGRESS.write_text(str(batch_idx + 1))
            logger.info("Checkpoint at batch %d (%d vectors)", batch_idx + 1, merged.shape[0])

    if new_embeddings:
        all_embeddings.append(np.concatenate(new_embeddings, axis=0))
    result = np.concatenate(all_embeddings, axis=0)

    if _EMBED_CHECKPOINT.exists():
        _EMBED_CHECKPOINT.unlink()
    if _EMBED_PROGRESS.exists():
        _EMBED_PROGRESS.unlink()

    return result


def build_chunks() -> pd.DataFrame:
    opinions = pd.read_parquet(config.CORPUS_DIR / "opinions.parquet")
    all_chunks = []
    for _, row in tqdm(opinions.iterrows(), total=len(opinions), desc="chunking"):
        all_chunks.extend(chunk_text(row["text"], row["opinion_id"], row["normalized_citation"]))
    df = pd.DataFrame(all_chunks)
    df.to_parquet(config.CORPUS_DIR / "chunks.parquet", index=False)
    logger.info("Wrote %d chunks from %d opinions", len(df), len(opinions))

    secondary_by_op: dict[str, set[str]] = {}
    for _, c in df.iterrows():
        cites = c["secondary_citations"]
        if cites is None or len(cites) == 0:
            continue
        secondary_by_op.setdefault(c["opinion_id"], set()).update(cites)
    opinions["secondary_citations"] = opinions["opinion_id"].map(
        lambda op_id: sorted(secondary_by_op.get(op_id, set()))
    )
    opinions.to_parquet(config.CORPUS_DIR / "opinions.parquet", index=False)
    logger.info("Updated opinions.parquet with secondary_citations (%d opinions have secondaries)",
                sum(1 for v in opinions["secondary_citations"] if len(v) > 0))
    return df


def build_bm25(chunks: pd.DataFrame) -> None:
    chunk_dicts = chunks.to_dict("records")
    retriever = BM25Retriever.from_chunks(chunk_dicts)
    retriever.save(config.INDEX_DIR / "bm25")
    logger.info("BM25 index saved to %s", config.INDEX_DIR / "bm25")


def build_dense(chunks: pd.DataFrame, batch_size: int = 64) -> None:
    texts = chunks["text"].tolist()
    chunk_ids = chunks["chunk_id"].tolist()

    embeddings = _encode_batched(
        texts,
        max_length=config.CHUNK_TOKENS,
        batch_size=batch_size,
    )

    np.save(config.INDEX_DIR / "embeddings.npy", embeddings)

    id_map = pd.DataFrame({"chunk_id": chunk_ids, "row": range(len(chunk_ids))})
    id_map.to_parquet(config.INDEX_DIR / "chunk_id_to_row.parquet", index=False)

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(config.INDEX_DIR / "dense.faiss"))
    logger.info("Dense index: %d vectors, dim=%d", index.ntotal, embeddings.shape[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["chunks", "bm25", "dense", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if args.step in ("chunks", "all"):
        chunks = build_chunks()
    else:
        chunks = pd.read_parquet(config.CORPUS_DIR / "chunks.parquet")

    if args.step in ("bm25", "all"):
        build_bm25(chunks)

    if args.step in ("dense", "all"):
        build_dense(chunks, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
