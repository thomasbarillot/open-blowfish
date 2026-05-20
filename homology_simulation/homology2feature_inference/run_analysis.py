#!/usr/bin/env python3
"""
OTTP Topology Analysis

Creates a topology analysis dataset from tagged documents by:
1. Fetching all tags starting with "extr_test." from DynamoDB
2. Sampling tagged documents (up to 5 per tag)
3. Extracting query texts at paragraph and sentence granularity from ExtractionSources
4. Running BM25 precedent search once per sample
5. Parsing retrieved documents at 3 corpus granularities (chunk/paragraph/sentence)
6. Computing embeddings (local HuggingFace) and persistent homology for each combination
7. Assembling results into a single DataFrame
"""

import argparse
import io
import pickle
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from clients.opensearch_client import Query, WorkflowDBClient
from common_tools.logger import EC2Logger
from core.extraction.word_parsing.word_doc import WordDoc
from core.extraction.word_parsing.word_revisions import SequenceSimilarity
from objects.mutable_objects import TagDefinition
from objects.transient_objects import DocumentLevelPrecedentSearchRequest, SearchFilterObject
from scipy.stats import entropy, norm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

from config import WORKFLOWDB_CONTENTS_INDEX, WORKFLOWDB_DOCUMENTS_INDEX
from scripts.homology_simulation.homology_calculation import calculate_homology_for_query_corpus

logger = EC2Logger(log_group_name="Dev2HomologyExpRMResearchMachineLogs",log_stream_name="ottp_topology_analysis-4")

MAX_NEIGHBOURS = 200
TOP_N_CANDIDATES = 200
TOKEN_LIMIT = 1500
S3_BUCKET = "homology-experiment"


class EmbeddingClientWrapper:
    def __init__(self, model_name: str = "joe32140/ModernBERT-base-msmarco"):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local model: {model_name}")
        self.local_model = SentenceTransformer(model_name)
        logger.info("Local model loaded")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.local_model.encode(texts, convert_to_numpy=True)


def fetch_extr_test_tags(db_client: WorkflowDBClient) -> list[TagDefinition]:

    Q = Query()
    Q.must("UserTagSet.ParagraphTags.*", "NDA Extractor", "wildcard")
    res = db_client.query(index=WORKFLOWDB_DOCUMENTS_INDEX, query=Q.q, size=1)
    if res:
        all_tags = [t for t in res[0][0].UserTagSet.ParagraphTags["None"]]
        # all_tags = TagDefinition.get_all_tags()
        # filtered = [t for t in all_tags if t.TagName.startswith("extr_test.")]
        logger.info(f"Found {len(all_tags)} tags from NDA extractor")
        return all_tags
    logger.warning("Found no tags from NDA extractor")
    return []


def sample_tagged_documents(tag_name: str, db_client: WorkflowDBClient, max_samples: int = 5) -> list[tuple]:
    """Find documents with the given tag and return up to max_samples (doc, tag_instance) tuples."""
    q = Query()
    q.must("UserTagSet.ParagraphTags.*.Key", tag_name, "wildcard")
    # q.must("UserTagSet.ParagraphTags.None.Key", "executed version", "match")

    results = db_client.query(
        index=WORKFLOWDB_DOCUMENTS_INDEX,
        query=q.q,
        size=max_samples * 2,
    )

    samples = []
    for doc, _score in results:
        if len(samples) >= max_samples:
            break
        tag_instance = _find_tag_instance(doc, tag_name)
        if tag_instance is None:
            continue
        if not tag_instance.ExtractionSources:
            continue
        samples.append((doc, tag_instance))

    logger.info(f"Tag '{tag_name}': sampled {len(samples)} documents")
    return samples


def _find_tag_instance(doc, tag_name: str):
    """Extract the matching TagInstance from a document's UserTagSet."""
    if not doc.UserTagSet or not doc.UserTagSet.ParagraphTags:
        return None
    for scope_key, tag_list in doc.UserTagSet.ParagraphTags.items():
        for tag in tag_list:
            if tag.Key == tag_name:
                return tag
    return None


def extract_query_texts(doc, tag_instance, word_doc=None) -> dict | None:
    """
    Extract paragraph and sentence query texts from the tag's ExtractionSources.
    Returns dict with 'paragraph', 'sentence', 'word_doc' keys, or None if extraction fails.
    """
    if not tag_instance.ExtractionSources:
        return None

    # Only use the first paragraph in the source
    source = tag_instance.ExtractionSources[0]
    sentence_text = source.Phrase
    if not sentence_text or len(sentence_text.strip()) < 10:
        return None

    if word_doc is None:
        if not doc.ExtractedText:
            return None
        word_doc = WordDoc.from_b64(doc.ExtractedText).parse()

    span = source.Span
    paragraphs = word_doc.doc.p
    pstart = min(span.Pstart, len(paragraphs) - 1)
    pend = min(span.Pend, len(paragraphs) - 1)

    paragraph_texts = []
    for i in range(pstart, pend + 1):
        if i < len(paragraphs) and paragraphs[i].text.strip():
            paragraph_texts.append(paragraphs[i].text)

    paragraph_text = " ".join(paragraph_texts)
    if not paragraph_text or len(paragraph_text.strip()) < 10:
        paragraph_text = sentence_text

    return {
        "paragraph": paragraph_text,
        "sentence": sentence_text,
        "pnum": pstart,
        "word_doc": word_doc,
    }


def retrieve_candidates(
    query_text: str,
    tag_name: str,
    source_doc_id: str,
    db_client: WorkflowDBClient,
    user_email: str | None,
) -> list[tuple]:
    """
    BM25 precedent search, then filter to documents that carry the tag of interest.
    Returns list of (doc_id, document_object) tuples.
    """
    search_request = DocumentLevelPrecedentSearchRequest(
        WorkItemId="", Client="", Filters=SearchFilterObject(Tags=[tag_name, "executed_version"], DocumentType="DOCX")
    )

    candidates = db_client.find_precedent(
        index=WORKFLOWDB_CONTENTS_INDEX,
        query=query_text,
        query_obj=search_request,
        require_tags=True,
        user_email=user_email,
        token_limit=TOKEN_LIMIT,
        size=TOP_N_CANDIDATES,
        fields=["ID"],
    )

    doc_ids_seen = set()
    candidate_doc_ids = []

    for chunk_obj in candidates:
        doc_id = chunk_obj["ID"][0]
        if doc_id not in doc_ids_seen and doc_id != source_doc_id:
            doc_ids_seen.add(doc_id)
            candidate_doc_ids.append(doc_id)
        if len(candidate_doc_ids) >= 20:
            break

    tagged_docs = []
    for doc_id in candidate_doc_ids:
        try:
            doc = db_client.get(index=WORKFLOWDB_DOCUMENTS_INDEX, id_=doc_id)
        except (KeyError, Exception):
            continue
        if doc is None:
            continue
        tag_inst = _find_tag_instance(doc, tag_name)
        if tag_inst is not None and tag_inst.ExtractionSources:
            tagged_docs.append((doc_id, doc))

    logger.info(f"BM25 returned {len(candidate_doc_ids)} unique docs, {len(tagged_docs)} have tag '{tag_name}'")
    return tagged_docs


def _sentence_overlaps_span(p_index: int, s_start: int, s_end: int, span) -> bool:
    """Check if a sentence (by paragraph index and char offsets) overlaps an ExtractionSpan."""
    if p_index < span.Pstart or p_index > span.Pend:
        return False
    if span.Pstart == span.Pend:
        return s_end > span.Start and s_start < span.End
    if p_index == span.Pstart:
        return s_end > span.Start
    if p_index == span.Pend:
        return s_start < span.End
    # Middle paragraph -- fully inside span
    return True


def _chunk_has_tag(chunk, tagged_p_indices: list) -> bool:
    """Check if a content object carries the given tag in its UserTagSet."""
    if not hasattr(chunk, "UserTagSet") or not chunk.UserTagSet:
        return False
    if not chunk.UserTagSet.ParagraphTags:
        return False
    if any(cp in tagged_p_indices for cp in chunk.ParagraphNumbers):
        return True
    return False


def parse_corpus_at_granularities(
    tagged_docs: list[tuple],
    tag_name: str,
    db_client: WorkflowDBClient,
    source_doc_id: str,
) -> dict[str, dict]:
    """
    Parse retrieved documents at chunk, paragraph, and sentence granularity.
    Returns dict mapping granularity name to {"texts": list[str], "n_tagged": int}.
    n_tagged counts items that specifically match the tag:
      - chunk: content objects carrying the tag in UserTagSet
      - paragraph: paragraphs within the tag's ExtractionSpan range
      - sentence: sentences overlapping the tag's ExtractionSpan character range
    """
    chunk_texts = []
    paragraph_texts = []
    sentence_texts = []
    n_tagged_chunks = 0
    n_tagged_paragraphs = 0
    n_tagged_sentences = 0

    tagged_doc_ids = set()
    for doc_id, doc in tagged_docs:
        if doc_id == source_doc_id:
            continue

        # Paragraph and sentence level: parse WordDoc
        if not doc.ExtractedText:
            continue

        tagged_doc_ids.update([doc_id])
        tag_inst = _find_tag_instance(doc, tag_name)

        # Collect ExtractionSpan objects for this tag
        extraction_spans = []
        if tag_inst and tag_inst.ExtractionSources:
            for src in tag_inst.ExtractionSources:
                extraction_spans.append(src.Span)
        tagged_p_indices = set()
        for span in extraction_spans:
            for pi in range(span.Pstart, span.Pend + 1):
                tagged_p_indices.add(pi)

        # Chunk level: query contents index
        doc_query = Query()
        doc_query.must("ID", doc_id, "match")
        content_results = db_client.field_query(
            index=WORKFLOWDB_CONTENTS_INDEX, query=doc_query.q, size=1000, fields=["ContentID"]
        )
        for chunk_id in content_results:
            chunk = db_client.get(index=WORKFLOWDB_CONTENTS_INDEX, id_=chunk_id["ContentID"][0])
            if chunk.Description and chunk.Description.strip():
                chunk_texts.append(chunk.Description)
                if _chunk_has_tag(chunk, tagged_p_indices):
                    n_tagged_chunks += 1

        word_doc = WordDoc.from_b64(doc.ExtractedText).parse()

        for p in word_doc.doc.p:
            if p.word_count > 0 and p.text.strip():
                paragraph_texts.append(p.text)
                if p.i in tagged_p_indices:
                    n_tagged_paragraphs += 1
                for s in p.sentences:
                    if s.text.strip() and s.word_count > 1:
                        sentence_texts.append(s.text)
                        if any(_sentence_overlaps_span(p.i, s.start, s.end, span) for span in extraction_spans):
                            n_tagged_sentences += 1

    n_documents = len(tagged_doc_ids)
    return {
        "chunk": {"texts": chunk_texts, "n_tagged": n_tagged_chunks, "n_documents": n_documents},
        "paragraph": {"texts": paragraph_texts, "n_tagged": n_tagged_paragraphs, "n_documents": n_documents},
        "sentence": {"texts": sentence_texts, "n_tagged": n_tagged_sentences, "n_documents": n_documents},
    }


def compute_cross_entropy_and_kl_div(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-entropy and KL divergence against a normal reference distribution."""
    if len(df) == 0:
        df["cross_entropy"] = []
        df["kl_divergence"] = []
        return df

    valid_rows = df.dropna(subset=["h0_dist"])
    if len(valid_rows) == 0:
        df["cross_entropy"] = np.nan
        df["kl_divergence"] = np.nan
        return df

    h0_array_ref = np.array(df.iloc[df.nneighbours.idxmax()]["h0_dist"])
    hist_ref, bin_edges = np.histogram(h0_array_ref, bins=np.arange(0, 1.2, 0.05), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_val = bin_centers[np.argmax(hist_ref)]
    std_val = h0_array_ref.std() if h0_array_ref.std() > 0 else 0.1

    cross_entropy_vals = []
    kl_divergence_vals = []

    for _idx, row in df.iterrows():
        h0_dist = row["h0_dist"]
        if isinstance(h0_dist, (list, np.ndarray)) and len(h0_dist) > 1:
            h0_array = np.array(h0_dist)
            hist, bin_edges = np.histogram(h0_array, bins=np.arange(0, 1.2, 0.05), density=True)
            hist = hist + 1e-10
            hist = hist / hist.sum()

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            normal_dist = norm.pdf(bin_centers, loc=mean_val, scale=std_val)
            normal_dist = normal_dist + 1e-10
            normal_dist = normal_dist / normal_dist.sum()

            ce = -np.sum(hist * np.log(normal_dist + 1e-10))
            cross_entropy_vals.append(ce)

            kl = np.sum(normal_dist * np.log((normal_dist + 1e-10) / (hist + 1e-10)))
            kl_divergence_vals.append(kl)
        else:
            cross_entropy_vals.append(np.nan)
            kl_divergence_vals.append(np.nan)

    df["cross_entropy"] = cross_entropy_vals
    df["kl_divergence"] = kl_divergence_vals
    return df


def compute_homology_pipeline(
    source_text: str,
    corpus_texts: list[str],
    embedding_client: EmbeddingClientWrapper,
) -> dict | None:
    """
    Compute embeddings and persistent homology for a query against a corpus.
    Returns dict with df_per_nneigh, df_rev_per_nneigh, similarity maps, and metadata.
    """
    if len(corpus_texts) < 3:
        return None

    source_embedding = embedding_client.embed_texts([source_text])
    corpus_embeddings = embedding_client.embed_texts(corpus_texts)

    # Token similarity
    source_tokens = source_text.lower().split()
    similarity_calc = SequenceSimilarity().set(source_tokens)
    token_similarity_map = {}
    for text in corpus_texts:
        token_similarity_map[text] = similarity_calc.per_token_similarity(text.lower().split())

    # Cosine similarity
    cosine_sims = cosine_similarity(source_embedding, corpus_embeddings)[0]
    cosine_similarity_map = {text: cosine_sims[i] for i, text in enumerate(corpus_texts)}

    # Euclidean distances
    all_embeds = np.vstack([source_embedding, corpus_embeddings])
    distances = euclidean_distances(all_embeds)
    unsorted_scores = distances[1:, 0]

    # Filter near-duplicates and cap at MAX_NEIGHBOURS
    non_dup_mask = unsorted_scores > 1e-4
    n_bypassed = int(np.sum(~non_dup_mask))

    filtered_embeddings = corpus_embeddings[non_dup_mask]
    filtered_scores = unsorted_scores[non_dup_mask]
    filtered_texts = [t for t, m in zip(corpus_texts, non_dup_mask) if m]

    sort_idxs = np.argsort(filtered_scores)
    sorted_scores = filtered_scores[sort_idxs]
    sorted_embeddings = filtered_embeddings[sort_idxs]
    sorted_texts = [filtered_texts[i] for i in sort_idxs]

    if len(sorted_scores) > MAX_NEIGHBOURS:
        sorted_scores = sorted_scores[:MAX_NEIGHBOURS]
        sorted_embeddings = sorted_embeddings[:MAX_NEIGHBOURS]
        sorted_texts = sorted_texts[:MAX_NEIGHBOURS]

    if len(sorted_scores) < 3:
        return None

    DEPSILON_RANGE = sorted_scores / sorted_scores[0] - 1

    all_homology_results = []

    for de in DEPSILON_RANGE:
        hr = calculate_homology_for_query_corpus(
            corpus_embeds=sorted_embeddings,
            query_embed=source_embedding[0],
            depsilon=de,
            reverse=False,
        )
        all_homology_results.append(hr | {"n_bypassed_neighbours": n_bypassed, "depsilon": de})

    df = pd.DataFrame(all_homology_results)

    sorted_texts_arr = np.array(sorted_texts)
    df["text"] = df["nneighbours"].apply(lambda x: sorted_texts_arr[max(int(x) - 1, 0)] if not np.isnan(x) else "")

    # Token/cosine similarity for boundary texts
    df["token_similarity"] = df["text"].apply(lambda t: token_similarity_map.get(t, np.nan))
    df["cosine_similarity"] = df["text"].apply(lambda t: cosine_similarity_map.get(t, np.nan))

    # Per-neighbour aggregation
    df_per_nneigh = (
        df.groupby("nneighbours", group_keys=False)
        .apply(lambda x: x.iloc[0], include_groups=False)
        .dropna(subset="h0_dist")
        .reset_index()
    )
    if len(df_per_nneigh) > 0:
        df_per_nneigh["h0_dist"] = df_per_nneigh["h0_dist"].apply(lambda x: list(x))

    df_per_nneigh = compute_cross_entropy_and_kl_div(df_per_nneigh)

    return {
        "df_per_nneigh": df_per_nneigh,
        "n_corpus": len(corpus_texts),
    }


def process_single_sample(
    tag_name: str,
    doc,
    tag_instance,
    db_client: WorkflowDBClient,
    embedding_client: EmbeddingClientWrapper,
    user_email: str | None,
) -> list[pd.DataFrame]:
    """
    Process one sampled document through the full pipeline.
    Returns list of annotated DataFrames (one per query_granularity x corpus_granularity combo).
    """
    query_texts = extract_query_texts(doc, tag_instance)
    if query_texts is None:
        logger.warning(f"No query texts extracted for doc {doc.ID}, tag '{tag_name}'")
        return []

    # BM25 once using paragraph text (more informative for retrieval)
    logger.info(f"Retrieving candidates for doc {doc.ID}, tag '{tag_name}'")
    tagged_docs = retrieve_candidates(
        query_text=query_texts["paragraph"],
        tag_name=tag_name,
        source_doc_id=doc.ID,
        db_client=db_client,
        user_email=user_email,
    )

    if not tagged_docs:
        logger.warning(f"No tagged candidates found for doc {doc.ID}, tag '{tag_name}'")
        return []

    logger.info(f"Parsing corpus at granularities for doc {doc.ID} ({len(tagged_docs)} candidate docs)")
    corpus = parse_corpus_at_granularities(tagged_docs, tag_name, db_client, doc.ID)

    result_dfs = []

    for query_gran, query_text, pnum in [
        ("paragraph", query_texts["paragraph"], query_texts["pnum"]),
        ("sentence", query_texts["sentence"], query_texts["pnum"]),
    ]:
        for corpus_gran, corpus_data in corpus.items():
            corpus_texts = corpus_data["texts"]
            n_tagged = corpus_data["n_tagged"]
            n_documents = corpus_data["n_documents"]
            if len(corpus_texts) < 3:
                continue

            logger.info(
                f"Computing homology for tag '{tag_name}', query={query_gran}, corpus={corpus_gran}"
                f" ({len(corpus_texts)} texts, {n_tagged} tagged)"
            )
            pipeline_result = compute_homology_pipeline(query_text, corpus_texts, embedding_client)
            if pipeline_result is None:
                logger.warning(
                    f"Homology pipeline returned None for tag '{tag_name}'"
                    f", query={query_gran}, corpus={corpus_gran}"
                )
                continue

            df = pipeline_result["df_per_nneigh"].copy()
            if len(df) == 0:
                continue
            df["tag_name"] = tag_name
            df["doc_name"] = getattr(doc, "Name", doc.ID)
            df["pnum"] = pnum
            df["query_granularity"] = query_gran
            df["corpus_granularity"] = corpus_gran
            df["source_text"] = query_text
            df["n_tagged"] = n_tagged
            df["n_documents"] = n_documents

            result_dfs.append(df)

    return result_dfs


def store_results(args, db_client, results, tag_name):
    try:
        df = pd.concat(results, ignore_index=True)
        key = f"ottp_topology_df_tag_{tag_name}.pkl"

        if args.to_s3:
            logger.info(f"Uploading {len(df)} rows to s3://{S3_BUCKET}/{key}")
            buf = io.BytesIO()
            pickle.dump(df, buf)
            buf.seek(0)
            db_client.s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.read())
            logger.info(f"Uploaded {len(df)} rows to s3://{S3_BUCKET}/{key}")
        else:
            output_path = Path(args.output) if args.output else Path(__file__).parent / "outputs" / key
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(output_path)
            logger.info(f"Saved {len(df)} rows to {output_path}")
    except Exception:
        logger.warning(f"Failed to save results for tag {tag_name}; {len(results)=}")


def main():
    parser = argparse.ArgumentParser(description="OTTP Topology Analysis")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch tags and print counts")
    parser.add_argument("--max-tags", type=int, default=None, help="Limit number of tags to process")
    parser.add_argument("--max-samples", type=int, default=5, help="Max samples per tag")
    parser.add_argument("--to-s3", action="store_true", help="Save results to S3 instead of filesystem")
    parser.add_argument("--user-email", type=str, default=None, help="User email for permission filtering")
    parser.add_argument("--output", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    db_client = WorkflowDBClient()
    embedding_client = EmbeddingClientWrapper()

    tags = fetch_extr_test_tags(db_client)

    if args.max_tags:
        tags = tags[: args.max_tags]
        logger.info(f"Processing {len(tags)} tags (limited by --max-tags)")

    if args.dry_run:
        for tag in tags:
            print(f"  {tag}")
        return

    #all_results = []
    #failed = []

    for tag_def in tqdm(tags, desc="Processing tags"):
        try:
            all_tag_results = []
            samples = sample_tagged_documents(tag_def.Key, db_client, max_samples=args.max_samples)

            for doc, tag_instance in tqdm(samples, desc=f"  {tag_def}", leave=False):
                try:
                    result_dfs = process_single_sample(
                        tag_def.Key, doc, tag_instance, db_client, embedding_client, args.user_email
                    )
                    all_tag_results.extend(result_dfs)
                except Exception:
                    #failed.append({"tag": tag_def.Key, "doc_id": doc.ID, "error": traceback.format_exc()})
                    logger.error(f"Failed processing doc {doc.ID} for tag '{tag_def.Key}': {traceback.format_exc()}")

            store_results(args, db_client, all_tag_results, tag_def.Key)

        except Exception:
            logger.error(f"Failed processing tag '{tag_def.Key}': {traceback.format_exc()}")

    # if failed:
    #     failed_df = pd.DataFrame(failed)
    #     failed_path = Path(__file__).parent / "outputs" / "failed_samples.csv"
    #     failed_path.parent.mkdir(parents=True, exist_ok=True)
    #     failed_df.to_csv(failed_path, index=False)
    #     logger.warning(f"Saved {len(failed)} failures to {failed_path}")


if __name__ == "__main__":
    main()
