from blowfish.inference.scorer import AmbiguityScorer
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

topic_features = {
    "docname": ["doc1", "doc2"],
    "chunk_embedding": [1.512, 2.123],
    "label": [1, 3],
    "silhouette_score": [0.719, 0.890],
    "hash_key": ["hash_key_1", "hash_key_2"]
}
topics_df = pd.DataFrame(topic_features)
scorer = AmbiguityScorer(kde=None, topics_df=topics_df)

def test_join_topics_to_query_and_chunks_topics():
    features = {
        "topn_docname": ["doc1", "doc2"],
        "topn_scores": [0.33, 0.21],
        "topn_rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123],
        "hash_key": ["hash_key_1", "hash_key_2"]
    }

    expected_values = {
        "docname": ["doc1", "doc2"],
        "score": [0.33, 0.21],
        "rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123],
        "hash_key": ["hash_key_1", "hash_key_2"],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "topic_label": ["doc1_1", "doc2_3"]
    }
    
    expected = pd.DataFrame(expected_values)

    test_df = pd.DataFrame(features)
    actual = scorer.join_topics_to_query_and_chunks(test_df).drop('index', axis=1)

    assert_frame_equal(actual, expected)


def test_join_topics_to_query_and_chunks_missing_columns():
    bad_features = {
        "topn_docname": ["doc1", "doc2"],
        "topn_scores": [0.33, 0.21],
        "topn_rank": [0, 1],
        "query_embedding": [1.321, 2.221],
        "chunk_embeddings": [1.512, 2.123]
    }
    fail_df = pd.DataFrame(bad_features)
    res = scorer.join_topics_to_query_and_chunks(fail_df)

    assert res == None


def test_format_topics():
    topic_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "hash_key": ["hash_key_1", "hash_key_2"]
    }

    expected_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890],
        "hash_key": ["hash_key_1", "hash_key_2"],
        "cluster_id": ["doc1_1", "doc2_3"]
    }

    topics_df = pd.DataFrame(topic_features)
    actual = scorer.format_topics(topics_df)
    expected = pd.DataFrame(expected_features).set_index(["docname"])
    
    assert_frame_equal(actual, expected)


def test_format_topics_duplicate_hash_key_raises():
    bad = pd.DataFrame({
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.0, 2.0],
        "label": [1, 3],
        "silhouette_score": [0.7, 0.8],
        "hash_key": ["same", "same"],
    })
    with pytest.raises(ValueError, match="unique hash_key"):
        AmbiguityScorer(kde=None, topics_df=bad)


def test_format_topics_missing_columns():
    bad_features = {
        "docname": ["doc1", "doc2"],
        "chunk_embedding": [1.512, 2.123],
        "label": [1, 3],
        "silhouette_score": [0.719, 0.890]
    }
    fail_df = pd.DataFrame(bad_features)
    res = scorer.format_topics(fail_df)

    assert res == None


def test_calculate_query_correctness_probability_returns_one_row(monkeypatch):
    query_df = pd.DataFrame({
        "score": [0.0, 0.2, 0.4],
        "chunk_embeddings": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0, 0.0],
        ],
        "query_embedding": [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
        "docname": ["doc1", "doc1", "doc2"],
        "topic_label": ["doc1_1", "doc1_1", "doc2_3"],
        "silhouette_score": [0.7, 0.8, 0.9],
    })
    monkeypatch.setattr(scorer, "get_correctness_probability", lambda sample: 0.42)

    out = scorer.calculate_query_correctness_probability(query_df)

    assert len(out) == 1
    assert out.p_correct.iloc[0] == 0.42
