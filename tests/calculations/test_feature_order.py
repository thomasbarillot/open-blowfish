import pandas as pd

from blowfish.calculations.calculations import (
    calculate_relevant_features,
    calculate_scaled_distance_distribution,
)


def test_returned_dict_iterates_in_kde_features_order():
    sub = pd.DataFrame({
        "score": [0.1, 0.2, 0.3],
        "chunk_embeddings": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0, 0.0],
        ],
        "query_embedding": [[0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
        "docname": ["a", "a", "b"],
        "topic_label": ["a_1", "a_1", "b_2"],
        "silhouette_score": [0.5, 0.6, 0.7],
    })

    order = [
        "top_k_topic_spread",
        "scale_min",
        "w1_h0",
        "silhouette_score_mean",
        "lt_max_h1",
    ]
    feats = calculate_relevant_features(sub, order)

    assert list(feats.keys()) == order


def test_unknown_feature_raises():
    sub = pd.DataFrame({
        "score": [0.1, 0.2],
        "chunk_embeddings": [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
        "query_embedding": [[0.0, 0.0, 0.0, 0.0, 0.0]] * 2,
        "docname": ["a", "b"],
        "topic_label": ["a_1", "b_2"],
        "silhouette_score": [0.5, 0.6],
    })
    try:
        calculate_relevant_features(sub, ["w1_h0", "not_a_real_feature"])
    except KeyError as e:
        assert "not_a_real_feature" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown feature")


def test_scaled_distance_distribution_handles_exact_match_zero_distance():
    sub = pd.DataFrame({"score": [0.0, 0.2, 0.4]})

    out = calculate_scaled_distance_distribution(sub)

    assert out["scale_min"] == 1.0
    assert out["scale_mean"] == 1.5
    assert out["iq25-75_scale"] == 0.5
