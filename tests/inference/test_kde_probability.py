import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from blowfish.inference.scorer import AmbiguityScorer


def test_correctness_probability_log_domain_matches_naive():
    np.random.seed(42)
    n, d = 40, 2
    Xf = np.random.randn(n, d).astype(np.float64)
    y = (Xf[:, 0] > 0).astype(np.float64)
    train = np.column_stack([y, Xf])
    kde = KernelDensity(bandwidth=0.4, kernel="gaussian").fit(train)

    topics = pd.DataFrame({
        "docname": ["a"],
        "chunk_embedding": [[0.0, 0.0]],
        "label": [0],
        "silhouette_score": [0.0],
        "hash_key": ["h0"],
    })
    scorer = AmbiguityScorer(kde=kde, topics_df=topics)

    x0, x1 = 0.3, -0.2
    got = scorer.get_correctness_probability([x0, x1])

    sample_correct = [1.0, x0, x1]
    sample_incorrect = [0.0, x0, x1]
    log_p1 = kde.score_samples([sample_correct])[0]
    log_p0 = kde.score_samples([sample_incorrect])[0]
    naive = float(np.exp(log_p1) / (np.exp(log_p1) + np.exp(log_p0)))
    assert abs(got - naive) < 1e-12
