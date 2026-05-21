"""Baseline class-attribute registry, mirror of ``VDBHooks`` pattern."""

from __future__ import annotations

from blowfish.baselines.b0_random import RandomBaseline
from blowfish.baselines.b1_top1 import Top1ScoreBaseline
from blowfish.baselines.b2_score_gap import ScoreGapBaseline
from blowfish.baselines.b3_score_entropy import ScoreEntropyBaseline
from blowfish.baselines.b4_knn_distance import MeanKnnDistanceBaseline
from blowfish.baselines.b5_knn_density import KnnDensityBaseline
from blowfish.baselines.b6_mahalanobis import MahalanobisCentroidBaseline
from blowfish.baselines.b7_calibrated_logistic import CalibratedLogisticBaseline
from blowfish.baselines.b8_gbm import GBMBaseline
from blowfish.baselines.b9_oracle import OracleBaseline


class BaselineHooks:
    B0 = RandomBaseline
    B1 = Top1ScoreBaseline
    B2 = ScoreGapBaseline
    B3 = ScoreEntropyBaseline
    B4 = MeanKnnDistanceBaseline
    B5 = KnnDensityBaseline
    B6 = MahalanobisCentroidBaseline
    B7 = CalibratedLogisticBaseline
    B8 = GBMBaseline
    B9 = OracleBaseline


ALL_BASELINE_IDS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")
