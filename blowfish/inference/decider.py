"""
Copyright 2024 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from blowfish.utils.constants import DEFAULT_KDE_FEATURES


def _import_shap():
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "FeedbackDecider requires the optional 'shap' dependency. "
            "Install with: pip install 'blowfish[explain]' or pip install shap"
        ) from e
    return shap


def _kde_binary_log_proba_col0_row(kde: KernelDensity, row: np.ndarray) -> float:
    """Log P(y=1 | x) from joint KDE over [y, x]; row shape (1, n_features) with col0 = y."""
    y_one = np.concatenate([np.ones((row.shape[0], 1), dtype=row.dtype), row[:, 1:]], axis=1)
    y_zero = np.concatenate([np.zeros((row.shape[0], 1), dtype=row.dtype), row[:, 1:]], axis=1)
    log_p1 = kde.score_samples(y_one)
    log_p0 = kde.score_samples(y_zero)
    return log_p1 - np.logaddexp(log_p1, log_p0)


class FeedbackDecider():
    
    def __init__(self, kde: KernelDensity, kde_features_order: List[str] = DEFAULT_KDE_FEATURES):
        shap = _import_shap()
        self.KDE = kde
        self.high_accuracy_samples = self.get_high_accuracy_samples()
        self.explainer = shap.KernelExplainer(self.KDE_prediction, self.high_accuracy_samples)
        self.KDE_features_order = kde_features_order
        
    def get_high_accuracy_samples(self, num_samples: int = 50) -> np.array:
        """
            Obtain [num_samples] samples from KDE
        """
        feat_samples = []

        while len(feat_samples) < num_samples:
            sample = self.KDE.sample(n_samples=1)
            if np.round(sample[0][0]) == 1.0:
                logp = _kde_binary_log_proba_col0_row(self.KDE, sample)
                proba = np.exp(logp)[0]
                if proba > 0.8:
                    feat_samples.append(sample[0][1:])

        return np.array(feat_samples)
    
    def KDE_prediction(self, sample):
        """
            Get the probability score for the sample from a given KDE
        """
        X_one = np.concatenate([np.ones((sample.shape[0], 1), dtype=sample.dtype), sample], axis=1)
        X_zero = np.concatenate([np.zeros((sample.shape[0], 1), dtype=sample.dtype), sample], axis=1)
        log_p1 = self.KDE.score_samples(X_one)
        log_p0 = self.KDE.score_samples(X_zero)
        return np.exp(log_p1 - np.logaddexp(log_p1, log_p0))
    
    def explain_query(self, 
                      query_features: pd.DataFrame, 
                      relevance_threshold: float = 0.1) -> str:
        """ This function calculates SHAP values for query and generate feedback prompt that depends on feature importance

        :param query_features: (pd.Series) query features engineered from its localization in embedding space with respect to k-nearest neighbours 
        :param relevance_threshold: (float) feature relevance threshold. Below, not considered for prompt generation
        :return: (str) new prompt generator type
        """
        
        query_features_array = query_features[self.KDE_features_order].to_numpy()
        query_shap_values = self.explainer(query_features_array)
        query_shap_values.feature_names = self.KDE_features_order
        
        query_shap_df = pd.DataFrame(dict(**{k:v for k,v in query_shap_values[0,:].__dict__["_s"].__dict__.items() if k in ["values","data"]},
                          **{"features":self.KDE_features_order})).sort_values(by="values")
        query_shap_df = query_shap_df.sort_values(by="values")
        query_shap_df = query_shap_df[query_shap_df["values"] < 0.0]
        query_shap_df["fraction"] = query_shap_df["values"]/query_shap_df["values"].sum()
        
        relevant_features = query_shap_df[query_shap_df["fraction"] > relevance_threshold]["features"].tolist()
        
        if "top_k_topic_spread" in relevant_features:
            return "topicspread"
        elif "top_k_doc_spread" in relevant_features:
            return "docspread"
        else:
            return "dataspread"
            