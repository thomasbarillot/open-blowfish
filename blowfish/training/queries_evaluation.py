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

from typing import List, ClassVar, Any, Tuple
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from collections import defaultdict
from tqdm import tqdm
import Levenshtein
import traceback
from blowfish.utils.vdb_factory import VDBHooks
from blowfish.utils.constants import MIN_TOP_K_FOR_TOPOLOGY_WARNING, PAPER_TOP_K_RESULTS

logger = logging.getLogger(__name__)

class BulkQueriesEvaluator(BaseModel):
    
    module_name: ClassVar[str] = "BulkQueriesEvaluator"

    top_k_results: int = Field()
    vdb_path: str = Field(default="./faiss.index")
    vdb_type: str = Field()
    #: Must match how the FAISS index was built (``"l2"`` or ``"cosine"``).
    vdb_metric: str = Field(default="l2")
    
    vdb_index: Any = None
    vdb_mapping: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.top_k_results < MIN_TOP_K_FOR_TOPOLOGY_WARNING:
            logger.warning(
                "top_k_results=%s is below %s; H1 / persistent homology features may be unstable. "
                "Paper (arXiv:2406.07990) uses k=%s.",
                self.top_k_results,
                MIN_TOP_K_FOR_TOPOLOGY_WARNING,
                PAPER_TOP_K_RESULTS,
            )
        index, idx_mapping = getattr(VDBHooks, self.vdb_type)(**kwargs).load_index()
        self.vdb_index = index
        self.vdb_mapping = idx_mapping
        
    
    def evaluate_isin(self,
                      query_series: pd.Series, 
                      answer: pd.Series) -> Tuple[bool]:
        """
            Evaluates if the retrieved document matches the answer. This is performed
            using levenshtein distance.
        """
        
        docname, content = answer[["docname", "Text"]]
        
        isindocument = query_series.docname == docname
        isinchunk = Levenshtein.distance(content,query_series.answer,weights=(0,1,0)) == 0
        
        return isindocument, isindocument*isinchunk
    

    def __call__(self, queries_df: pd.DataFrame, topics_df: pd.DataFrame) -> pd.DataFrame:
        """ Performs a vector db retrieval and compares the retrieved chunks with the original chunk """
        
        try:  
            assert isinstance(queries_df, pd.DataFrame)
            assert isinstance(topics_df, pd.DataFrame)
            
            input_columns = set(queries_df)
            expected_cols = {"query", "docname", "query_embedding", "answer"}
            assert input_columns.issuperset(expected_cols)
            
            input_columns = set(topics_df)
            expected_cols = {'docname', 'Text', 'chunk_embedding', 'hash_key','X', 'Y', 'label', 'silhouette_score'}
            assert input_columns.issuperset(expected_cols)
            
            if topics_df["hash_key"].duplicated().any():
                dup_examples = topics_df.loc[topics_df["hash_key"].duplicated(keep=False), "hash_key"].unique()[:10]
                raise ValueError(
                    "topics_df.hash_key must be unique for retrieval joins. "
                    f"Example duplicate keys: {dup_examples.tolist()}"
                )
            topics_df = topics_df.set_index("hash_key")
            
            QA_output = []
            
            for _, row in tqdm(queries_df.iterrows()):
                query_results = defaultdict(list)

                Qembed = np.asarray(row.query_embedding, dtype=np.float32)[np.newaxis, :]
                if self.vdb_metric == "cosine":
                    nrm = np.linalg.norm(Qembed, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-12)
                    Qembed = Qembed / nrm
                D, I = self.vdb_index.search(Qembed, self.top_k_results)
                raw = np.asarray(D[0], dtype=np.float64)
                if self.vdb_metric == "cosine":
                    # FAISS IndexFlatIP returns inner products in descending order
                    # (largest = most similar). Convert to cosine *distance* so the
                    # downstream `scale_*` features (which assume small = closer)
                    # behave consistently with the L2 path.
                    dist_or_ip = 1.0 - raw
                else:
                    dist_or_ip = raw
                for score, fidx in zip(dist_or_ip, I[0]):
                    hash_key = self.vdb_mapping[fidx]
                    retrieved_chunk = topics_df.loc[hash_key]
                    query_results["chunk_topn_answer"].append(retrieved_chunk["Text"])
                    query_results["chunk_topn_scores"].append(score)
                    query_results["chunk_topn_docname"].append(retrieved_chunk["docname"])
                    query_results["topic_labels"].append(retrieved_chunk['label'])
                    query_results["chunk_embeddings"].append(retrieved_chunk["chunk_embedding"])
                    dmatch, cmatch = self.evaluate_isin(row, retrieved_chunk)
                    query_results["doc_match"].append(dmatch)
                    query_results["chunk_match"].append(cmatch)
                    query_results["silhouette_score"].append(retrieved_chunk["silhouette_score"])
                    
                QA_output.append(pd.concat([row, pd.Series(query_results)]))
            return pd.DataFrame(QA_output)
        
        except AssertionError:
            print(f"Error parsing query.\n Missing columns: {expected_cols.difference(input_columns)}")
            print(f"Error position: {traceback.format_exc()}")
            return None