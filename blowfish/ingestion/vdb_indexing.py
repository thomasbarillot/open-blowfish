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

import numpy as np
import pandas as pd
import os
import json
from faiss import write_index, read_index, IndexFlatL2, IndexFlatIP
from typing import ClassVar, Tuple, Union
from pydantic import BaseModel, Field

_INDEX = Union[IndexFlatL2, IndexFlatIP]

class FaissVDBIndexing(BaseModel):
    
    module_name: ClassVar[str] = "FaissIndexer"
    
    vdb_path: str = Field(default="./faiss.index")
    json_index_path: str = Field(default="./index.json")
    vdb_reset_faiss_index: bool = Field(default=False)
    vdb_vector_size: int = Field()
    #: ``"l2"`` (default, backward compatible) or ``"cosine"`` (unit-normalized + inner product).
    vdb_metric: str = Field(default="l2")
    
    class Config:
        extra = "ignore"
    
    def save_index(self,
                   vector_index: _INDEX,
                   index_mapping: list):
        write_index(vector_index, self.vdb_path)
        with open(self.json_index_path, "w") as f:
            json.dump({"map": index_mapping, "metric": self.vdb_metric}, f)
            
    def load_index(self):
        vector_index = read_index(self.vdb_path)
        with open(self.json_index_path, "r") as f:
            payload = json.load(f)
            index_mapping = payload["map"]
        stored_metric = payload.get("metric")
        if stored_metric is not None and stored_metric != self.vdb_metric:
            raise ValueError(
                f"FAISS index at {self.vdb_path!r} was built with vdb_metric="
                f"{stored_metric!r} but is being loaded with vdb_metric="
                f"{self.vdb_metric!r}. Rebuild the index or pass the matching metric."
            )
        return vector_index, index_mapping
                
    def reset_index(self):
        if self.vdb_metric == "cosine":
            vector_index = IndexFlatIP(self.vdb_vector_size)
        elif self.vdb_metric == "l2":
            vector_index = IndexFlatL2(self.vdb_vector_size)
        else:
            raise ValueError('vdb_metric must be "l2" or "cosine"')
        index_mapping = []
        self.save_index(vector_index,index_mapping)
        return vector_index, index_mapping
    

    def _prepare_vectors(self, input: pd.DataFrame) -> np.ndarray:
        vectors = np.asarray(input["chunk_embedding"].to_list(), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.vdb_vector_size:
            raise ValueError(
                f"chunk_embedding must have shape (n, {self.vdb_vector_size}), got {vectors.shape}"
            )
        if self.vdb_metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms
        return vectors
    
    def __call__(self, 
                 input: pd.DataFrame) -> Tuple:
        
        if not os.path.isfile(self.vdb_path):
            vector_index, index_mapping = self.reset_index()
        elif self.vdb_reset_faiss_index:
            vector_index, index_mapping = self.reset_index()
        else: 
            vector_index, index_mapping = self.load_index()
            
        vectors = self._prepare_vectors(input)
        hash_keys = input["hash_key"].to_list()
        duplicated_input_keys = pd.Series(hash_keys).duplicated(keep=False)
        if duplicated_input_keys.any():
            examples = pd.Series(hash_keys)[duplicated_input_keys].unique()[:5].tolist()
            raise ValueError(f"input.hash_key values must be unique. Example duplicates: {examples}")
        existing_keys = set(index_mapping)
        duplicate_existing = [key for key in hash_keys if key in existing_keys]
        if duplicate_existing:
            raise ValueError(
                "FAISS index mapping already contains hash_key values from this input. "
                f"Example duplicates: {duplicate_existing[:5]}"
            )
        vector_index.add(vectors)
        index_mapping += hash_keys
        self.save_index(vector_index, index_mapping)
        
        return vector_index, index_mapping
