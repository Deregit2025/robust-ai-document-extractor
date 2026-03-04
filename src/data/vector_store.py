"""
LocalVectorStore (FAISS Edition)

A professional-grade vector store using FAISS for high-performance 
semantic search and JSONL for metadata persistence.
"""

import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

class LocalVectorStore:
    def __init__(self, 
                 storage_dir: str = ".refinery", 
                 index_file: str = "vector_store.faiss",
                 metadata_file: str = "vector_store_metadata.jsonl",
                 dim: int = 1536):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, index_file)
        self.metadata_path = os.path.join(storage_dir, metadata_file)
        self.dim = dim
        self.metadata = []
        
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize FAISS index
        # IndexFlatIP + normalization = Cosine Similarity
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self._load_metadata()
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        self.metadata.append(json.loads(line))
                    except Exception:
                        continue

    def add(self, vector: List[float], metadata: Dict[str, Any]):
        """Adds a vector and its metadata to the FAISS store."""
        vec_np = np.array([vector]).astype("float32")
        
        # Normalize for cosine similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(vec_np)
        
        self.index.add(vec_np)
        self.metadata.append(metadata)
        
        # Persist
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Performs semantic search using FAISS."""
        if self.index.ntotal == 0:
            return []
        
        q_vec = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(q_vec)
        
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue # FAISS returns -1 if not enough results
            results.append({
                "metadata": self.metadata[idx],
                "score": float(score)
            })
        return results

    def clear(self):
        """Clears the FAISS index and metadata."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
