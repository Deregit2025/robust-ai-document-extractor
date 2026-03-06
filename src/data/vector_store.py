"""
LocalVectorStore (FAISS Edition)

Satisfies Feedback #5 requirements:
1. Versioning: Supports versioned snapshots of the index.
2. Deduplication: Avoids re-indexing identical content hashes.
3. Cross-Verification: Ensures PageIndex entries match vector store records.
4. Atomicity: Uses secure file handling for metadata.
"""

import json
import os
import numpy as np
import faiss
from datetime import datetime
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
        self.seen_hashes = set()
        
        os.makedirs(self.storage_dir, exist_ok=True)
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self._load_metadata()
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        self.metadata.append(record)
                        if "content_hash" in record:
                            self.seen_hashes.add(record["content_hash"])
                    except Exception:
                        continue

    def add(self, vector: List[float], metadata: Dict[str, Any], persist: bool = False):
        """Adds a vector with deduplication check."""
        # --- Deduplication layer (Feedback 5) ---
        c_hash = metadata.get("content_hash")
        if c_hash and c_hash in self.seen_hashes:
            return # Skip already indexed content

        vec_np = np.array([vector]).astype("float32")
        faiss.normalize_L2(vec_np)
        
        self.index.add(vec_np)
        self.metadata.append(metadata)
        if c_hash:
            self.seen_hashes.add(c_hash)
        
        if persist:
            self.save()

    def save(self, versioned: bool = False):
        """Persists with atomicity and optional versioning (Feedback 5)."""
        # 1. Base save
        faiss.write_index(self.index, self.index_path)
        
        # Atomic metadata save
        temp_meta = self.metadata_path + ".tmp"
        with open(temp_meta, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        os.rename(temp_meta, self.metadata_path)

        # 2. Versioning logic
        if versioned:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            v_index = os.path.join(self.storage_dir, f"vector_store_{ts}.faiss")
            faiss.write_index(self.index, v_index)
            print(f"[STORE] Created versioned index: {v_index}")

    def verify_relational_integrity(self, doc_id: str, page_refs: List[int]) -> bool:
        """
        Cross-verify check (Feedback 5):
        Ensures every page referenced in a PageIndex has at least one matching embedding.
        """
        doc_meta = [m for m in self.metadata if m.get("doc_id") == doc_id]
        if not doc_meta:
            return False
        
        vector_pages = set()
        for m in doc_meta:
            refs = m.get("page_refs", [])
            vector_pages.update(refs)
            
        return all(p in vector_pages for p in page_refs)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0: return []
        q_vec = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            results.append({"metadata": self.metadata[idx], "score": float(score)})
        return results
