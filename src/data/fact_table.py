"""
FactTable (Audit Trail)

Stores validated LDUs and their extraction metadata in a structured, 
ledger-style JSONL format for accountability and downstream RAG.
"""

import json
import os
from typing import List, Optional
from src.models.ldu import LDU

class FactTable:
    def __init__(self, storage_path: str = ".refinery/fact_table.jsonl"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def append_ldus(self, ldus: List[LDU]):
        """
        Appends a batch of LDUs to the fact table.
        """
        with open(self.storage_path, "a", encoding="utf-8") as f:
            for ldu in ldus:
                f.write(ldu.model_dump_json() + "\n")

    def get_by_doc_id(self, doc_id: str) -> List[LDU]:
        """
        Retrieves all facts associated with a document.
        """
        results = []
        if not os.path.exists(self.storage_path):
            return results
        
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("doc_id") == doc_id:
                    results.append(LDU(**data))
        return results

    def clear(self):
        """Clears the fact table."""
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
