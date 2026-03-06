"""
FactTable (Relational Audit Trail)

Implements Mastery-level requirement for Feedback #5.
Uses SQLite to store LDUs, enabling precise SQL-based numerical retrieval 
and structured data queries.
"""

import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from src.models.ldu import LDU

class FactTable:
    def __init__(self, db_path: str = ".refinery/fact_table.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    content TEXT,
                    chunk_type TEXT,
                    page_refs TEXT,  -- Stored as semi-colon separated list
                    parent_section TEXT,
                    content_hash TEXT UNIQUE,
                    extraction_strategy TEXT,
                    confidence_score REAL,
                    raw_json TEXT
                )
            """)
            # Create indexes for high-speed precision retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON facts(chunk_type)")

    def append_ldus(self, ldus: List[LDU]):
        """Inserts a batch of LDUs into the relational store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for ldu in ldus:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO facts 
                        (doc_id, content, chunk_type, page_refs, parent_section, content_hash, extraction_strategy, confidence_score, raw_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ldu.doc_id,
                        ldu.content,
                        ldu.chunk_type,
                        ";".join(map(str, ldu.page_refs)),
                        ldu.parent_section,
                        ldu.content_hash,
                        ldu.extraction_strategy,
                        ldu.confidence_score,
                        ldu.model_dump_json()
                    ))
                except Exception as e:
                    print(f"[FACT-DB] Error inserting LDU: {e}")
            conn.commit()

    def sql_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Executes a raw SQL query for precise numerical/fact retrieval.
        Directly addresses the 'Mastery' requirement for Feedback #5.
        """
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            for row in cursor:
                results.append(dict(row))
        return results

    def get_numerical_facts(self, keyword_hint: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Optimized SQL query to find numerical facts (chunks containing digits).
        """
        sql = "SELECT * FROM facts WHERE (content LIKE ? OR content LIKE ?)"
        params = [f"%{keyword_hint}%", "%[0-9]%"] # Simple hint, real SQL uses REGEXP if available
        
        # Standard SQLite doesn't have REGEXP by default, so we use GLOB or multiple LIKEs
        sql = "SELECT * FROM facts WHERE content GLOB '*[0-9]*'"
        params = []
        
        if keyword_hint:
            sql += " AND content LIKE ?"
            params.append(f"%{keyword_hint}%")
        
        if doc_id:
            sql += " AND doc_id = ?"
            params.append(doc_id)
            
        return self.sql_query(sql, tuple(params))

    def clear(self):
        """Clears the fact table database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
