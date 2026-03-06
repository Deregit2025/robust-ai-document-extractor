"""
Structured Query Tool

Provides structured access to the FactTable ledger for numerical queries.
Leverages SQL backend for Mastery-level precision (Feedback 5).
"""

from typing import Optional
from src.data.fact_table import FactTable

class StructuredQueryTool:
    def __init__(self):
        self.fact_table = FactTable()

    def aggregate_numerical(self, query: str, doc_id: Optional[str] = None) -> str:
        """
        Uses SQL to find precise numerical facts (Feedback 5).
        """
        # Determine keywords to filter by (skip small words)
        important_words = [w for w in query.split() if len(w) > 4 and w.lower() not in ["total", "amount", "value", "report"]]
        hint = important_words[0] if important_words else ""
        
        print(f"[STRUCTURED-SQL] Searching for '{hint}' with numerical filter...")
        
        # Mastery: Use the SQL-powered FactTable to get numeric facts
        matches = self.fact_table.get_numerical_facts(keyword_hint=hint, doc_id=doc_id)
        
        if not matches:
            return "No precise numerical facts found in the relational store."
        
        results = []
        for m in matches[:15]: # Return top 15 for context (Mastery)
            results.append(
                f"[FACT-ID: {m['content_hash'][:8]}] Doc: {m['doc_id']} | Page: {m['page_refs']}\n"
                f"Data: {m['content'][:500]}..."
            )
            
        return "\n\n".join(results)
