"""
QueryAgent (RAG Engine)

Two-phase retrieval:
  Phase 1 - PageIndex Traversal: Score each SectionNode against the query
            using keyword + entity matching to identify the top-3 relevant sections.
  Phase 2 - Filtered FAISS Search: Restrict vector search to chunks whose
            page_refs fall within the page ranges of the top-3 sections.

This significantly improves retrieval precision over pure vector search.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from src.utils.llm_utils import LLMUtils
from src.data.vector_store import LocalVectorStore
from src.models.page_index import PageIndex, SectionNode

# Directory where PageIndex JSONs are persisted
PROCESSED_DIR = "data/processed"


class PageIndexTraverser:
    """
    Traverses saved PageIndex trees to find the top-N most relevant sections
    for a given query. Uses keyword + entity scoring (no LLM at this stage,
    so it's fast and cheap).
    """

    def __init__(self, processed_dir: str = PROCESSED_DIR):
        self.processed_dir = processed_dir

    def _load_all_indexes(self) -> List[Tuple[str, PageIndex]]:
        """Load all *_index.json files from the processed directory."""
        indexes = []
        if not os.path.exists(self.processed_dir):
            return indexes
        for fname in os.listdir(self.processed_dir):
            if fname.endswith("_index.json"):
                path = os.path.join(self.processed_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    indexes.append((fname, PageIndex(**data)))
                except Exception:
                    continue
        return indexes

    def _score_section(self, node: SectionNode, query_terms: List[str]) -> float:
        """Score a SectionNode against a list of query keywords."""
        score = 0.0
        query_lower = " ".join(query_terms).lower()

        # 1. Match against LLM-generated summary
        if node.summary:
            for term in query_terms:
                if term.lower() in node.summary.lower():
                    score += 2.0  # Summary match is high-signal

        # 2. Match against extracted entities
        for entity in node.key_entities:
            for term in query_terms:
                if term.lower() in entity.lower():
                    score += 3.0  # Entity match is highest-signal

        # 3. Match against section title
        if node.title:
            for term in query_terms:
                if term.lower() in node.title.lower():
                    score += 1.5

        # 4. Recurse into child sections (with a decay factor)
        for child in node.child_sections:
            score += 0.5 * self._score_section(child, query_terms)

        return score

    def get_top_sections(
        self,
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Traverse all PageIndex trees and return top_k sections most relevant
        to the query.

        Returns a list of dicts with: doc_id, page_start, page_end, title, summary, score.
        """
        query_terms = [w for w in query.split() if len(w) > 2]  # skip stop words
        all_scored = []

        indexes = self._load_all_indexes()
        for fname, page_index in indexes:
            # Filter by doc_id if requested
            if doc_id and page_index.doc_id != doc_id:
                continue

            for node in page_index.root_sections:
                score = self._score_section(node, query_terms)
                if score > 0:
                    all_scored.append({
                        "doc_id": page_index.doc_id,
                        "page_start": node.page_start,
                        "page_end": node.page_end,
                        "title": node.title,
                        "summary": node.summary or "",
                        "entities": node.key_entities,
                        "score": score
                    })

        # Sort by score descending, return top_k
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        return all_scored[:top_k]


class QueryAgent:
    def __init__(self, llm_utils: Optional[LLMUtils] = None):
        self.llm = llm_utils or LLMUtils()
        self.vector_store = LocalVectorStore()
        self.traverser = PageIndexTraverser()

    def answer(self, query: str, doc_id: Optional[str] = None) -> str:
        """
        Answers a query using two-phase RAG:
          1. PageIndex traversal to identify the top-3 relevant sections.
          2. Filtered FAISS vector search restricted to those sections' page ranges.
        """
        # ─── Phase 1: PageIndex Traversal ────────────────────────────────────
        top_sections = self.traverser.get_top_sections(query, top_k=3, doc_id=doc_id)

        # Build a set of allowed page numbers from top sections
        allowed_pages: set = set()
        section_context_lines = []
        for s in top_sections:
            pages = range(s["page_start"], s["page_end"] + 1)
            allowed_pages.update(pages)
            section_context_lines.append(
                f"  [{s['doc_id']}] {s['title']} (p.{s['page_start']}–{s['page_end']}) "
                f"— Score: {s['score']:.1f}"
            )

        # ─── Phase 2: Filtered Vector Search ─────────────────────────────────
        query_vec = self.llm.get_embeddings(query)
        raw_results = self.vector_store.search(query_vec, k=10)  # retrieve more, then filter

        # Filter results to chunks within the top sections' page ranges
        if allowed_pages:
            filtered = [
                r for r in raw_results
                if any(
                    p in allowed_pages
                    for p in (r["metadata"].get("page_refs") or [])
                )
            ]
        else:
            filtered = raw_results  # fallback: use all results if traversal found nothing

        # Further filter by doc_id if provided
        if doc_id:
            filtered = [r for r in filtered if r["metadata"].get("doc_id") == doc_id]

        # Fallback: use top raw results if filter is too aggressive
        if not filtered:
            filtered = raw_results[:3]

        if not filtered:
            return "I couldn't find any relevant information in the processed documents."

        # ─── Phase 3: Grounded Answer Generation ─────────────────────────────
        context_parts = []
        for r in filtered[:5]:
            meta = r["metadata"]
            context_parts.append(
                f"[Doc: {meta.get('doc_id')} | Type: {meta.get('chunk_type')}]\n"
                f"{meta.get('summary', 'No summary available.')}"
            )
        context = "\n---\n".join(context_parts)

        nav_trace = (
            "\n\nNavigation Trace (PageIndex Traversal):\n" + "\n".join(section_context_lines)
            if section_context_lines else ""
        )

        prompt = (
            "You are a professional document intelligence assistant. "
            "Answer the user query based ONLY on the provided context. "
            "If the answer is not in the context, say 'I don't know.'\n\n"
            f"Context:\n{context}"
            f"{nav_trace}\n\n"
            f"Query: {query}"
        )

        return self.llm.completions([{"role": "user", "content": prompt}])

    def get_navigation_trace(self, query: str, doc_id: Optional[str] = None) -> str:
        """
        Returns a human-readable trace of which document sections were
        selected during PageIndex traversal. Useful for precision evaluation.
        """
        top_sections = self.traverser.get_top_sections(query, top_k=3, doc_id=doc_id)
        if not top_sections:
            return "No relevant sections found in the PageIndex."

        lines = ["PageIndex Traversal Results:", "=" * 40]
        for i, s in enumerate(top_sections, 1):
            lines.append(
                f"{i}. [{s['doc_id']}] {s['title']}\n"
                f"   Pages: {s['page_start']}–{s['page_end']} | Score: {s['score']:.2f}\n"
                f"   Summary: {s['summary'][:150]}..."
            )
        return "\n".join(lines)