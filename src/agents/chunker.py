"""
Semantic Chunking Engine (Chunker)

- Converts ExtractedDocument into LDUs
- Enforces chunking rules
- Computes content_hash for provenance
"""

from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock, FigureBlock
from src.models.ldu import LDU
from typing import List
import hashlib


class ChunkValidator:
    """
    Validates chunking rules.
    Ensures LDU integrity: tables with headers, captions with figures, etc.
    """
    @staticmethod
    def validate(ldu: LDU):
        # Rule 1: table cell not split from header
        if ldu.chunk_type == "table":
            if not hasattr(ldu, "headers") or not ldu.headers:
                raise ValueError("Table LDU missing headers")

        # Rule 2: figure caption present
        if ldu.chunk_type == "figure":
            if not hasattr(ldu, "caption"):
                raise ValueError("Figure LDU missing caption")

        # Rule 3: section metadata exists
        if not hasattr(ldu, "parent_section"):
            ldu.parent_section = None  # default to None

        # Rule 4 & 5: cross-reference and numbered list handling can be added later


class ChunkingEngine:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.validator = ChunkValidator()

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        ldus: List[LDU] = []

        # Process text blocks
        for tb in doc.text_blocks:
            content_hash = self._generate_hash(tb.content)
            ldu = LDU(
                content=tb.content,
                chunk_type="text",
                page_refs=tb.page_refs,
                bounding_box=tb.bounding_box,
                parent_section=None,
                token_count=len(tb.content.split()),
                content_hash=content_hash,
            )
            self.validator.validate(ldu)
            ldus.append(ldu)

        # Process table blocks
        for tbl in doc.table_blocks:
            # Flatten table into string for content
            content_str = "\n".join([", ".join(row) for row in tbl.rows])
            content_hash = self._generate_hash(content_str)
            ldu = LDU(
                content=content_str,
                chunk_type="table",
                page_refs=tbl.page_refs,
                bounding_box=tbl.bounding_box,
                parent_section=None,
                token_count=sum(len(row) for row in tbl.rows),
                content_hash=content_hash,
            )
            ldu.headers = tbl.headers
            self.validator.validate(ldu)
            ldus.append(ldu)

        # Process figure blocks
        for fig in doc.figure_blocks:
            content_hash = self._generate_hash(fig.caption)
            ldu = LDU(
                content=fig.caption,
                chunk_type="figure",
                page_refs=fig.page_refs,
                bounding_box=fig.bounding_box,
                parent_section=None,
                token_count=len(fig.caption.split()),
                content_hash=content_hash,
            )
            ldu.caption = fig.caption
            self.validator.validate(ldu)
            ldus.append(ldu)

        return ldus

    @staticmethod
    def _generate_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Quick test
if __name__ == "__main__":
    from src.strategies.fast_text import FastTextExtractor
    from src.agents.triage import TriageAgent
    from src.agents.extraction_router import ExtractionRouter

    doc_path = "../../data/raw/CBE_ANNUAL_REPORT_2023_24.pdf"

    triage = TriageAgent()
    profile = triage.profile_document(doc_path)

    router = ExtractionRouter()
    extracted = router.route(profile, doc_path)

    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)

    print(f"Generated {len(ldus)} LDUs")
    for l in ldus[:3]:  # print first 3 LDUs
        print(l)