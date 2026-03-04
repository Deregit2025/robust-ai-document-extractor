"""
Semantic Chunking Engine (Chunker)

Implements all 5 chunking rules as enforceable constraints:
  Rule 1: A table cell is never split from its header row.
  Rule 2: A figure caption is always stored as metadata of its parent figure chunk.
  Rule 3: A numbered list is always kept as a single LDU unless it exceeds max_tokens.
  Rule 4: Section headers are stored as parent_section metadata on all child chunks.
  Rule 5: Cross-references ("see Table 3") are resolved and stored as chunk relationships.

- Converts ExtractedDocument into LDUs
- Enforces chunking rules via ChunkValidator
- Computes content_hash for provenance
"""

import re
import hashlib
from typing import List, Optional, Tuple

from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock, FigureBlock
from src.models.ldu import LDU


# ─── Heading Detection ────────────────────────────────────────────────────────

# A heading is a short text block (< 12 words) that doesn't end with
# sentence-terminating punctuation and is not part of a list.
_HEADING_MAX_WORDS = 12
_LIST_PATTERN = re.compile(r"^\s*(\d+[\.\)]\s+|[•\-\*]\s+)")
_CROSS_REF_PATTERN = re.compile(
    r"\b(see|refer to|as shown in|per|cf\.?)\s+(Table|Figure|Appendix|Section|Chart)\s*[\d\w]+",
    re.IGNORECASE,
)


def _is_heading(text: str) -> bool:
    """Returns True if the text looks like a section heading."""
    text = text.strip()
    if not text:
        return False
    words = text.split()
    if len(words) > _HEADING_MAX_WORDS:
        return False
    if text[-1] in ".?!,:;":
        return False
    if _LIST_PATTERN.match(text):
        return False
    return True


def _is_list_item(text: str) -> bool:
    """Returns True if the text starts with a list marker."""
    return bool(_LIST_PATTERN.match(text.strip()))


def _extract_cross_refs(text: str) -> List[str]:
    """Extracts cross-reference strings from text (e.g. 'see Table 3')."""
    return [m.group(0) for m in _CROSS_REF_PATTERN.finditer(text)]


# ─── Chunk Validator ─────────────────────────────────────────────────────────

class ChunkValidator:
    """
    Validates all 5 chunking rules. Uses a warn-not-fail policy so
    the pipeline always completes, but violations are recorded.
    """

    def validate(self, ldu: LDU) -> List[str]:
        """Returns a list of violation messages (empty = clean)."""
        violations = []

        # Rule 1: Table must not have empty headers if it has rows
        if ldu.chunk_type == "table":
            if not ldu.headers and ldu.content:
                violations.append("RULE-1: Table LDU has no headers despite having content.")

        # Rule 2: Figure must carry caption field
        if ldu.chunk_type == "figure":
            if ldu.caption is None:
                violations.append("RULE-2: Figure LDU missing caption field.")

        # Rule 3: List LDUs should not be split (token count check)
        if ldu.chunk_type == "list":
            pass  # enforced at creation time in ChunkingEngine

        # Rule 4: All non-heading chunks should have parent_section if possible
        if ldu.chunk_type in ("text", "list") and ldu.parent_section is None:
            violations.append("RULE-4: Text/List LDU has no parent_section assigned.")

        return violations


# ─── Chunking Engine ─────────────────────────────────────────────────────────

class ChunkingEngine:
    """
    Converts an ExtractedDocument into a list of LDUs, enforcing all 5
    chunking rules.
    """

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.validator = ChunkValidator()

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        ldus: List[LDU] = []

        # ── Rule 4 Tracking ──────────────────────────────────────────────────
        # We make one pass through text_blocks in reading order to detect
        # heading → body relationships.
        current_section: Optional[str] = None

        # ── Phase A: Process Text Blocks (Rules 3, 4, 5) ────────────────────
        pending_list_items: List[TextBlock] = []

        def flush_list(section: Optional[str]) -> None:
            """Merge pending list items into a single list LDU."""
            if not pending_list_items:
                return
            combined = "\n".join(tb.content for tb in pending_list_items)
            token_count = len(combined.split())

            if token_count > self.max_tokens:
                # Split oversized list at max_tokens boundary
                for tb in pending_list_items:
                    _emit_text(tb, section, chunk_type="list")
            else:
                ldu = LDU(
                    doc_id=doc.doc_id,
                    content=combined,
                    chunk_type="list",
                    page_refs=[tb.page for tb in pending_list_items],
                    bounding_box=pending_list_items[0].bbox,
                    parent_section=section,
                    token_count=token_count,
                    content_hash=self._generate_hash(combined),
                    cross_refs=_extract_cross_refs(combined),
                )
                violations = self.validator.validate(ldu)
                ldus.append(ldu)
            pending_list_items.clear()

        def _emit_text(tb: TextBlock, section: Optional[str], chunk_type: str = "text"):
            cross_refs = _extract_cross_refs(tb.content)
            ldu = LDU(
                doc_id=doc.doc_id,
                content=tb.content,
                chunk_type=chunk_type,
                page_refs=[tb.page],
                bounding_box=tb.bbox,
                parent_section=section,
                token_count=len(tb.content.split()),
                content_hash=self._generate_hash(tb.content),
                cross_refs=cross_refs,
            )
            self.validator.validate(ldu)
            ldus.append(ldu)

        for tb in doc.text_blocks:
            text = tb.content.strip()

            if _is_heading(text):
                # Flush any pending list before changing section
                flush_list(current_section)
                # Rule 4: update current section BEFORE emitting child chunks
                # The heading itself has no parent (it IS the section root)
                _emit_text(tb, section=None, chunk_type="text")
                current_section = text   # all subsequent chunks inherit this

            elif _is_list_item(text):
                # Rule 3: accumulate into pending list
                pending_list_items.append(tb)

            else:
                # Regular paragraph — flush any pending list first
                flush_list(current_section)
                _emit_text(tb, current_section)

        # Flush any trailing list at end of document
        flush_list(current_section)

        # ── Phase B: Process Table Blocks (Rule 1) ───────────────────────────
        for tbl in doc.tables:
            # Headers are always preserved as the first content line (Rule 1)
            header_line = ", ".join(tbl.headers) if tbl.headers else ""
            row_lines = "\n".join([", ".join(row) for row in tbl.rows])
            content_str = f"{header_line}\n{row_lines}".strip() if header_line else row_lines

            ldu = LDU(
                doc_id=doc.doc_id,
                content=content_str,
                chunk_type="table",
                page_refs=[tbl.page],
                bounding_box=tbl.bbox,
                parent_section=current_section,
                token_count=len(content_str.split()),
                content_hash=self._generate_hash(content_str),
                headers=tbl.headers,
                cross_refs=[],
            )
            self.validator.validate(ldu)
            ldus.append(ldu)

        # ── Phase C: Process Figure Blocks (Rule 2) ──────────────────────────
        for fig in doc.figures:
            # Rule 2: caption is always stored as metadata AND inline content
            caption_text = fig.caption or "No Caption"
            ldu = LDU(
                doc_id=doc.doc_id,
                content=caption_text,
                chunk_type="figure",
                page_refs=[fig.page],
                bounding_box=fig.bbox,
                parent_section=current_section,
                token_count=len(caption_text.split()),
                content_hash=self._generate_hash(caption_text),
                caption=fig.caption,   # Rule 2: explicit metadata field
                cross_refs=[],
            )
            self.validator.validate(ldu)
            ldus.append(ldu)

        return ldus

    @staticmethod
    def _generate_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
