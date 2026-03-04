"""
LDU Model (Logical Document Unit)

Represents a semantically coherent, self-contained unit
of document content suitable for retrieval and indexing.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import hashlib


BoundingBox = Tuple[float, float, float, float]


class LDU(BaseModel):
    """
    Logical Document Unit.

    This is the atomic retrieval object stored in the vector store.
    """

    doc_id: str = Field(..., description="Document identifier")

    content: str = Field(..., description="Text content of this unit")

    chunk_type: str = Field(
        ...,
        description="Type of chunk: text | table | figure | list | section"
    )

    page_refs: List[int] = Field(
        ...,
        description="Pages this chunk spans"
    )

    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Primary bounding box reference for provenance"
    )

    parent_section: Optional[str] = Field(
        None,
        description="Section title this chunk belongs to"
    )

    token_count: int = Field(
        ...,
        ge=0,
        description="Estimated token count of this chunk"
    )

    content_hash: str = Field(
        ...,
        description="Stable content hash for verification and provenance"
    )

    headers: Optional[List[str]] = Field(
        None,
        description="Table headers if this is a table chunk"
    )

    caption: Optional[str] = Field(
        None,
        description="Figure caption if this is a figure chunk"
    )

    cross_refs: Optional[List[str]] = Field(
        None,
        description="Cross-references found in this chunk (e.g. 'see Table 3')"
    )

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """
        Generates a SHA256 hash for stable provenance verification.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        doc_id: str,
        content: str,
        chunk_type: str,
        page_refs: List[int],
        token_count: int,
        bounding_box: Optional[BoundingBox] = None,
        parent_section: Optional[str] = None,
        headers: Optional[List[str]] = None,
        caption: Optional[str] = None,
    ):
        """
        Factory constructor that auto-generates content_hash.
        """
        return cls(
            doc_id=doc_id,
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bounding_box,
            parent_section=parent_section,
            token_count=token_count,
            content_hash=cls.generate_content_hash(content),
            headers=headers,
            caption=caption
        )