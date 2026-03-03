"""
Provenance Models

Defines traceable citation structures used in answers,
audit mode, and verification workflows.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple


BoundingBox = Tuple[float, float, float, float]


class ProvenanceEntry(BaseModel):
    """
    Single citation reference pointing to a precise location
    inside a document.
    """

    document_name: str = Field(..., description="Original document filename")
    doc_id: str = Field(..., description="Internal document identifier")

    page_number: int = Field(..., ge=1, description="Source page number")

    bounding_box: BoundingBox = Field(
        ...,
        description="Spatial bounding box coordinates (x0, y0, x1, y1)"
    )

    content_hash: str = Field(
        ...,
        description="Hash of the content for integrity verification"
    )

    excerpt: str = Field(
        ...,
        description="Short text excerpt from source for human verification"
    )


class ProvenanceChain(BaseModel):
    """
    Collection of provenance entries supporting a system answer.
    """

    answer: str = Field(..., description="Generated answer")

    citations: List[ProvenanceEntry] = Field(
        default_factory=list,
        description="List of source citations backing the answer"
    )

    verified: bool = Field(
        default=False,
        description="Whether the claim has been verified against source"
    )