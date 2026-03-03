"""
ExtractedDocument Model

Normalized representation that ALL extraction strategies
must output, regardless of extraction method.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


BoundingBox = Tuple[float, float, float, float]


class TextBlock(BaseModel):
    content: str
    page: int
    bbox: BoundingBox


class TableBlock(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    page: int
    bbox: BoundingBox


class FigureBlock(BaseModel):
    caption: Optional[str] = None
    page: int
    bbox: BoundingBox


class ExtractedDocument(BaseModel):
    """
    Unified normalized document representation.
    """

    doc_id: str

    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableBlock] = Field(default_factory=list)
    figures: List[FigureBlock] = Field(default_factory=list)

    reading_order: List[int] = Field(
        default_factory=list,
        description="Ordered list of content block indices"
    )

    total_pages: int