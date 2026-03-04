"""
ExtractedDocument Model

Normalized representation that ALL extraction strategies
must output, regardless of extraction method.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from src.models.common import BBox


class TextBlock(BaseModel):
    content: str
    page: int
    bbox: BBox


class TableBlock(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    page: int
    bbox: BBox


class FigureBlock(BaseModel):
    caption: Optional[str] = None
    page: int
    bbox: BBox


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

    escalation_history: List[str] = Field(
        default_factory=list,
        description="Log of extraction strategies attempted and failed"
    )

    needs_human_review: bool = Field(
        default=False,
        description="Flag denoting if all tiers failed or yielded critically low confidence"
    )

    total_pages: int
    strategy_name: Optional[str] = None
    confidence: Optional[float] = None
