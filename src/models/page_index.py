"""
PageIndex Models

Defines the hierarchical navigation structure
built over a document for intelligent retrieval.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SectionNode(BaseModel):
    """
    Represents a section in the hierarchical PageIndex tree.
    """

    title: str = Field(..., description="Section title")

    page_start: int = Field(..., ge=1, description="Starting page number")
    page_end: int = Field(..., ge=1, description="Ending page number")

    child_sections: List["SectionNode"] = Field(
        default_factory=list,
        description="Nested subsections"
    )

    key_entities: List[str] = Field(
        default_factory=list,
        description="Named entities extracted from this section"
    )

    summary: Optional[str] = Field(
        None,
        description="2–3 sentence LLM-generated summary"
    )

    data_types_present: List[str] = Field(
        default_factory=list,
        description="Types of data in this section (tables, figures, equations, etc.)"
    )


class PageIndex(BaseModel):
    """
    Root container for the document navigation tree.
    """

    doc_id: str
    root_sections: List[SectionNode]