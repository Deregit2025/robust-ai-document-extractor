"""
DocumentProfile Model

Defines the classification output of the Triage Agent.
This profile governs downstream extraction strategy routing.
"""

from pydantic import BaseModel, Field
from typing import Literal


class DocumentProfile(BaseModel):
    """
    Metadata profile describing a document before extraction.
    """

    doc_id: str = Field(..., description="Unique identifier for the document")

    origin_type: Literal[
        "native_digital",
        "scanned_image",
        "mixed",
        "form_fillable"
    ] = Field(..., description="Document origin classification")
    
    origin_confidence: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Confidence score for origin classification"
    )

    layout_complexity: Literal[
        "single_column",
        "multi_column",
        "table_heavy",
        "figure_heavy",
        "mixed"
    ] = Field(..., description="Layout structural complexity")
    
    layout_confidence: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Confidence score for layout complexity"
    )

    language: str = Field(..., description="Detected language code (e.g., 'en')")
    language_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score for language detection"
    )

    domain_hint: Literal[
        "financial",
        "legal",
        "technical",
        "medical",
        "general"
    ] = Field(..., description="Domain classification hint")
    
    domain_confidence: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Confidence score for domain classification hint"
    )

    estimated_extraction_cost: Literal[
        "fast_text_sufficient",
        "needs_layout_model",
        "needs_vision_model"
    ] = Field(..., description="Strategy tier decision output")

    def model_dump_json_pretty(self) -> str:
        """Utility helper for readable JSON export."""
        return self.model_dump_json(indent=4)