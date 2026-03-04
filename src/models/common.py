"""Common models shared across the application."""
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

class BBox(BaseModel):
    """Spatial bounding box coordinates."""
    x0: float = Field(..., description="Left x-coordinate")
    y0: float = Field(..., description="Top y-coordinate")
    x1: float = Field(..., description="Right x-coordinate")
    y1: float = Field(..., description="Bottom y-coordinate")

    @model_validator(mode='after')
    def validate_coordinates(self) -> Self:
        if self.x0 > self.x1:
            raise ValueError("x0 must be <= x1")
        if self.y0 > self.y1:
            raise ValueError("y0 must be <= y1")
        return self
