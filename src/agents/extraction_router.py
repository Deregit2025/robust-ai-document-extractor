"""
Extraction Router

Decides which extraction strategy to use based on DocumentProfile
and routes the document accordingly.
"""

from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
# VisionExtractor can be added later when ready
# from src.strategies.vision import VisionExtractor


class ExtractionRouter:
    def __init__(self):
        self.fast_text = FastTextExtractor()
        self.layout = LayoutExtractor()
        # self.vision = VisionExtractor()  # placeholder

    def route_extraction(self, doc_path: str, profile: DocumentProfile) -> ExtractedDocument:
        tier = profile.estimated_extraction_cost

        if tier == "fast_text_sufficient":
            return self.fast_text.extract(doc_path)
        elif tier == "needs_layout_model":
            return self.layout.extract(doc_path)
        elif tier == "needs_vision_model":
            # return self.vision.extract(doc_path)
            raise NotImplementedError("VisionExtractor not implemented yet.")
        else:
            raise ValueError(f"Unknown extraction tier: {tier}")