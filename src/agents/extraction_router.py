"""
Extraction Router

Decides which extraction strategy to use based on DocumentProfile
and routes the document accordingly.
"""

import os
from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor


class ExtractionRouter:
    def __init__(self):
        self.fast_text = FastTextExtractor()
        self.layout = LayoutExtractor()
        self.vision = VisionExtractor()

    def route_extraction(self, doc_path: str, profile: DocumentProfile) -> ExtractedDocument:
        tier = profile.estimated_extraction_cost
        doc_id = profile.doc_id

        try:
            if tier == "fast_text_sufficient":
                return self.fast_text.extract(doc_path)
            elif tier == "needs_layout_model":
                return self.layout.extract(doc_path, doc_id)
            elif tier == "needs_vision_model":
                return self.vision.extract(doc_path, doc_id)
            else:
                return self.vision.extract(doc_path, doc_id)
        except Exception as e:
            print(f"[RECOVERY] Strategy {tier} failed for {doc_id}: {e}")
            print("[RECOVERY] Falling back to FastTextExtractor (Tier A) for guaranteed completion...")
            try:
                # FastText is local and doesn't require credits, making it a reliable last resort
                return self.fast_text.extract(doc_path)
            except Exception as f_err:
                print(f"[CRITICAL] Last resort fallback failed: {f_err}")
                return ExtractedDocument(
                    doc_id=doc_id,
                    text_blocks=[],
                    tables=[],
                    figures=[],
                    total_pages=1,
                    strategy_name="TOTAL_FAILURE",
                    confidence=0.0
                )
