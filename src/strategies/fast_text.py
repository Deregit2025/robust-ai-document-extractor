"""
FastTextExtractor (Strategy A)

- Uses pdfplumber for quick text extraction
- Computes confidence score per page
- Escalates if confidence < threshold
"""

import os
import pdfplumber
from typing import Optional
from src.strategies.base import BaseExtractor
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock
from src.models.common import BBox

class FastTextExtractor(BaseExtractor):
    def __init__(self, min_char_count: int = 50, max_image_ratio: float = 0.5):
        """
        Args:
            min_char_count (int): Minimum characters per page to consider confident
            max_image_ratio (float): Maximum ratio of page area covered by images
        """
        self.min_char_count = min_char_count
        self.max_image_ratio = max_image_ratio

    def compute_confidence(self, page) -> float:
        """
        Confidence based on character count, image coverage, and font presence.
        """
        text = page.extract_text() or ""
        char_count = len(text)
        
        # Check fonts - are these real text chars or just vector paths?
        valid_font_chars = 0
        total_page_chars = len(page.chars) if hasattr(page, 'chars') and page.chars else 0
        
        if total_page_chars > 0:
            for char in page.chars:
                if char.get("fontname"):
                    valid_font_chars += 1
                    
            font_ratio = valid_font_chars / total_page_chars
        else:
            font_ratio = 1.0 if char_count == 0 else 0.5 # If no chars structure but it extracted text? Rare edge case
            
        page_area = page.width * page.height
        image_area = sum((img["width"] * img["height"] for img in page.images), 0)
        image_ratio = image_area / page_area if page_area > 0 else 0

        # Confidence is 0–1
        char_conf = min(char_count / self.min_char_count, 1.0)
        
        # Multiply text confidence by font ratio (penalize vector paths masquerading as text)
        char_conf = char_conf * font_ratio
        
        image_conf = 1.0 if image_ratio <= self.max_image_ratio else max(0.0, 1 - (image_ratio - self.max_image_ratio))
        confidence = (char_conf + image_conf) / 2
        return confidence

    def extract(self, file_path: str, doc_id: Optional[str] = None) -> ExtractedDocument:
        """
        Extract text and tables as structured blocks
        """
        assigned_doc_id = doc_id or os.path.basename(file_path).replace(".pdf", "")
        with pdfplumber.open(file_path) as pdf:
            text_blocks = []
            table_blocks = []
            page_confidences = []

            for i, page in enumerate(pdf.pages, start=1):
                confidence = self.compute_confidence(page)
                page_confidences.append(confidence)

                # Text block
                text = page.extract_text() or ""
                if text.strip():
                    text_blocks.append(
                        TextBlock(
                            content=text,
                            page=i,
                            bbox=BBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height))
                        )
                    )

                # Tables
                tables = page.find_tables()
                for table in tables:
                    table_rows = []
                    for row in table.extract():
                        if row: table_rows.append([str(cell or "") for cell in row])
                    if table_rows:
                        table_blocks.append(
                            TableBlock(
                                headers=table_rows[0],
                                rows=table_rows[1:],
                                page=i,
                                bbox=BBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height))
                            )
                        )

        avg_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0

        # Log extraction
        self.log_extraction(file_path, confidence=avg_confidence, strategy_name="FastTextExtractor")

        return ExtractedDocument(
            doc_id=assigned_doc_id,
            text_blocks=text_blocks,
            tables=table_blocks,
            figures=[],
            total_pages=len(pdf.pages),
            reading_order=list(range(len(text_blocks) + len(table_blocks))),
            strategy_name="FastTextExtractor",
            confidence=avg_confidence
        )


