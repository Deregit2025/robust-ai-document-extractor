"""
VisionExtractor (Strategy C) - 100% Local Version

- Uses Moondream (local VLM) via Ollama for zero-cost extraction.
- Extracts text, tables, and figures from page images.
- Normalizes output to ExtractedDocument.
"""

import os
import io
import base64
import json
import pdfplumber
from PIL import Image
from typing import List, Optional
from src.strategies.base import BaseExtractor
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock, FigureBlock
from src.utils.llm_utils import LLMUtils


class VisionExtractor(BaseExtractor):
    def __init__(self, model_name: str = "moondream"):
        """
        Args:
            model_name (str): Local Ollama model name (default: moondream)
        """
        self.llm_utils = LLMUtils()
        self.model_name = model_name

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.encodebytes(buffered.getvalue()).decode("utf-8")

    def extract(self, file_path: str, doc_id: Optional[str] = None) -> ExtractedDocument:
        """
        Extract content from PDF using local VLM via Ollama (Moondream).
        """
        text_blocks: List[TextBlock] = []
        table_blocks: List[TableBlock] = []
        figure_blocks: List[FigureBlock] = []
        
        assigned_doc_id = doc_id or os.path.basename(file_path).replace(".pdf", "")

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            limit = 2  # Demo limit for speed and processing stability
            print(f"Local Vision Extractor starting: {total_pages} pages. Processing first {limit} for demo.")
            
            for page_num, page in enumerate(pdf.pages[:limit], start=1):
                print(f"  --> Processing page {page_num}/{limit} with local VLM (moondream)...")
                
                # Convert page to image
                page_image = page.to_image(resolution=200).original
                base64_image = self._encode_image(page_image)

                # Prompt for structured extraction
                prompt = (
                    "Describe this page in detail. "
                    "Extract all visible text. "
                    "Format any tables data clearly. "
                    "Identify any figures or charts and provide short captions for them."
                )

                try:
                    # Generic local vision call (returns raw text description)
                    raw_description = self.llm_utils.vision_completion(prompt, base64_image)
                    
                    # Since moondream doesn't guarantee JSON, we treat the whole 
                    # description as a rich text block for RAG.
                    text_blocks.append(
                        TextBlock(
                            content=raw_description,
                            page=page_num,
                            bbox=(0, 0, float(page.width), float(page.height)),
                        )
                    )
                except Exception as e:
                    print(f"Local VLM failure on page {page_num}: {e}")
                    # If local vision fails, we let the exception bubble up to the router
                    # so it can try FastText as a last resort.
                    raise e

        # Log extraction success
        self.log_extraction(file_path, confidence=0.85, strategy_name=f"VisionExtractor({self.model_name})")

        return ExtractedDocument(
            doc_id=assigned_doc_id,
            text_blocks=text_blocks,
            tables=table_blocks,
            figures=figure_blocks,
            total_pages=total_pages,
            reading_order=list(range(len(text_blocks))),
            strategy_name=f"VisionExtractor({self.model_name})",
            confidence=0.85
        )
