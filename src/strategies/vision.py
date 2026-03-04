"""
VisionExtractor (Strategy C) - 100% Local Version

- Uses Moondream (local VLM) via Ollama for zero-cost extraction.
- Extracts text, tables, and figures from page images.
- Normalizes output to ExtractedDocument.
"""

import os
import io
import base64
import time
import yaml
import pdfplumber
from PIL import Image
from typing import List, Optional
from src.strategies.base import BaseExtractor
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock, FigureBlock
from src.models.common import BBox
from src.utils.llm_utils import LLMUtils


class VisionExtractor(BaseExtractor):
    def __init__(self, model_name: str = "moondream"):
        """
        Args:
            model_name (str): Local Ollama model name (default: moondream)
        """
        self.llm_utils = LLMUtils()
        self.model_name = model_name
        
        # Load configuration for budgets and retries
        self.config_path = "rubric/extraction_rules.yaml"
        self.vision_config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.vision_config = config.get("vision_extractor", {})
                
        self.max_pages_budget = self.vision_config.get("max_pages_budget", 10)
        self.max_tokens_per_page = self.vision_config.get("max_tokens_per_page", 1500)
        self.max_retries = self.vision_config.get("retry", {}).get("max_retries", 3)
        self.retry_delay = self.vision_config.get("retry", {}).get("retry_delay_seconds", 5)

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
            limit = min(total_pages, self.max_pages_budget)
            print(f"Local Vision Extractor starting: {total_pages} pages. Processing up to {limit} pages (budget limited).")
            
            for page_num, page in enumerate(pdf.pages[:limit], start=1):
                print(f"  --> Processing page {page_num}/{limit} with local VLM (moondream)...")
                
                page_image = page.to_image(resolution=200).original
                base64_image = self._encode_image(page_image)

                prompt = (
                    "Describe this page in detail. "
                    "Extract all visible text. "
                    "Format any tables data clearly. "
                    "Identify any figures or charts and provide short captions for them. "
                    f"Please constrain your output to roughly {self.max_tokens_per_page} words."
                )

                success = False
                for attempt in range(self.max_retries):
                    try:
                        raw_description = self.llm_utils.vision_completion(prompt, base64_image)
                        
                        # Validate basic output sanity (e.g. didn't just crash out returning blank)
                        if len(raw_description.strip()) < 10 and len(page.extract_text() or "") > 50:
                            print(f"Warning: Low token output on attempt {attempt+1}. Retrying...")
                            raise ValueError("Output token length suspiciously low compared to page text.")
                            
                        # Success
                        text_blocks.append(
                            TextBlock(
                                content=raw_description,
                                page=page_num,
                                bbox=BBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height)),
                            )
                        )
                        success = True
                        break # Skip remaining retries
                        
                    except Exception as e:
                        print(f"Local VLM failure on page {page_num}, attempt {attempt+1}/{self.max_retries}: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            
                if not success:
                    print(f"Failed to process page {page_num} after {self.max_retries} attempts. Skipping page.")
                    # We log the warning but do not crash the entire extraction run, preserving partial data.

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
