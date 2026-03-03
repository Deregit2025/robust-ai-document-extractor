"""
VisionExtractor (Strategy C)

- Uses Vision-Language Model (VLM) for scanned or low-confidence pages
- Extracts text, tables, and figures from images
- Normalizes output to ExtractedDocument
"""

from src.strategies.base import BaseExtractor
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock, FigureBlock
import pdfplumber
from typing import List
from PIL import Image
import io

# OpenRouter stub for VLM
try:
    import openrouter
except ImportError:
    print("OpenRouter client not installed. Run `pip install openrouter`.")


class VisionExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gpt-4o-mini:vision"):
        """
        Args:
            model_name (str): VLM model name for extraction
        """
        self.model_name = model_name
        # Example OpenRouter client initialization
        # self.client = openrouter.Client(api_key="YOUR_API_KEY")

    def extract(self, file_path: str) -> ExtractedDocument:
        """
        Extract content from scanned PDF using VLM
        """
        text_blocks: List[TextBlock] = []
        table_blocks: List[TableBlock] = []
        figure_blocks: List[FigureBlock] = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Convert page to image
                page_image = page.to_image(resolution=300).original
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                page_image.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()

                # VLM API call (stub)
                # Replace with actual call, e.g., self.client.predict(...)
                # For demo, we simulate output
                vlm_text = f"Extracted text from page {page_num} (simulated)"
                vlm_tables = []  # Normally a list of extracted tables
                vlm_figures = []  # Normally a list of figures with captions

                # Build TextBlock
                text_blocks.append(
                    TextBlock(
                        content=vlm_text,
                        chunk_type="text",
                        page_refs=[page_num],
                        bounding_box=[0, 0, page.width, page.height],
                    )
                )

                # Build TableBlocks
                for table in vlm_tables:
                    table_blocks.append(
                        TableBlock(
                            headers=table.get("headers", []),
                            rows=table.get("rows", []),
                            page_refs=[page_num],
                            bounding_box=[0, 0, page.width, page.height],
                        )
                    )

                # Build FigureBlocks
                for fig in vlm_figures:
                    figure_blocks.append(
                        FigureBlock(
                            caption=fig.get("caption", ""),
                            page_refs=[page_num],
                            bounding_box=[0, 0, page.width, page.height],
                        )
                    )

        # Log extraction
        self.log_extraction(file_path, confidence=0.95, strategy_name="VisionExtractor")

        return ExtractedDocument(
            text_blocks=text_blocks,
            table_blocks=table_blocks,
            figure_blocks=figure_blocks,
            page_count=len(pdf.pages),
        )


# Quick test
if __name__ == "__main__":
    extractor = VisionExtractor()
    doc_path = "../../data/raw/Audit_Report_-_2023.pdf"  # scanned PDF
    extracted = extractor.extract(doc_path)
    print(extracted)