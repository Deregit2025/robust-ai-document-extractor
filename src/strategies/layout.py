# src/strategies/layout.py

from typing import Optional
from docling.document_converter import DocumentConverter
from src.models.extracted_document import (
    ExtractedDocument,
    TextBlock,
    TableBlock,
    FigureBlock,
)

class LayoutExtractor:
    """
    Layout-aware extraction using Docling via Python API.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def extract(
        self, pdf_path: str, doc_id: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Convert a PDF to Docling’s structured document and normalize it
        into an ExtractedDocument for the pipeline.
        """
        # Run conversion
        result = self.converter.convert(pdf_path)
        doc = result.document

        text_blocks = []
        tables = []
        figures = []

        # Collect all pages with their extracted items
        for page in doc.pages:
            # Text segments
            for block in getattr(page, "text_blocks", []):
                text_blocks.append(
                    TextBlock(
                        content=block.text,
                        page=page.page_number,
                        bbox=(block.x0, block.y0, block.x1, block.y1),
                    )
                )

            # Tables
            for table in getattr(page, "tables", []):
                tables.append(
                    TableBlock(
                        headers=table.headers,
                        rows=table.rows,
                        page=page.page_number,
                        bbox=(table.x0, table.y0, table.x1, table.y1),
                    )
                )

            # Figures
            for fig in getattr(page, "figures", []):
                figures.append(
                    FigureBlock(
                        caption=getattr(fig, "caption", None),
                        page=page.page_number,
                        bbox=(fig.x0, fig.y0, fig.x1, fig.y1),
                    )
                )

        return ExtractedDocument(
            doc_id=doc_id or pdf_path.split("/")[-1],
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=list(range(len(text_blocks) + len(tables) + len(figures))),
            total_pages=len(doc.pages),
        )