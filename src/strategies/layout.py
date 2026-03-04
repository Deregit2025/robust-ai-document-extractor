import os
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
        """
        from docling_core.types.doc.document import TextItem, TableItem, PictureItem
        print(f"Layout Extractor starting for {pdf_path}")

        # Run conversion
        result = self.converter.convert(pdf_path)
        doc = result.document

        text_blocks = []
        tables = []
        figures = []

        # Use Docling's built-in iteration for all elements
        for item, level in doc.iterate_items():
            if isinstance(item, TextItem):
                # Text blocks usually have a single box, but can be multi-prov
                bbox = item.prov[0].bbox if item.prov else None
                text_blocks.append(
                    TextBlock(
                        content=item.text,
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=(bbox.l, bbox.t, bbox.r, bbox.b) if bbox else (0,0,0,0),
                    )
                )
            elif isinstance(item, TableItem):
                # Extract table headers and rows
                # Note: item.data is the table representation
                # This depends on Docling version, usually item.data.to_list() works
                rows = []
                headers = []
                try:
                    # Attempt to get table data
                    if hasattr(item, "data") and item.data:
                        rows = item.data.to_list()
                        if rows:
                            headers = rows[0]
                except Exception:
                    pass

                bbox = item.prov[0].bbox if item.prov else None
                tables.append(
                    TableBlock(
                        headers=headers,
                        rows=rows,
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=(bbox.l, bbox.t, bbox.r, bbox.b) if bbox else (0,0,0,0),
                    )
                )
            elif isinstance(item, PictureItem):
                bbox = item.prov[0].bbox if item.prov else None
                figures.append(
                    FigureBlock(
                        caption=getattr(item, "caption", None),
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=(bbox.l, bbox.t, bbox.r, bbox.b) if bbox else (0,0,0,0),
                    )
                )

        return ExtractedDocument(
            doc_id=doc_id or pdf_path.split("/")[-1],
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=list(range(len(text_blocks) + len(tables) + len(figures))),
            total_pages=max([tb.page for tb in text_blocks] + [1]),
            strategy_name="LayoutExtractor",
            confidence=0.88
        )