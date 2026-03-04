import os
from typing import Optional
from docling.document_converter import DocumentConverter
from src.models.extracted_document import (
    ExtractedDocument,
    TextBlock,
    TableBlock,
    FigureBlock,
)
from src.models.common import BBox

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
        reading_order = []
        global_index = 0

        # Use Docling's built-in iteration for all elements
        # Iterate docling elements sequentially to preserve true reading order
        for item, level in doc.iterate_items():
            if isinstance(item, TextItem):
                bbox = item.prov[0].bbox if item.prov else None
                text_blocks.append(
                    TextBlock(
                        content=item.text,
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=BBox(x0=bbox.l, y0=bbox.t, x1=bbox.r, y1=bbox.b) if bbox else BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0),
                    )
                )
                reading_order.append(global_index)
                global_index += 1
            elif isinstance(item, TableItem):
                rows = []
                headers = []
                # Use robust Docling export method if available
                if hasattr(item, "export_to_dataframe"):
                    try:
                        df = item.export_to_dataframe()
                        headers = df.columns.tolist() if not df.empty else []
                        rows = df.values.tolist() if not df.empty else []
                    except Exception:
                        pass
                
                # Fallback to internal iteration over Grid cells to manually construct table if dataframe export failed
                if not rows and hasattr(item, "data") and item.data and hasattr(item.data, "table_cells"):
                    try:
                        max_row = max(cell.start_row_offset_idx for cell in item.data.table_cells) + 1
                        max_col = max(cell.start_col_offset_idx for cell in item.data.table_cells) + 1
                        grid = [["" for _ in range(max_col)] for _ in range(max_row)]
                        for cell in item.data.table_cells:
                            grid[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text
                        if grid:
                            headers = grid[0]
                            rows = grid[1:]
                    except Exception:
                        pass

                bbox = item.prov[0].bbox if item.prov else None
                tables.append(
                    TableBlock(
                        headers=headers,
                        rows=rows,
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=BBox(x0=bbox.l, y0=bbox.t, x1=bbox.r, y1=bbox.b) if bbox else BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0),
                    )
                )
                reading_order.append(global_index)
                global_index += 1
            elif isinstance(item, PictureItem):
                bbox = item.prov[0].bbox if item.prov else None
                figures.append(
                    FigureBlock(
                        caption=getattr(item, "caption", None),
                        page=item.prov[0].page_no if item.prov else 1,
                        bbox=BBox(x0=bbox.l, y0=bbox.t, x1=bbox.r, y1=bbox.b) if bbox else BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0),
                    )
                )
                reading_order.append(global_index)
                global_index += 1

        return ExtractedDocument(
            doc_id=doc_id or pdf_path.split("/")[-1],
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
            total_pages=max([tb.page for tb in text_blocks] + [1]),
            strategy_name="LayoutExtractor",
            confidence=0.88
        )