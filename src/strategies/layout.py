import os
from typing import Optional
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from src.models.extracted_document import (
    ExtractedDocument,
    TextBlock,
    TableBlock,
    FigureBlock,
)
from src.models.common import BBox


def _safe_bbox(bbox) -> BBox:
    """
    Safely construct a BBox from a Docling bbox object.
    Docling occasionally returns inverted coordinates (l > r or t > b)
    on certain table cells and figures. We clamp them to ensure
    x0 <= x1 and y0 <= y1 always holds, preventing Pydantic ValidationError.
    """
    if bbox is None:
        return BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    l, t, r, b = float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)
    return BBox(
        x0=min(l, r),
        y0=min(t, b),
        x1=max(l, r),
        y1=max(t, b),
    )

class LayoutExtractor:
    """
    Layout-aware extraction using Docling via Python API.
    """

    def __init__(self):
        # Configure memory-efficient pipeline options
        options = PdfPipelineOptions()
        options.do_ocr = False  # Disable OCR to avoid memory allocation for long reports
        options.do_table_structure = True
        
        # Use RapidOCR with optimized settings for memory if needed
        # (RapidOCR is generally lighter than Tesseract)
        ocr_options = RapidOcrOptions()
        options.ocr_options = ocr_options
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )

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
                        bbox=_safe_bbox(bbox),
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
                        bbox=_safe_bbox(bbox),
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
                        bbox=_safe_bbox(bbox),
                    )
                )
                reading_order.append(global_index)
                global_index += 1

        # Use actual page count from docling rather than inferring from extracted blocks
        actual_pages = len(doc.pages) if hasattr(doc, 'pages') else max([tb.page for tb in text_blocks] + [1])

        return ExtractedDocument(
            doc_id=doc_id or pdf_path.split("/")[-1],
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
            total_pages=actual_pages,
            strategy_name="LayoutExtractor",
            confidence=0.88
        )