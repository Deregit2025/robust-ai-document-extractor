"""
Base Extractor

Defines the interface and shared utilities for all extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List
from src.models.extracted_document import ExtractedDocument


class BaseExtractor(ABC):
    """
    Abstract base class for all extraction strategies.
    """

    @abstractmethod
    def extract(self, file_path: str, doc_id: str = None) -> ExtractedDocument:
        """
        Extracts structured content from a document and returns
        an ExtractedDocument instance.

        Args:
            file_path (str): Path to the input PDF
            doc_id (str, optional): Document identifier. Derived from filename if not provided.

        Returns:
            ExtractedDocument: Normalized extraction output
        """
        pass

    def log_extraction(self, file_path: str, confidence: float, strategy_name: str) -> None:
        """
        Optional utility to log extraction info in the ledger.
        Silenced in favor of main.py's centralized and metadata-rich logging.
        """
        pass
