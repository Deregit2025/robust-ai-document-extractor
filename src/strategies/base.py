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
    def extract(self, file_path: str) -> ExtractedDocument:
        """
        Extracts structured content from a document and returns
        an ExtractedDocument instance.

        Args:
            file_path (str): Path to the input PDF

        Returns:
            ExtractedDocument: Normalized extraction output
        """
        pass

    def log_extraction(self, file_path: str, confidence: float, strategy_name: str) -> None:
        """
        Optional utility to log extraction info in the ledger.
        """
        from datetime import datetime
        import json
        import os

        ledger_path = ".refinery/extraction_ledger.jsonl"
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "document": file_path,
            "strategy_used": strategy_name,
            "confidence_score": confidence
        }

        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")