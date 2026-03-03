"""
Triage Agent

Responsible for:
- Detecting document origin type (digital vs scanned)
- Classifying layout complexity
- Producing domain hints
- Estimating extraction cost tier
"""

import os
from typing import Optional
import pdfplumber

from src.models.document_profile import DocumentProfile


class TriageAgent:
    def __init__(self):
        # Keyword lists for domain detection
        self.financial_keywords = ["revenue", "income", "balance sheet", "profit"]
        self.legal_keywords = ["contract", "agreement", "plaintiff", "hereby"]
        self.technical_keywords = ["algorithm", "performance", "implementation"]
        self.medical_keywords = ["patient", "diagnosis", "treatment"]

    def classify_origin_type(self, file_path: str) -> str:
        """Heuristic: mostly empty or image pages → scanned"""
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            scanned_pages = 0
            for page in pdf.pages:
                char_count = len(page.extract_text() or "")
                if char_count < 50:
                    scanned_pages += 1
            return "scanned_image" if scanned_pages / total_pages > 0.5 else "native_digital"

    def classify_layout_complexity(self, file_path: str) -> str:
        """Simple heuristic for tables vs figures vs single column"""
        with pdfplumber.open(file_path) as pdf:
            table_pages = 0
            figure_pages = 0
            for page in pdf.pages:
                if page.find_tables():
                    table_pages += 1
                if page.images:
                    figure_pages += 1
            total_pages = len(pdf.pages)
            if table_pages / total_pages > 0.3:
                return "table_heavy"
            elif figure_pages / total_pages > 0.3:
                return "figure_heavy"
            else:
                return "single_column"

    def detect_domain_hint(self, file_path: str) -> str:
        """Keyword-based domain classification"""
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or ""
        text_content_lower = text_content.lower()

        for kw in self.financial_keywords:
            if kw in text_content_lower:
                return "financial"
        for kw in self.legal_keywords:
            if kw in text_content_lower:
                return "legal"
        for kw in self.technical_keywords:
            if kw in text_content_lower:
                return "technical"
        for kw in self.medical_keywords:
            if kw in text_content_lower:
                return "medical"
        return "general"

    def estimate_extraction_cost(self, origin_type: str, layout_complexity: str) -> str:
        """Maps origin + layout to extraction tier"""
        if origin_type == "native_digital" and layout_complexity in ["single_column"]:
            return "fast_text_sufficient"
        elif layout_complexity in ["table_heavy", "figure_heavy", "mixed"]:
            return "needs_layout_model"
        else:
            return "needs_vision_model"

    def profile_document(self, file_path: str, doc_id: Optional[str] = None) -> DocumentProfile:
        """
        Runs all heuristics and returns a DocumentProfile instance.
        This method matches what main.py expects.
        """
        if not doc_id:
            doc_id = os.path.basename(file_path).replace(".pdf", "")

        origin_type = self.classify_origin_type(file_path)
        layout_complexity = self.classify_layout_complexity(file_path)
        domain_hint = self.detect_domain_hint(file_path)
        extraction_cost = self.estimate_extraction_cost(origin_type, layout_complexity)

        # Simple placeholder for language detection
        language = "en"
        language_confidence = 0.99

        return DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=language,
            language_confidence=language_confidence,
            domain_hint=domain_hint,
            estimated_extraction_cost=extraction_cost
        )


# Quick test when run standalone
if __name__ == "__main__":
    triage = TriageAgent()
    doc_path = os.path.join("data", "raw", "CBE ANNUAL REPORT 2023-24.pdf")
    profile = triage.profile_document(doc_path)
    print(profile.model_dump_json_pretty())