"""
Triage Agent

Responsible for:
- Detecting document origin type (digital vs scanned)
- Classifying layout complexity
- Producing domain hints
- Estimating extraction cost tier
"""

import os
import yaml
from typing import Optional
import pdfplumber

from src.models.document_profile import DocumentProfile


class TriageAgent:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        # Load professional rules
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Keyword lists for domain detection (can be expanded in YAML later)
        self.financial_keywords = ["revenue", "income", "balance sheet", "profit"]
        self.legal_keywords = ["contract", "agreement", "plaintiff", "hereby"]
        self.technical_keywords = ["algorithm", "performance", "implementation"]
        self.medical_keywords = ["patient", "diagnosis", "treatment"]

    def classify_origin_type(self, file_path: str) -> str:
        """Heuristic: mostly empty or image pages → scanned (Sampling first 10 pages)"""
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:10]
            total_scanned = 0
            for page in pages:
                char_count = len(page.extract_text() or "")
                if char_count < 50:
                    total_scanned += 1
            return "scanned_image" if total_scanned / len(pages) > 0.5 else "native_digital"

    def classify_layout_complexity(self, file_path: str) -> str:
        """Simple heuristic for tables vs figures vs single column (Sampling first 10 pages)"""
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:10]
            table_pages = 0
            figure_pages = 0
            for page in pages:
                if page.find_tables():
                    table_pages += 1
                if page.images:
                    figure_pages += 1
            total_pages = len(pages)
            if table_pages / total_pages > 0.3:
                return "table_heavy"
            elif figure_pages / total_pages > 0.3:
                return "figure_heavy"
            else:
                return "single_column"

    def detect_domain_hint(self, file_path: str) -> str:
        """Keyword-based domain classification (Sampling first 10 pages)"""
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:10]:
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
        """
        Maps origin + layout to extraction tier based on professional rules.
        """
        # Follow Tier C: Vision Model for scanned images
        vision_rules = self.config["strategy_tiers"]["needs_vision_model"]["applicable_layouts"]
        if origin_type in vision_rules or layout_complexity in vision_rules:
            return "needs_vision_model"
        
        # Follow Tier B: Layout Model for complex digital layouts
        layout_rules = self.config["strategy_tiers"]["needs_layout_model"]["applicable_layouts"]
        if layout_complexity in layout_rules:
            return "needs_layout_model"
            
        # Default to Tier A: Fast Text
        return "fast_text_sufficient"

    def profile_document(self, file_path: str, doc_id: Optional[str] = None) -> DocumentProfile:
        """
        Runs all heuristics and returns a DocumentProfile instance.
        """
        if not doc_id:
            doc_id = os.path.basename(file_path).replace(".pdf", "")

        origin_type = self.classify_origin_type(file_path)
        layout_complexity = self.classify_layout_complexity(file_path)
        domain_hint = self.detect_domain_hint(file_path)
        
        # New cost estimation based on rubric
        extraction_cost = self.estimate_extraction_cost(origin_type, layout_complexity)

        return DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language="en",
            language_confidence=0.99,
            domain_hint=domain_hint,
            estimated_extraction_cost=extraction_cost
        )


