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
from typing import Optional, Tuple
import pdfplumber

from src.models.document_profile import DocumentProfile
from src.agents.classifiers import KeywordDomainClassifier


class TriageAgent:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        # Load professional rules
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.triage_config = self.config.get("triage", {})
        self.origin_config = self.triage_config.get("origin", {})
        self.layout_config = self.triage_config.get("layout", {})
        
        domain_keywords = self.triage_config.get("domains", {
            "financial": ["revenue", "income", "balance sheet", "profit"],
            "legal": ["contract", "agreement", "plaintiff", "hereby"],
            "technical": ["algorithm", "performance", "implementation"],
            "medical": ["patient", "diagnosis", "treatment"]
        })
        self.domain_classifier = KeywordDomainClassifier(domain_keywords)

    def classify_origin_type(self, file_path: str) -> Tuple[str, float]:
        """
        Uses multiple signals: image area, interactive forms, text density, and font metadata
        to classify origin as: scanned_image (zero-text), native_digital, mixed, or form_fillable.
        """
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:10]
            if not pages:
                return "scanned_image", 1.0 # Empty pdf -> zero-text fallback
                
            total_chars = 0
            scanned_pages = 0
            form_fields = 0
            embedded_fonts = set()
            
            # Check for interactive AcroForm widgets/fields
            if pdf.doc.catalog.get("AcroForm"):
                form_fields = 1 # Indicator present
                
            for page in pages:
                # 1. Text Density & Fonts
                text = page.extract_text() or ""
                char_count = len(text.strip())
                total_chars += char_count
                
                # Check for fonts - if a page has text but no fonts, it might be rendered paths
                if page.chars:
                    embedded_fonts.update([char.get("fontname") for char in page.chars if char.get("fontname")])
                
                # 2. Image Area vs Page Area
                page_area = page.width * page.height
                image_area = sum((img["width"] * img["height"] for img in page.images), 0)
                image_coverage_ratio = image_area / page_area if page_area > 0 else 0
                
                # Threshold from config or default to 0.8
                img_coverage_threshold = self.origin_config.get("image_area_coverage_threshold", 0.8)
                char_limit = self.origin_config.get("zero_text_char_limit", 50)
                
                if char_count < char_limit and image_coverage_ratio > img_coverage_threshold:
                    scanned_pages += 1
                elif char_count == 0 and not page.images:
                    # Blank page counts towards scanned/needs vision (could be corrupt)
                    scanned_pages += 0.5 
                    
            scanned_ratio = scanned_pages / len(pages)
            scanned_threshold = self.origin_config.get("scanned_image_ratio_threshold", 0.5)
            
            # Explicit branching logic
            if form_fields > 0:
                return "form_fillable", 0.95
            
            # Zero-text explicit check across sampled pages
            if total_chars < self.origin_config.get("zero_text_char_limit", 50) and len(embedded_fonts) == 0:
                return "scanned_image", 0.99 # explicitly zero-text
                
            if scanned_ratio >= scanned_threshold:
                # E.g if ratio is 0.7, confidence is higher than if it's 0.5
                confidence = min(0.5 + (scanned_ratio - scanned_threshold), 0.95)
                return "scanned_image", round(confidence, 2)
            
            if 0 < scanned_ratio < scanned_threshold:
                # It's mixed! Has valid digital text on some pages, but full scanned images on others.
                confidence = 0.6 + (scanned_ratio * 0.5)
                return "mixed", round(confidence, 2)
                
            # It's native digital
            confidence = 1.0 if len(embedded_fonts) > 0 else 0.8 # Higher confidence if we explicitly found fonts
            return "native_digital", confidence

    def classify_layout_complexity(self, file_path: str) -> Tuple[str, float]:
        """Multi-signal heuristic for complex layouts."""
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:10]
            if not pages:
                return "single_column", 1.0
                
            table_pages = 0
            figure_pages = 0
            for page in pages:
                if page.find_tables():
                    table_pages += 1
                if page.images:
                    figure_pages += 1
                    
            total_pages = len(pages)
            table_ratio = table_pages / total_pages
            figure_ratio = figure_pages / total_pages
            
            table_threshold = self.layout_config.get("table_heavy_threshold", 0.3)
            figure_threshold = self.layout_config.get("figure_heavy_threshold", 0.3)
            
            if table_ratio > table_threshold and figure_ratio > figure_threshold:
                return "mixed", round(min(table_ratio + figure_ratio, 0.95), 2)
            elif table_ratio > table_threshold:
                return "table_heavy", round(min(0.5 + table_ratio, 0.95), 2)
            elif figure_ratio > figure_threshold:
                return "figure_heavy", round(min(0.5 + figure_ratio, 0.95), 2)
            else:
                return "single_column", 0.85

    def detect_domain_hint(self, file_path: str) -> Tuple[str, float]:
        """Leverages the pluggable classifier for domain hints."""
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:5]: # First 5 pages are usually enough for domain
                text_content += page.extract_text() or ""
                
        return self.domain_classifier.classify_domain(text_content)

    def estimate_extraction_cost(self, origin_type: str, layout_complexity: str) -> str:
        """
        Maps origin + layout to extraction tier based on professional rules.
        Uses explicit origin checks before layout rules to avoid accidental
        membership matches on shared string values (BUG-06 / BUG-07 fix).
        """
        # Tier C: Any scanned or form document goes directly to Vision
        if origin_type == "scanned_image":
            return "needs_vision_model"

        # Tier C: Mixed documents have scanned pages — Vision can handle both
        if origin_type == "mixed":
            return "needs_layout_model"  # Layout first; Vision is fallback via escalation

        # Tier C: Complex layout types from the config always need Vision
        vision_layout_rules = self.config["strategy_tiers"]["needs_vision_model"]["applicable_layouts"]
        if layout_complexity in vision_layout_rules:
            return "needs_vision_model"

        # Tier B: Layout model for structured digital layouts
        layout_rules = self.config["strategy_tiers"]["needs_layout_model"]["applicable_layouts"]
        if layout_complexity in layout_rules:
            return "needs_layout_model"

        # Default Tier A: Fast Text for simple native digital docs
        return "fast_text_sufficient"

    def profile_document(self, file_path: str, doc_id: Optional[str] = None) -> DocumentProfile:
        """
        Runs all heuristics and returns a DocumentProfile instance.
        """
        if not doc_id:
            doc_id = os.path.basename(file_path).replace(".pdf", "")

        origin_type, origin_conf = self.classify_origin_type(file_path)
        layout_complexity, layout_conf = self.classify_layout_complexity(file_path)
        domain_hint, domain_conf = self.detect_domain_hint(file_path)
        
        # New cost estimation based on rubric
        extraction_cost = self.estimate_extraction_cost(origin_type, layout_complexity)

        return DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            origin_confidence=origin_conf,
            layout_complexity=layout_complexity,
            layout_confidence=layout_conf,
            language="en",
            language_confidence=0.99,
            domain_hint=domain_hint,
            domain_confidence=domain_conf,
            estimated_extraction_cost=extraction_cost
        )


