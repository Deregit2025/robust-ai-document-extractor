"""
Extraction Router

Decides which extraction strategy to use based on DocumentProfile
and routes the document accordingly.
"""

import os
import yaml
from src.models.document_profile import DocumentProfile
from src.models.extracted_document import ExtractedDocument
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor


class ExtractionRouter:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        self.fast_text = FastTextExtractor()
        self.layout = LayoutExtractor()
        self.vision = VisionExtractor()
        
        # Load confidence thresholds from config
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                
        # Base cascade order (FastText -> Layout -> Vision)
        self.strategy_tiers = {
            "fast_text_sufficient": {
                "instance": self.fast_text,
                "threshold": self.config.get("strategy_tiers", {}).get("fast_text_sufficient", {}).get("confidence_threshold", 0.92)
            },
            "needs_layout_model": {
                "instance": self.layout,
                "threshold": self.config.get("strategy_tiers", {}).get("needs_layout_model", {}).get("confidence_threshold", 0.88)
            },
            "needs_vision_model": {
                "instance": self.vision,
                "threshold": self.config.get("strategy_tiers", {}).get("needs_vision_model", {}).get("confidence_threshold", 0.85)
            }
        }
        
        self.cascade_order = [
            "fast_text_sufficient",
            "needs_layout_model",
            "needs_vision_model"
        ]

    def route_extraction(self, doc_path: str, profile: DocumentProfile) -> ExtractedDocument:
        target_tier = profile.estimated_extraction_cost
        doc_id = profile.doc_id
        
        # Find where to start in the cascade based on triage profile
        try:
            start_idx = self.cascade_order.index(target_tier)
        except ValueError:
            start_idx = 0 # Default to FastText if tier unknown
            
        escalation_history = []
        best_doc = None
        
        for idx in range(start_idx, len(self.cascade_order)):
            tier_name = self.cascade_order[idx]
            strategy_info = self.strategy_tiers[tier_name]
            extractor = strategy_info["instance"]
            threshold = strategy_info["threshold"]
            
            try:
                print(f"[ROUTER] Attempting extraction with {tier_name} strategy.")
                extracted_doc = extractor.extract(doc_path, doc_id)
                
                # Check confidence
                confidence = extracted_doc.confidence or 0.0
                if confidence >= threshold:
                    print(f"[ROUTER] Sub-strategy {tier_name} succeeded with confidence {confidence} (threshold {threshold})")
                    extracted_doc.escalation_history = escalation_history
                    return extracted_doc
                else:
                    print(f"[ROUTER] Confidence {confidence} below threshold {threshold} for {tier_name}. Escalating...")
                    escalation_history.append(f"{tier_name} (Low Confidence: {confidence})")
                    best_doc = extracted_doc # Keep it around in case we flat out fail everything else
                    
            except Exception as e:
                print(f"[ROUTER] Strategy {tier_name} crashed for {doc_id}: {e}")
                escalation_history.append(f"{tier_name} (Failed: {str(e)})")
                
                # SPECIAL CASE: if we hit a memory error (like std::bad_alloc on huge PDFs), 
                # try the lightweight FastText as a rescue fallback.
                if "bad_alloc" in str(e).lower() and tier_name != "fast_text_sufficient":
                    print(f"[ROUTER RESCUE] Attempting lightweight FastText rescue for {doc_id} due to memory exhaustion.")
                    try:
                        extracted_doc = self.fast_text.extract(doc_path, doc_id)
                        extracted_doc.escalation_history = escalation_history + ["Memory-Safe Rescue"]
                        extracted_doc.confidence = 0.5 # Lower confidence due to rescue
                        return extracted_doc
                    except Exception:
                        pass
                
        # If we exhausted the cascade without hitting the threshold
        print(f"[ROUTER CRITICAL] All cascading strategies exhausted or failed for {doc_id}.")
        
        if best_doc:
            print("[ROUTER] Returning best available extraction flagged for human review.")
            best_doc.escalation_history = escalation_history
            best_doc.needs_human_review = True
            return best_doc
        else:
            print("[ROUTER] Total failure across the board. Emitting empty failure document.")
            # Total failure document
            return ExtractedDocument(
                doc_id=doc_id,
                text_blocks=[],
                tables=[],
                figures=[],
                escalation_history=escalation_history,
                needs_human_review=True,
                total_pages=1,
                strategy_name="TOTAL_FAILURE",
                confidence=0.0
            )
