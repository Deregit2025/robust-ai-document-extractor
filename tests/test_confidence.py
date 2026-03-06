# tests/test_confidence.py
import os
import pytest
from src.agents.extraction_router import ExtractionRouter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TEST_PDFS = [
    "Audit Report - 2023.pdf",
    "CBE ANNUAL REPORT 2023-24.pdf",
    "fta_performance_survey_final_report_2022.pdf",
    "tax_expenditure_ethiopia_2021_22.pdf"
]

from src.agents.triage import TriageAgent
from src.agents.extraction_router import ExtractionRouter

@pytest.fixture
def triage():
    return TriageAgent()

@pytest.fixture
def router():
    return ExtractionRouter()

def test_strategy_routing(router, triage):
    for pdf_file in TEST_PDFS:
        path = os.path.join(DATA_DIR, pdf_file)
        profile = triage.profile_document(path)
        result = router.route_extraction(path, profile)
        
        # Ensure result contains confidence and strategy info
        assert getattr(result, "strategy_name", None) is not None
        assert getattr(result, "confidence", None) is not None
        assert 0.0 <= result.confidence <= 1.0