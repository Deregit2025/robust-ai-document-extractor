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

@pytest.fixture
def router():
    return ExtractionRouter()

def test_strategy_routing(router):
    for pdf_file in TEST_PDFS:
        path = os.path.join(DATA_DIR, pdf_file)
        result = router.process_document(path)
        # Ensure result contains confidence keys and strategy info
        assert "strategy_used" in result
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 1.0