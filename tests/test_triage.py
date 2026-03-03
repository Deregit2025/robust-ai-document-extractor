# tests/test_triage.py
import os
import pytest
from src.agents.triage import TriageAgent

# Test PDFs in data/raw
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TEST_PDFS = [
    "Audit Report - 2023.pdf",
    "CBE ANNUAL REPORT 2023-24.pdf",
    "fta_performance_survey_final_report_2022.pdf",
    "tax_expenditure_ethiopia_2021_22.pdf"
]

@pytest.fixture
def triage_agent():
    return TriageAgent()

def test_classify_origin_type(triage_agent):
    for pdf_file in TEST_PDFS:
        path = os.path.join(DATA_DIR, pdf_file)
        origin = triage_agent.classify_origin_type(path)
        assert origin in ["native_digital", "scanned_image"]

def test_classify_layout_complexity(triage_agent):
    for pdf_file in TEST_PDFS:
        path = os.path.join(DATA_DIR, pdf_file)
        layout = triage_agent.classify_layout_complexity(path)
        assert layout in ["single_column", "table_heavy", "figure_heavy"]

def test_detect_domain_hint(triage_agent):
    for pdf_file in TEST_PDFS:
        path = os.path.join(DATA_DIR, pdf_file)
        domain = triage_agent.detect_domain_hint(path)
        assert domain in ["financial", "legal", "technical", "medical", "general"]

def test_estimate_extraction_cost(triage_agent):
    # Basic combinations
    assert triage_agent.estimate_extraction_cost("native_digital", "single_column") == "fast_text_sufficient"
    assert triage_agent.estimate_extraction_cost("native_digital", "table_heavy") == "needs_layout_model"
    assert triage_agent.estimate_extraction_cost("scanned_image", "figure_heavy") == "needs_vision_model"