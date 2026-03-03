

# Extractor AI — Document Processing Pipeline

## **Project Overview**

`Extractor AI` is an end-to-end document extraction framework that classifies PDFs, determines the optimal extraction strategy, and outputs structured data. The pipeline supports multiple extraction tiers:

1. **FastTextExtractor** – Lightweight text extraction for native PDFs.
2. **LayoutExtractor** – Table and column detection for structured PDFs.
3. **VisionExtractor** – OCR-based extraction for scanned or figure-heavy PDFs.

The system implements a **confidence-gated escalation**, ensuring that computationally heavy strategies are only used when simpler methods are insufficient.

---

## **Features**

* Document triage: origin type, layout complexity, domain hints, and extraction cost tier.
* Unified output: `ExtractedDocument` Pydantic schema with text, tables, and figures.
* Modular extraction strategies for flexibility and future expansion.
* JSON-based document profiles and extraction ledger for reproducibility and auditing.
* Full pipeline test coverage for triage and extraction confidence scoring.

---

## **Folder Structure**

```
extractor_ai/
│
├── src/
│   ├── agents/
│   │   ├── triage.py
│   │   ├── extraction_router.py
│   │   ├── indexer.py
│   │   ├── chunker.py
│   │   └── query_agent.py
│   │
│   ├── strategies/
│   │   ├── base.py
│   │   ├── fast_text.py
│   │   ├── layout.py
│   │   └── vision.py
│   │
│   └── models/
│       ├── document_profile.py
│       ├── extracted_document.py
│       ├── ldu.py
│       ├── page_index.py
│       └── provenance_chain.py
│
├── data/
│   └── raw/               # Input PDF files
│
├── .refinery/
│   ├── profiles/          # JSON outputs from TriageAgent
│   └── extraction_ledger.jsonl
│
├── rubric/
│   └── extraction_rules.yaml
│
├── tests/                 # Unit tests
├── main.py
├── pyproject.toml
└── README.md
```

---

## **Setup Instructions**

1. **Clone the repository**

```bash
git clone <repository_url>
cd extractor_ai
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Required packages include `pydantic`, `pdfplumber`, and `docling`.

---

## **Running the Pipeline**

1. Place your PDF documents in `data/raw/`.
2. Run `main.py` to generate document profiles and extract structured data:

```bash
python main.py
```

3. Outputs:

   * Document profiles: `.refinery/profiles/{doc_id}.json`
   * Extraction ledger: `.refinery/extraction_ledger.jsonl`

---

## **Testing**

Unit tests cover:

* TriageAgent classification logic (origin type, layout complexity, domain hint).
* Extraction confidence scoring and strategy selection.

Run tests with:

```bash
pytest tests/
```

---

## **Project Notes**

* **Pipeline:** Confidence-gated, 3-tier extraction (FastText → Layout → Vision).
* **Domain Onboarding:** Observations from Phase 0 include PDF character density, table frequency, and OCR quality.
* **Cost Analysis:** Heavy strategies are selectively applied to minimize computational overhead.

---

## **References**

* [Pydantic V2 Documentation](https://docs.pydantic.dev/)
* [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
* [Docling Documentation](https://pypi.org/project/docling/)

---

## **Future Work**

* Integrate VLM-based VisionExtractor for advanced figure-heavy PDFs.
* Expand domain-specific keyword classifiers.
* Implement multi-threaded or GPU-accelerated processing for large corpora.


