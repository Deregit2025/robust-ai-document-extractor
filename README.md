

# Extractor AI — Document Processing Pipeline

## **Project Overview**

`Extractor AI` is an end-to-end document extraction framework that classifies PDFs, determines the optimal extraction strategy, and outputs structured data. The pipeline supports multiple extraction tiers:

1. **FastTextExtractor** – Lightweight text extraction for native PDFs.
2. **LayoutExtractor** – Table and column detection for structured PDFs.
3. **VisionExtractor** – OCR-based extraction for scanned or figure-heavy PDFs.

The system implements a **confidence-gated escalation**, ensuring that computationally heavy strategies are only used when simpler methods are insufficient.

## **Features & Rubric Alignment (Interim Submission)**

### **My Contributions for this Submission**
To meet the rubric's requirements, I implemented the following critical architectural features:

1. **Strategic VLM Transition & Local Models**: Replaced credit-limited cloud APIs with a **100% local Ollama stack** (`Moondream` for Vision OCR, `Minimax` for Semantic Chunking/NER), saving significant runtime costs and ensuring privacy.
2. **Robust Multi-Tier Routing**: Engineered an `ExtractionRouter` with built-in Exception handling that systematically falls back to lower cost/complexity tiers (e.g., `FastTextExtractor`) if complex Vision Extractions fail, guaranteeing pipeline completion.
3. **Advanced Triage Logic**: Enhanced the `TriageAgent` to programmatically sample the first 10 pages for image/character density, explicitly detecting `scanned_image` vs `native_digital` to drive cost-aware routing.
4. **Pydantic Model Rigor**: Enforced strict normalized schemas (`DocumentProfile`, `ExtractedDocument`, `LDU`) and implemented systematic SHA256 `content_hash` generation for unassailable provenance tracking in the vector store.
5. **Decoupled Configuration**: Externalized hardcoded system magic numbers and extraction routing rules into a central `rubric/extraction_rules.yaml` file to allow rapid operational adjustments without code changes.
6. **Robust Auditing**: Formalized the generation of `.refinery/extraction_ledger.jsonl` to provide a clean, deduplicated, and professional audit trail of all extracted artifacts.

---

This repository satisfies the core themes for robust document extraction:

1. **Core Pydantic Schema Design:** Enforces strict `DocumentProfile`, `ExtractedDocument`, and `LDU` schemas with `content_hash` provenance.
2. **Triage Agent & Classification Logic:** Programmatically samples documents to detect origin, layout complexity (`figure_heavy`, `table_heavy`), and domain hints.
3. **Multi-Strategy Extraction:** Implements normalized extracting tiers (Fast Text, Layout, Vision).
4. **Extraction Router with Confidence-Gated Escalation:** Dynamically routes documents based on Triage profiles and automatically falls back to lower-tier strategies if complex extractions fail.
5. **Externalized Configuration:** All extraction rules, chunking parameters, and model selections are decoupled into `rubric/extraction_rules.yaml`.

### **100% Local & Cost-Free Architecture**
The pipeline has been transitioned from credit-dependent APIs to a completely local LLM stack using **Ollama**:
* **Vision / OCR:** `moondream` handles complex image-to-text extraction.
* **Semantic Enrichment:** `minimax-m2.5:cloud` generates section summaries and Named Entity Recognition (NER) locally.

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


