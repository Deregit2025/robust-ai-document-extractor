import os
import json
from src.agents.triage import TriageAgent
from src.agents.extraction_router import ExtractionRouter

RAW_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"
REFINERY_PROFILES = ".refinery/profiles"
REFINERY_LEDGER = ".refinery/extraction_ledger.jsonl"

# Create output folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFINERY_PROFILES, exist_ok=True)
os.makedirs(os.path.dirname(REFINERY_LEDGER), exist_ok=True)

triage_agent = TriageAgent()
router = ExtractionRouter()

# Loop over all PDFs in the raw folder
for file_name in os.listdir(RAW_FOLDER):
    if file_name.lower().endswith(".pdf"):
        doc_path = os.path.join(RAW_FOLDER, file_name)
        print(f"\n=== Processing document: {file_name} ===")

        # 1️⃣ Profile the document
        profile = triage_agent.profile_document(doc_path)
        profile_json = profile.model_dump_json(indent=2)
        print("DocumentProfile:", profile_json)

        # Save profile to .refinery/profiles/
        profile_file = os.path.join(REFINERY_PROFILES, f"{profile.doc_id}.json")
        with open(profile_file, "w", encoding="utf-8") as f:
            f.write(profile_json)

        # 2️⃣ Route to extraction strategy
        extracted_doc = router.route_extraction(doc_path, profile)
        print("ExtractedDocument summary:")
        print(f"- Text blocks: {len(extracted_doc.text_blocks)}")
        print(f"- Tables: {len(extracted_doc.tables)}")
        print(f"- Figures: {len(extracted_doc.figures)}")

        # Save extracted document JSON
        output_file = os.path.join(OUTPUT_FOLDER, f"{profile.doc_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_doc.model_dump_json(indent=2))

        # 3️⃣ Append extraction decision to ledger
        ledger_entry = {
            "doc_id": profile.doc_id,
            "origin_type": profile.origin_type,
            "layout_complexity": profile.layout_complexity,
            "domain_hint": profile.domain_hint,
            "estimated_extraction_cost": profile.estimated_extraction_cost,
            "strategy_used": extracted_doc.strategy_name if hasattr(extracted_doc, "strategy_name") else "unknown",
            "confidence_score": extracted_doc.confidence if hasattr(extracted_doc, "confidence") else None
        }
        with open(REFINERY_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")

print("\nAll documents processed successfully!")