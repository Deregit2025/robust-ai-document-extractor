import os
import json
from src.agents.triage import TriageAgent
from src.agents.extraction_router import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.utils.llm_utils import LLMUtils

# Constants
RAW_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"
REFINERY_PROFILES_DIR = ".refinery/profiles"
REFINERY_PROFILES_LEDGER = ".refinery/profiles/profiles_ledger.jsonl"
REFINERY_LEDGER = ".refinery/extraction_ledger.jsonl"

# Create output folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFINERY_PROFILES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REFINERY_LEDGER), exist_ok=True)

# Initialize Agents
llm_utils = LLMUtils()
triage_agent = TriageAgent()
router = ExtractionRouter()
chunker = ChunkingEngine()
indexer = PageIndexBuilder(llm_utils=llm_utils)

def process_documents():
    print("Starting E2E Document Extraction Pipeline...")
    
    # 0. Deep clean to ensure no stale artifacts
    for folder in [OUTPUT_FOLDER, REFINERY_PROFILES_DIR]:
        for f in os.listdir(folder):
            if f.endswith(".json"):
                os.remove(os.path.join(folder, f))
    if os.path.exists(REFINERY_LEDGER):
        os.remove(REFINERY_LEDGER)

    # Loop over all PDFs in the raw folder
    for file_name in os.listdir(RAW_FOLDER):
        if file_name.lower().endswith(".pdf"):
            doc_path = os.path.join(RAW_FOLDER, file_name)
            doc_id = file_name.replace(".pdf", "")
            print(f"\n=== Processing: {file_name} ===")

            # Initialize ledger entry with failure state as default
            ledger_entry = {
                "doc_id": doc_id,
                "status": "PROCESSING_INTERRUPTED",
                "timestamp": None
            }

            try:
                # 1. Triage/Profile
                profile = triage_agent.profile_document(doc_path)
                
                # Save profile to individual file immediately
                profile_file = os.path.join(REFINERY_PROFILES_DIR, f"{doc_id}.json")
                with open(profile_file, "w", encoding="utf-8") as f:
                    f.write(profile.model_dump_json(indent=2))

                # Update ledger with triage info
                ledger_entry.update({
                    "origin_type": profile.origin_type,
                    "layout_complexity": profile.layout_complexity,
                    "estimated_extraction_cost": profile.estimated_extraction_cost,
                    "domain_hint": profile.domain_hint
                })

                # 2. Extraction (with fallback logic)
                extracted_doc = router.route_extraction(doc_path, profile)
                print(f"Extraction strategy: {extracted_doc.strategy_name}")

                # 3. Chunking
                ldus = chunker.chunk(extracted_doc)
                print(f"Generated {len(ldus)} chunks (LDUs)")

                # 4. Indexing & Storage (Vector + Fact Table)
                page_index = indexer.build(ldus, doc_id=doc_id)
                print(f"Built PageIndex with {len(page_index.root_sections)} sections")

                # 5. Save PageIndex and ExtractedDoc
                idx_output = os.path.join(OUTPUT_FOLDER, f"{doc_id}_index.json")
                with open(idx_output, "w", encoding="utf-8") as f:
                    f.write(page_index.model_dump_json(indent=2))
                
                doc_output = os.path.join(OUTPUT_FOLDER, f"{doc_id}_extracted.json")
                with open(doc_output, "w", encoding="utf-8") as f:
                    f.write(extracted_doc.model_dump_json(indent=2))

                # Final Success Update
                import datetime
                ledger_entry.update({
                    "strategy_used": extracted_doc.strategy_name,
                    "confidence_score": extracted_doc.confidence,
                    "chunks_count": len(ldus),
                    "status": "SUCCESS",
                    "timestamp": datetime.datetime.now().isoformat()
                })

            except Exception as e:
                print(f"FAILED to process {file_name}: {str(e)}")
                ledger_entry.update({
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": None
                })

            # Write to ledger (one file, one line per doc, final state)
            with open(REFINERY_LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps(ledger_entry) + "\n")

    print("\nAll documents processed successfully!")

if __name__ == "__main__":
    process_documents()
