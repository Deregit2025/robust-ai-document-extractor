import os
import glob
import json
import logging
import datetime
import gc
from typing import List, Optional
from src.agents.triage import TriageAgent
from src.agents.extraction_router import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.utils.llm_utils import LLMUtils

# Constants
RAW_FOLDER = "data/raw"
OUTPUT_FOLDER = "data/processed"
REFINERY_PROFILES_DIR = ".refinery/profiles"
REFINERY_PAGEINDEX_DIR = ".refinery/pageindex"
REFINERY_LEDGER = ".refinery/extraction_ledger.jsonl"

# Create output folders
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFINERY_PROFILES_DIR, exist_ok=True)
os.makedirs(REFINERY_PAGEINDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REFINERY_LEDGER), exist_ok=True)

# Initialize Agents
llm_utils = LLMUtils()
triage_agent = TriageAgent()
router = ExtractionRouter()
chunker = ChunkingEngine()
indexer = PageIndexBuilder(llm_utils=llm_utils)

def process_documents():
    print("Starting E2E Document Extraction Pipeline...")
    
    # 0. Clean old records for a fresh run
    print("[SYSTEM] Initializing fresh run. Cleaning artifacts...")
    for pattern in [os.path.join(OUTPUT_FOLDER, "*.json"), os.path.join(REFINERY_PROFILES_DIR, "*.json"), os.path.join(REFINERY_PAGEINDEX_DIR, "*.json")]:
        for f in glob.glob(pattern):
            try: os.remove(f)
            except: pass
    if os.path.exists(REFINERY_LEDGER):
        try: os.remove(REFINERY_LEDGER)
        except: pass
    
    # Get all PDFs
    pdf_files = sorted([f for f in os.listdir(RAW_FOLDER) if f.lower().endswith(".pdf")])
    print(f"Total documents identified: {len(pdf_files)}")

    for file_name in pdf_files:
        doc_path = os.path.join(RAW_FOLDER, file_name)
        doc_id = file_name.replace(".pdf", "")
        print(f"\n--- [{datetime.datetime.now().strftime('%H:%M:%S')}] Processing: {file_name} ---")

        ledger_entry = {
            "doc_id": doc_id,
            "status": "PROCESSING_FAILED",
            "timestamp": None
        }

        try:
            # 1. Triage
            print(f"  > Step 1/4: Triaging complexity...")
            profile = triage_agent.profile_document(doc_path)
            
            # Save Profile
            profile_file = os.path.join(REFINERY_PROFILES_DIR, f"{doc_id}.json")
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(profile.model_dump_json(indent=2))

            ledger_entry.update({
                "origin_type": profile.origin_type,
                "layout_complexity": profile.layout_complexity,
                "estimated_extraction_cost": profile.estimated_extraction_cost
            })

            # 2. Extract
            print(f"  > Step 2/4: Extracting content (Strategy: {profile.estimated_extraction_cost})...")
            extracted_doc = router.route_extraction(doc_path, profile)
            
            # 3. Chunk
            print(f"  > Step 3/4: Semantic chunking ({len(extracted_doc.text_blocks)} blocks)...")
            ldus = chunker.chunk(extracted_doc)
            
            # 4. Index
            print(f"  > Step 4/4: Building RAG Index ({len(ldus)} LDUs)...")
            page_index = indexer.build(ldus, doc_id=doc_id)

            # 5. Output
            index_json = page_index.model_dump_json(indent=2)
            with open(os.path.join(OUTPUT_FOLDER, f"{doc_id}_index.json"), "w", encoding="utf-8") as f:
                f.write(index_json)
            with open(os.path.join(REFINERY_PAGEINDEX_DIR, f"{doc_id}.json"), "w", encoding="utf-8") as f:
                f.write(index_json)
            with open(os.path.join(OUTPUT_FOLDER, f"{doc_id}_extracted.json"), "w", encoding="utf-8") as f:
                f.write(extracted_doc.model_dump_json(indent=2))

            ledger_entry.update({
                "status": "SUCCESS",
                "strategy_used": extracted_doc.strategy_name,
                "chunks": len(ldus),
                "timestamp": datetime.datetime.now().isoformat()
            })
            print(f"  [COMPLETED] {file_name} processed successfully.")

        except Exception as e:
            print(f"  [CRITICAL ERROR] {file_name}: {str(e)}")
            ledger_entry["error"] = str(e)

        # Update Ledger
        with open(REFINERY_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        
        # Immediate memory cleanup 
        gc.collect()

    print("\n[PIPELINE COMPLETE] All files processed.")

if __name__ == "__main__":
    process_documents()
