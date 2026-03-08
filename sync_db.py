import json
import os
import sqlite3

# Define paths
JSONL_PATH = ".refinery/fact_table.jsonl"
DB_PATH = ".refinery/fact_table.db"

def sync():
    if not os.path.exists(JSONL_PATH):
        print(f"[ERROR] {JSONL_PATH} not found.")
        return

    print(f"[SYNC] Populating {DB_PATH} from {JSONL_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            content TEXT,
            chunk_type TEXT,
            page_refs TEXT,
            parent_section TEXT,
            content_hash TEXT UNIQUE,
            extraction_strategy TEXT,
            confidence_score REAL,
            raw_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)")
    
    count = 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                page_refs = data.get("page_refs", [])
                if isinstance(page_refs, list):
                    page_refs_str = ";".join(map(str, page_refs))
                else:
                    page_refs_str = str(page_refs)

                conn.execute("""
                    INSERT OR IGNORE INTO facts 
                    (doc_id, content, chunk_type, page_refs, parent_section, content_hash, extraction_strategy, confidence_score, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("doc_id"),
                    data.get("content"),
                    data.get("chunk_type", "text"),
                    page_refs_str,
                    data.get("parent_section"),
                    data.get("content_hash"),
                    data.get("extraction_strategy", "FastText"),
                    data.get("confidence_score", 1.0),
                    json.dumps(data)
                ))
                count += 1
            except Exception as e:
                print(f"Error at line: {e}")
                
    conn.commit()
    print(f"[SUCCESS] {count} rows processed. Database is ready.")
    conn.close()

if __name__ == "__main__":
    sync()
