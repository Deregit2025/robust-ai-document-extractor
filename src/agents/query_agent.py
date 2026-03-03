"""
PageIndex Builder

- Builds hierarchical section tree from LDUs
- Generates LLM summaries (stubbed)
- Extracts key entities per section (stubbed)
"""

from typing import List, Optional
from src.models.ldu import LDU
from src.models.page_index import PageIndexNode, PageIndex
import hashlib

# LLM & NER stubs
def llm_summary(text: str) -> str:
    """
    Generate a short 2-3 sentence summary of the text.
    Replace with actual LLM call.
    """
    return f"Summary of: {text[:50]}..."  # stub

def extract_key_entities(text: str) -> List[str]:
    """
    Extract key entities from text.
    Replace with spaCy or any NER model.
    """
    return ["Entity1", "Entity2"]  # stub


class PageIndexBuilder:
    def __init__(self):
        pass

    def build(self, ldus: List[LDU]) -> PageIndex:
        """
        Build PageIndex tree from LDUs
        """
        # For simplicity, treat each LDU as a "section" if it has page_refs
        nodes: List[PageIndexNode] = []
        for idx, ldu in enumerate(ldus):
            content_hash = ldu.content_hash
            node = PageIndexNode(
                title=f"{ldu.chunk_type.capitalize()} Section {idx+1}",
                page_start=min(ldu.page_refs),
                page_end=max(ldu.page_refs),
                child_sections=[],
                key_entities=extract_key_entities(ldu.content),
                summary=llm_summary(ldu.content),
                data_types_present=[ldu.chunk_type],
                ldu_refs=[content_hash],  # link to LDU via content_hash
            )
            nodes.append(node)

        # For now, we create a flat PageIndex; later can build hierarchy using section headers
        page_index = PageIndex(
            document_title="Demo Document",
            root_sections=nodes
        )
        return page_index


# Quick test
if __name__ == "__main__":
    from src.agents.chunker import ChunkingEngine
    from src.agents.triage import TriageAgent
    from src.agents.extraction_router import ExtractionRouter

    doc_path = "data/raw/CBE ANNUAL REPORT 2023-24.pdf"

    triage = TriageAgent()
    profile = triage.profile_document(doc_path)

    router = ExtractionRouter()
    extracted = router.route(profile, doc_path)

    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)

    indexer = PageIndexBuilder()
    page_index = indexer.build(ldus)

    print(f"PageIndex root sections: {len(page_index.root_sections)}")
    for sec in page_index.root_sections[:3]:
        print(sec.title, sec.page_start, sec.page_end, sec.summary)