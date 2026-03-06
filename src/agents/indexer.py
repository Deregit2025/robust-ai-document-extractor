from typing import List, Optional
from src.models.ldu import LDU
from src.models.page_index import SectionNode, PageIndex
from src.utils.llm_utils import LLMUtils
from src.data.vector_store import LocalVectorStore
from src.data.fact_table import FactTable
import json

class PageIndexBuilder:
    def __init__(self, llm_utils: Optional[LLMUtils] = None):
        self.llm = llm_utils or LLMUtils()
        self.vector_store = LocalVectorStore()
        self.fact_table = FactTable()

    def build(self, ldus: List[LDU], doc_id: str) -> PageIndex:
        """
        Build PageIndex tree from LDUs with LLM-enhanced metadata.
        Optimized to avoid calling LLM for every atomic chunk.
        """
        nodes: List[SectionNode] = []
        
        # We only summarize "Headings", "Tables", or "Figures" to save time
        # Standard text chunks just get a generic summary or are skipped for indexing.
        for idx, ldu in enumerate(ldus):
            # 1. Determine importance (Is it a heading or a visual element?)
            is_important = ldu.chunk_type in ("table", "figure") or (ldu.chunk_type == "text" and ldu.parent_section is None)
            
            summary = "Generic content segment."
            entities = []

            if is_important:
                # Summarize only important structural anchors
                summary_prompt = (
                    "Provide a 1-sentence technical summary of the following doc segment. "
                    f"Content: {ldu.content[:1000]}"
                )
                try:
                    summary = self.llm.completions([{"role": "user", "content": summary_prompt}])
                except Exception:
                    summary = "Summary unavailable"

                # Extract entities only for anchors
                entity_prompt = "Extract ORGANIZATION and DATE from this text. Respond in JSON: {'entities': []}"
                try:
                    entity_resp = self.llm.completions(
                        [{"role": "user", "content": f"{entity_prompt}\nText: {ldu.content[:500]}"}],
                        json_mode=True
                    )
                    raw_entities = json.loads(entity_resp).get("entities", [])
                    # Ensure all entities are strings (handles cases where LLM returns dicts)
                    entities = []
                    for e in raw_entities:
                        if isinstance(e, dict):
                            entities.append(f"{e.get('type', 'ENTITY')}: {e.get('value', 'UNKNOWN')}")
                        else:
                            entities.append(str(e))
                    entities = entities[:10]
                except Exception:
                    entities = []

            # 2. Add to Index if it represents a structural node
            if is_important or idx == 0:
                section_title = ldu.parent_section if ldu.parent_section else f"DOCUMENT SEGMENT {idx+1}"
                node = SectionNode(
                    title=section_title,
                    page_start=min(ldu.page_refs),
                    page_end=max(ldu.page_refs),
                    child_sections=[],
                    key_entities=entities,
                    summary=summary,
                    data_types_present=[ldu.chunk_type],
                )
                nodes.append(node)

            # 3. Always add to Vector Store for RAG (this is metadata-only, no LLM call)
            vector = self.llm.get_embeddings(ldu.content)
            self.vector_store.add(
                vector=vector,
                metadata={
                    "doc_id": doc_id,
                    "content_hash": ldu.content_hash,
                    "chunk_type": ldu.chunk_type,
                    "page_refs": ldu.page_refs,
                    "parent_section": ldu.parent_section,
                }
            )

        # Save vector store to disk once after all LDUs are processed
        self.vector_store.save()

        # 5. Persist the full LDU facts for auditability
        self.fact_table.append_ldus(ldus)

        return PageIndex(
            doc_id=doc_id,
            root_sections=nodes
        )
