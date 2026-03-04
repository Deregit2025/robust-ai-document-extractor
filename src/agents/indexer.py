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
        """
        nodes: List[SectionNode] = []
        
        for idx, ldu in enumerate(ldus):
            # 1. Professional Summary Prompt
            summary_prompt = (
                "You are an expert document analyst. Provide a concise, 2-3 sentence technical summary "
                "of the following document chunk. Focus on the core message and key data points.\n\n"
                f"Content: {ldu.content}"
            )
            summary = self.llm.completions([{"role": "user", "content": summary_prompt}])
            
            # 2. Robust Entity Extraction (Schema-driven)
            entity_prompt = (
                "Identify and extract all key entities from the text segment below. "
                "Classify them into: ORGANIZATION, PERSON, DATE, MONETARY_VALUE, or POLICY_NUM. "
                "Return only a valid JSON object with the key 'entities' containing a list of strings."
            )
            try:
                entity_resp = self.llm.completions(
                    [{"role": "system", "content": "You are a legal/financial extraction agent. Output strictly JSON."},
                     {"role": "user", "content": f"{entity_prompt}\nText: {ldu.content}"}],
                    json_mode=True
                )
                raw_json = json.loads(entity_resp)
                entities = raw_json.get("entities", [])
                if not entities and isinstance(raw_json, list):
                    entities = raw_json
                # Cleanup: ensure it's a list of strings
                entities = [str(e) for e in entities][:10] # Cap for efficiency
            except Exception:
                entities = ["NER_UNAVAILABLE"]

            # 3. Build Section Node
            node = SectionNode(
                title=f"{ldu.chunk_type.upper()} SECTION {idx+1}",
                page_start=min(ldu.page_refs),
                page_end=max(ldu.page_refs),
                child_sections=[],
                key_entities=entities,
                summary=summary,
                data_types_present=[ldu.chunk_type],
            )
            nodes.append(node)
            
            # 4. Storage Integration
            vector = self.llm.get_embeddings(ldu.content)
            self.vector_store.add(
                vector=vector,
                metadata={
                    "doc_id": doc_id,
                    "content_hash": ldu.content_hash,
                    "chunk_type": ldu.chunk_type,
                    "summary": summary,
                    "entities": entities
                }
            )

        # 5. Persist the full LDU facts for auditability
        self.fact_table.append_ldus(ldus)

        return PageIndex(
            doc_id=doc_id,
            root_sections=nodes
        )
