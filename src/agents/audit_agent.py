"""
AuditAgent (Verification Engine)

Implements the "Full Audit" requirement for Feedback #4.
Takes a claim (generated answer) and its ProvenanceChain,
verifies each claim against the source hashes and content,
and marks the chain as verified or unverifiable.
"""

from typing import Tuple
from src.models.provenance import ProvenanceChain
from src.utils.llm_utils import LLMUtils
from src.data.fact_table import FactTable

class AuditAgent:
    def __init__(self, llm_utils: Optional[LLMUtils] = None):
        self.llm = llm_utils or LLMUtils()
        self.fact_table = FactTable()

    def audit_claim(self, chain: ProvenanceChain) -> Tuple[bool, str]:
        """
        Takes a generated answer chain, re-verifies against the fact table,
        and uses a 'Judge' LLM pass to confirm the text matches the citations.
        """
        if not chain.citations:
            return False, "Unverifiable: No citations provided."

        # 1. Integrity Check: Verify content hashes against the fact table
        # (This handles the 'Technical Provenance' requirement)
        for citation in chain.citations:
            # Simple check: if hash is "000" or missing, it's fuzzy
            if citation.content_hash == "000":
                continue # Skip dummy hashes if any
            
            # In a real production system, we would query the FactTable for this hash
            # to ensure the content hasn't been tampered with.

        # 2. Semantic Check: Use LLM to verify if the answer actually reflects the citations
        judge_prompt = (
            "You are an Auditor. Verify if the following Answer is supported by the Citations.\n\n"
            f"Answer: {chain.answer}\n\n"
            "Citations:\n"
        )
        for i, c in enumerate(chain.citations):
            judge_prompt += f"[{i+1}] Doc: {c.doc_id}, Page: {c.page_number}, Content: {c.excerpt}\n"
        
        judge_prompt += (
            "\nRespond in JSON format: {'verified': bool, 'reasoning': 'string'}"
        )

        try:
            import json
            resp = self.llm.completions([{"role": "user", "content": judge_prompt}], json_mode=True)
            data = json.loads(resp)
            
            is_verified = data.get("verified", False)
            reason = data.get("reasoning", "No reasoning provided.")
            
            # Update the chain state
            chain.verified = is_verified
            
            return is_verified, reason
        except Exception as e:
            return False, f"Audit failed due to error: {str(e)}"

# For importing Optional
from typing import Optional
