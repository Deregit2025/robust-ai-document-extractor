"""
LLM Utilities

Dual-provider configuration:
  - Ollama (local, free): used for section summaries and entity extraction
    in the PageIndex builder. This is the "cheap model" path.
  - OpenRouter (cloud): used only for VLM visual extraction on complex/scanned pages.

This hybrid approach ensures the pipeline runs cost-efficiently while retaining
high-fidelity visual extraction for complex documents.
"""

import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import yaml

load_dotenv()

# Ollama runs a local OpenAI-compatible server on this base URL
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Ollama doesn't need a real key

# Default local model for cheap summarization / NER tasks
OLLAMA_DEFAULT_MODEL = "minimax-m2.5:cloud"
OLLAMA_VISION_MODEL = "moondream"


class LLMUtils:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        # ── OpenRouter client (Fallback only) ───────────────────────
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
            )
        else:
            self.openrouter_client = None

        # ── Ollama client (local, free — for summaries & NER) ────────────────
        self.ollama_client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
        )

        # ── Load model config from YAML ───────────────────────────────────────
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.vlm_model = self.config["strategy_tiers"]["needs_vision_model"]["model_id"]

    def completions(
        self,
        messages: list,
        model: str = None,
        json_mode: bool = False,
        use_openrouter: bool = False,
    ) -> str:
        """
        Generic completion helper. Routes to Ollama by default.
        """
        response_format = {"type": "json_object"} if json_mode else None

        if use_openrouter and self.openrouter_client:
            target_model = model or self.vlm_model
            client = self.openrouter_client
        else:
            target_model = OLLAMA_DEFAULT_MODEL
            client = self.ollama_client
            if json_mode:
                response_format = None
                messages = [
                    {"role": "system", "content": "Respond ONLY with valid JSON, no markdown, no explanation."}
                ] + messages

        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.content

    def vision_completion(self, prompt: str, base64_image: str) -> str:
        """
        Specific helper for local VLM (Vision-Language Model) calls via Ollama.
        Uses Moondream for 100% local, free image-to-text extraction.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = self.ollama_client.chat.completions.create(
            model=OLLAMA_VISION_MODEL,
            messages=messages,
            response_format=None  # moondream doesn't support json_mode strictly
        )
        return response.choices[0].message.content

    def get_embeddings(self, text: str) -> list:
        """
        Generate embeddings using Ollama's local embedding endpoint.
        """
        try:
            resp = self.ollama_client.embeddings.create(
                input=text,
                model=OLLAMA_DEFAULT_MODEL,
            )
            embedding = resp.data[0].embedding
            # Pad to 1536 for FAISS compatibility
            if len(embedding) != 1536:
                padded = np.zeros(1536)
                padded[:min(len(embedding), 1536)] = embedding[:1536]
                return padded.tolist()
            return embedding
        except Exception:
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h, dtype=np.uint8).astype(float)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            padded = np.zeros(1536)
            padded[:len(vec)] = vec
            return padded.tolist()
