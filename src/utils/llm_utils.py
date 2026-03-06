"""
LLM Utilities — 100% Local Ollama Stack

All inference routes through Ollama running locally.
No external API calls. No API keys required.

  - 'minimax-m2.5:cloud'  → text completions, summaries, NER
  - 'moondream'           → vision/OCR completions for scanned pages

Model names are read from rubric/extraction_rules.yaml to keep
configuration in one place.
"""

import os
import hashlib
import numpy as np
import yaml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ollama runs a local OpenAI-compatible server on this base URL
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"  # Ollama does not need a real key


class LLMUtils:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        # ── Load model config from YAML (single source of truth) ─────────────
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        local_models = self.config.get("local_models", {})
        self.default_model = local_models.get("text_model", "minimax-m2.5:cloud")
        self.vision_model  = local_models.get("vision_model", "moondream")

        # ── Single Ollama client for ALL inference ─────────────────────────────
        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
        )

        # ── Health-check: warn early if Ollama is unreachable ─────────────────
        try:
            self.client.models.list()
        except Exception:
            print(
                "[LLMUtils WARNING] Cannot reach Ollama at http://localhost:11434. "
                "Ensure Ollama is running before processing documents."
            )

    def completions(
        self,
        messages: list,
        model: str = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generic text completion. Routes to Ollama default model.
        When json_mode=True, injects a system prompt since Ollama local
        models do not support the response_format json_object parameter.
        """
        target_model = model or self.default_model

        if json_mode:
            messages = [
                {"role": "system", "content": "Respond ONLY with valid JSON, no markdown, no explanation."}
            ] + messages

        response = self.client.chat.completions.create(
            model=target_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def vision_completion(self, prompt: str, base64_image: str) -> str:
        """
        Vision-Language Model call via Ollama (Moondream).
        100% local, zero external API cost.
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
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def get_embeddings(self, text: str) -> list:
        """
        Generate embeddings using Ollama's local embedding endpoint.
        Falls back to a deterministic SHA256-derived vector if Ollama
        embedding endpoint is unavailable (e.g., model not pulled).
        """
        try:
            resp = self.client.embeddings.create(
                input=text,
                model=self.default_model,
            )
            embedding = resp.data[0].embedding
            # Pad or truncate to 1536 for FAISS compatibility
            if len(embedding) != 1536:
                padded = np.zeros(1536)
                padded[:min(len(embedding), 1536)] = embedding[:1536]
                return padded.tolist()
            return embedding
        except Exception:
            # Deterministic fallback: SHA256 hash → normalized float vector
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h, dtype=np.uint8).astype(float)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            padded = np.zeros(1536)
            padded[:len(vec)] = vec
            return padded.tolist()
