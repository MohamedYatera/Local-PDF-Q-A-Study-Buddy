from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Settings


class OllamaServiceError(RuntimeError):
    pass


class OllamaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(base_url=settings.ollama_base_url, timeout=settings.ollama_generation_timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def health_check(self) -> bool:
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self.client.post(
                "/api/embed",
                json={
                    "model": self.settings.embedding_model,
                    "input": texts,
                },
                timeout=self.settings.ollama_embed_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaServiceError(
                "Embedding request to Ollama failed. Make sure Ollama is running and the embedding model is installed."
            ) from exc

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaServiceError("Embedding response did not contain embeddings.")
        return embeddings

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self.client.post(
                "/api/chat",
                json={
                    "model": self.settings.generation_model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.settings.ollama_generation_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaServiceError(
                "Generation request to Ollama timed out or failed. Try a smaller model, reduce top-k, or increase OLLAMA_GENERATION_TIMEOUT_SECONDS."
            ) from exc

        payload = response.json()
        content = payload.get("message", {}).get("content", "").strip()
        if not content:
            raise OllamaServiceError("Generation response was empty.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise OllamaServiceError(f"Invalid JSON from model: {content}")
            parsed = json.loads(content[start : end + 1])

        return _normalize_model_json(parsed)


def _normalize_model_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and (
        {"answer", "enough_evidence"} <= payload.keys() or "supported" in payload or "unsupported_claims" in payload
    ):
        return payload

    if isinstance(payload, dict):
        for key in ("strictJSON", "json", "response", "data", "output", "result"):
            if key not in payload:
                continue
            normalized = _coerce_nested_json(payload[key])
            if normalized is not None:
                return normalized

        if len(payload) == 1:
            normalized = _coerce_nested_json(next(iter(payload.values())))
            if normalized is not None:
                return normalized

    raise OllamaServiceError(f"Model returned JSON in an unsupported shape: {payload}")


def _coerce_nested_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        try:
            return _normalize_model_json(value)
        except OllamaServiceError:
            return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return _normalize_model_json(json.loads(stripped))
        except (json.JSONDecodeError, OllamaServiceError):
            return None

    return None
