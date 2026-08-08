"""Filesystem cache decorator for any language model implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model.ports.inference import LanguageModel, LLMRequest, LLMResponse


class CachedLanguageModel:
    def __init__(self, inner: LanguageModel, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = hashlib.sha256(request.fingerprint().encode("utf-8")).hexdigest()
        path = self._cache_dir / (key + ".json")
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=payload["text"],
                model=payload.get("model", ""),
                usage=payload.get("usage", {}),
            )
        response = self._inner.complete(request)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"text": response.text, "model": response.model, "usage": response.usage},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return response

