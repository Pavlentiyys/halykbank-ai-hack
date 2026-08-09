"""Ollama-compatible Gemma adapter with schema-constrained JSON output."""

from __future__ import annotations

import json
import time
from json import JSONDecodeError
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model.config import GemmaSettings
from model.ports.inference import LLMRequest, LLMResponse


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    json.loads(candidate)
                except JSONDecodeError as exc:
                    raise ValueError("model returned invalid JSON") from exc
                return candidate
    raise ValueError("model response contains unbalanced JSON")


class GemmaClient:
    def __init__(self, settings: GemmaSettings) -> None:
        self._settings = settings

    def complete(self, request: LLMRequest) -> LLMResponse:
        endpoint = self._settings.endpoint.rstrip("/") + "/api/chat"
        body: Dict[str, Any] = {
            "model": self._settings.model_id,
            "stream": False,
            "keep_alive": self._settings.keep_alive,
            "think": self._settings.think,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "options": {
                "temperature": self._settings.temperature,
                "num_predict": self._settings.max_output_tokens,
                "num_ctx": self._settings.num_ctx,
            },
        }
        if request.schema is not None:
            body["format"] = request.schema

        last_error = None
        for attempt in range(2):
            try:
                response = self._post(endpoint, body)
                content = str(response.get("message", {}).get("content", ""))
                if request.schema is not None:
                    content = extract_json_object(content)
                usage = {
                    "prompt_tokens": int(response.get("prompt_eval_count", 0)),
                    "completion_tokens": int(response.get("eval_count", 0)),
                }
                return LLMResponse(
                    text=content,
                    model=str(response.get("model", self._settings.model_id)),
                    usage=usage,
                )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt < 1:
                    time.sleep(2**attempt)
        raise RuntimeError("Gemma request failed after retries") from last_error

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._settings.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))
