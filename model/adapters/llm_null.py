"""Deterministic language model substitute for offline tests."""

import json

from model.ports.inference import LLMRequest, LLMResponse


class NullLanguageModel:
    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "status": "COMPLIANT",
            "actual": 0.01,
            "threshold": None,
            "evidence_txn_id": None,
            "quote": "",
            "used_document": "",
            "related_parties_used": [],
            "reasoning": "offline deterministic response",
        }
        return LLMResponse(text=json.dumps(payload), model="null")

