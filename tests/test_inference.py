import json
from decimal import Decimal
from datetime import date

import pytest

from model.adapters.llm_gemma import GemmaClient
from model.adapters.llm_null import NullLanguageModel
from model.config import EnsembleSettings, GemmaSettings
from model.domain.ledger import Txn
from model.domain.types import BorrowerContext, CovenantTask, PageChunk
from model.ensemble.ensemble import CovenantEnsemble
from model.ensemble.estimate import CovenantEstimate
from model.ensemble.policy import NumericAuthoritativePolicy
from model.ensemble.semantic import SemanticEstimator, validate_semantic_payload
from model.adapters.llm_gemma import extract_json_object
from model.ensemble.numeric import extract_threshold
from model.ports.inference import LLMRequest, LLMResponse


def test_balanced_json_extraction() -> None:
    assert extract_json_object('prefix {"a": {"b": 1}} suffix') == '{"a": {"b": 1}}'
    with pytest.raises(ValueError):
        extract_json_object("no object")


def test_threshold_fallbacks() -> None:
    assert extract_threshold("коэффициент не превышал 0.42x") == (
        Decimal("0.42"),
        "at_most",
    )
    assert extract_threshold("Минимальная выручка не менее $7,100,000.00") == (
        Decimal("7100000.00"),
        "at_least",
    )


class _FixedEstimator:
    def __init__(self, estimate: CovenantEstimate) -> None:
        self.name = estimate.source
        self._estimate = estimate
        self.calls = 0

    def estimate(self, _task, _context) -> CovenantEstimate:
        self.calls += 1
        return self._estimate


def test_gaps_only_stops_after_complete_numeric_estimate() -> None:
    numeric = _FixedEstimator(
        CovenantEstimate(
            source="numeric",
            status="COMPLIANT",
            actual=Decimal("1.00"),
            confidence=0.95,
        )
    )
    gemma = _FixedEstimator(
        CovenantEstimate(
            source="gemma",
            status="BREACH",
            actual=Decimal("2.00"),
            confidence=0.78,
        )
    )
    ensemble = CovenantEnsemble(
        (numeric, gemma),
        NumericAuthoritativePolicy(),
        stop_on_complete=True,
    )

    answer = ensemble.analyze(CovenantTask("T1", "6.1"), BorrowerContext("T1", "ACC-1"))

    assert answer.status == "COMPLIANT"
    assert numeric.calls == 1
    assert gemma.calls == 0


def test_gaps_only_calls_gemma_after_numeric_gap() -> None:
    numeric = _FixedEstimator(CovenantEstimate(source="numeric", confidence=0.0))
    gemma = _FixedEstimator(
        CovenantEstimate(
            source="gemma",
            status="BREACH",
            actual=Decimal("2.00"),
            confidence=0.78,
        )
    )
    ensemble = CovenantEnsemble(
        (numeric, gemma),
        NumericAuthoritativePolicy(),
        stop_on_complete=True,
    )

    answer = ensemble.analyze(CovenantTask("T1", "6.1"), BorrowerContext("T1", "ACC-1"))

    assert answer.status == "BREACH"
    assert gemma.calls == 1


def test_semantic_prompt_has_shared_prefix_and_drops_contract_boilerplate() -> None:
    context = BorrowerContext(
        "T1",
        "ACC-1",
        pages=(
            PageChunk(
                "contract.pdf",
                1,
                "ДЕЙСТВУЮЩИЙ ДОГОВОР\nСтатья 1 — Определения\nEBITDA означает показатель.",
            ),
            PageChunk(
                "contract.pdf",
                2,
                "Статья 2 — Условия\nBOILERPLATE_SHOULD_NOT_BE_SENT",
            ),
            PageChunk("audit.pdf", 1, "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"),
        ),
        transactions=(
            Txn(
                "TXN-T1-1",
                date(2025, 1, 1),
                "ACC-1",
                "Customer",
                "Service sales settlement",
                Decimal("100"),
                "USD",
            ),
        ),
        document_kinds={"contract.pdf": "active_contract", "audit.pdf": "final_audit"},
    )
    estimator = SemanticEstimator(NullLanguageModel(), EnsembleSettings())
    first = estimator._build_request(CovenantTask("T1", "6.1", "first rule"), context).prompt
    second = estimator._build_request(CovenantTask("T1", "6.2", "second rule"), context).prompt

    assert first.split("TASK —", 1)[0] == second.split("TASK —", 1)[0]
    assert "BOILERPLATE_SHOULD_NOT_BE_SENT" not in first
    assert first.endswith("first rule\n")


def test_gemma_request_keeps_model_loaded_and_limits_generation(monkeypatch) -> None:
    settings = GemmaSettings()
    client = GemmaClient(settings)
    captured = {}

    def fake_post(_endpoint, body):
        captured.update(body)
        return {"message": {"content": '{"ok": true}'}}

    monkeypatch.setattr(client, "_post", fake_post)
    client.complete(LLMRequest("system", "prompt"))

    assert captured["keep_alive"] == "30m"
    assert captured["think"] is False
    assert captured["options"]["num_predict"] == 512


def _semantic_context(metrics=None) -> BorrowerContext:
    return BorrowerContext(
        "T1",
        "ACC-1",
        pages=(PageChunk("contract.pdf", 1, "Максимальный коэффициент не превышает 9.00x."),),
        metrics=metrics or {},
        document_kinds={"contract.pdf": "active_contract"},
    )


def _semantic_payload(**overrides):
    payload = {
        "status": "COMPLIANT",
        "actual": 0.74,
        "threshold": 9.0,
        "evidence_txn_id": None,
        "quote": "Максимальный коэффициент не превышает 9.00x.",
        "used_document": "contract.pdf",
        "related_parties_used": [],
        "reasoning": "Расчётное отношение составляет 0.736x при лимите 9.00x.",
    }
    payload.update(overrides)
    return payload


def test_semantic_response_is_validated_before_becoming_an_estimate() -> None:
    task = CovenantTask("T1", "6.1", "Максимальный коэффициент не превышает 9.00x.")

    validated = validate_semantic_payload(_semantic_payload(), task, _semantic_context())

    assert validated["actual"] == Decimal("0.74")
    assert validated["threshold"] == Decimal("9.0")


def test_semantic_response_rejects_actual_inconsistent_with_reasoning() -> None:
    task = CovenantTask("T1", "6.1", "Максимальный коэффициент не превышает 9.00x.")
    payload = _semantic_payload(
        actual=5.31,
        reasoning="Отношение: 1703882.44 / 2312216.15 ≈ 0.736x; лимит 9.00x.",
    )

    with pytest.raises(ValueError, match="actual is inconsistent"):
        validate_semantic_payload(payload, task, _semantic_context())


def test_semantic_response_rejects_status_inconsistent_with_contract() -> None:
    task = CovenantTask("T1", "6.1", "Максимальный коэффициент не превышает 9.00x.")
    payload = _semantic_payload(actual=9.45, reasoning="Отношение составляет 9.45x.")

    with pytest.raises(ValueError, match="status is inconsistent"):
        validate_semantic_payload(payload, task, _semantic_context())


class _RecordingLanguageModel:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, _request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(json.dumps(self.payload), model="test")


def test_group_capex_answer_is_rejected_when_source_metric_is_absent() -> None:
    task = CovenantTask(
        "P5",
        "6.1",
        "Максимальное отношение капитальных затрат Группы к EBITDA Заёмщика "
        "не превышает 9.00x.",
    )
    llm = _RecordingLanguageModel(_semantic_payload())
    estimator = SemanticEstimator(llm, EnsembleSettings())

    with pytest.raises(ValueError, match="group_capex is not disclosed"):
        estimator.estimate(task, _semantic_context())

    assert llm.calls == 0


def test_semantic_response_rejects_unknown_evidence_transaction() -> None:
    task = CovenantTask("T1", "6.1", "Максимальный коэффициент не превышает 9.00x.")

    with pytest.raises(ValueError, match="not present in the borrower ledger"):
        validate_semantic_payload(
            _semantic_payload(evidence_txn_id="TXN-T1-9999"),
            task,
            _semantic_context(),
        )
