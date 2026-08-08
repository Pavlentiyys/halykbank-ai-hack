from decimal import Decimal

import pytest

from model.adapters.llm_gemma import extract_json_object
from model.ensemble.numeric import extract_threshold


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

