"""Counterparty name normalisation for KYC-to-ledger matching."""

from __future__ import annotations

import re
import unicodedata

_LEGAL_SUFFIXES = {
    "llp",
    "jsc",
    "ltd",
    "llc",
    "lp",
    "inc",
    "corp",
    "company",
    "co",
}
_DECORATORS = {
    "group",
    "trading",
    "house",
    "supply",
    "enterprise",
    "service",
    "centre",
    "works",
}


def normalize_counterparty(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).lower()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"\bl\s*\.\s*l\s*\.\s*p\s*\.?\b", " llp ", value)
    value = re.sub(r"[^a-zа-яё0-9]+", " ", value, flags=re.IGNORECASE)
    tokens = [token for token in value.split() if token not in _LEGAL_SUFFIXES]
    while tokens and tokens[-1] in _DECORATORS:
        tokens.pop()
    return " ".join(tokens)


def counterparties_match(left: str, right: str) -> bool:
    return bool(left and right) and normalize_counterparty(left) == normalize_counterparty(right)

