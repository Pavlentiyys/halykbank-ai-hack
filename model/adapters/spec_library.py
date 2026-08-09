"""Disk-backed library of covenant specifications learned so far."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from model.domain.formula_spec import FormulaSpec, spec_from_payload, spec_to_payload


class SpecLibrary:
    """Remembers how a covenant wording was parsed, so it is parsed once."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._entries: Dict[str, FormulaSpec] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for key, entry in payload.get("specs", {}).items():
            self._entries[key] = spec_from_payload(entry)

    def get(self, key: str) -> Optional[FormulaSpec]:
        self._load()
        return self._entries.get(key)

    def remember(self, key: str, spec: FormulaSpec) -> None:
        self._load()
        self._entries[key] = spec
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "specs": {name: spec_to_payload(value) for name, value in sorted(self._entries.items())},
        }
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def __len__(self) -> int:
        self._load()
        return len(self._entries)
