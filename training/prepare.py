"""Build semantic SFT data without reading covenant answer keys."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping

from model import DatasetRef, Settings, build_pipeline
from model.ensemble.numeric import extract_threshold
from model.services.metrics import category_for

_SPLIT = {
    "P1": "train",
    "P2": "train",
    "P3": "train",
    "P4": "train",
    "P5": "train",
    "P6": "train",
    "P7": "train",
    "P8": "train",
    "P9": "valid",
    "P10": "valid",
    "B1": "test",
    "B4": "test",
}


def _message(system: str, user: str, answer: Mapping[str, object]) -> Dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, sort_keys=True),
            },
        ]
    }


def _formula_kind(text: str) -> str:
    value = text.casefold()
    if "springing drawdown" in value:
        return "financing_to_ebitda"
    if "capital intensity" in value or "капиталоёмк" in value:
        return "capex_to_operating_plus_rent"
    if "cover of applications" in value:
        return "sources_to_applications"
    if "proportion of revenue" in value:
        return "related_party_to_revenue"
    if "платеж" in value and ("связан" in value or "аффилирован" in value):
        return "related_party_payments"
    if "капитальн" in value and "затрат" in value:
        return "capital_expenditure"
    if "выруч" in value:
        return "revenue_based"
    if "процент" in value and "покрыт" in value:
        return "interest_coverage"
    if "персонал" in value:
        return "personnel_based"
    return "custom_ratio"


def _examples_for_context(context: object) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for covenant_id, text in context.covenant_texts.items():
        threshold, direction = extract_threshold(text)
        examples.append(
            _message(
                "Извлеки определение финансового ковенанта. Не считай actual и не придумывай данные.",
                text,
                {
                    "covenant_id": covenant_id,
                    "direction": direction,
                    "formula_kind": _formula_kind(text),
                    "threshold": float(threshold) if threshold is not None else None,
                },
            )
        )

    pages_by_document: Dict[str, list] = defaultdict(list)
    for page in context.pages:
        pages_by_document[page.doc_name].append(page)
    for document_name, pages in pages_by_document.items():
        kind = context.document_kinds.get(document_name, "other")
        excerpt = "\n".join(page.text for page in pages[:2])[:1200]
        examples.append(
            _message(
                "Классифицируй статус документа по явным пометкам. Верни JSON.",
                excerpt,
                {"document_kind": kind, "usable_for_final_covenant": kind not in {"inactive_contract", "interim_workpaper", "other"}},
            )
        )

    if context.related_parties:
        kyc_pages = [
            page.text
            for page in context.pages
            if context.document_kinds.get(page.doc_name) == "kyc"
        ]
        if kyc_pages:
            kyc_text = "\n".join(kyc_pages)
            marker = kyc_text.find("Доля голосующих прав")
            if marker >= 0:
                kyc_text = kyc_text[max(0, marker - 300) : marker + 1500]
            else:
                kyc_text = kyc_text[:1500]
            examples.append(
                _message(
                    "Извлеки только связанные стороны, удовлетворяющие порогу из KYC. Верни JSON.",
                    kyc_text,
                    {"related_parties": list(context.related_parties)},
                )
            )

    per_category: Dict[str, int] = defaultdict(int)
    for txn in context.transactions:
        category = context.category_overrides.get(txn.txn_id) or category_for(txn.description)
        if category == "other" or per_category[category] >= 3:
            continue
        per_category[category] += 1
        examples.append(
            _message(
                "Классифицируй назначение банковской операции. Положительная сумма сама по себе не означает выручку.",
                "description: {}\namount_sign: {}".format(
                    txn.description,
                    "missing" if txn.amount is None else ("positive" if txn.amount > 0 else "negative"),
                ),
                {"category": category},
            )
        )
    return examples


def prepare_training_data(
    dataset_root: Path,
    output_dir: Path,
    settings: Settings,
) -> Dict[str, object]:
    pipeline = build_pipeline(settings)
    contexts = pipeline.build_contexts(DatasetRef(dataset_root))
    split_examples: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    scenario_counts: Dict[str, int] = {}
    for scenario_id, context in contexts.items():
        examples = _examples_for_context(context)
        split_examples[_SPLIT[scenario_id]].extend(examples)
        scenario_counts[scenario_id] = len(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, examples in split_examples.items():
        content = "".join(
            json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
            for example in examples
        )
        (output_dir / (split + ".jsonl")).write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest: Dict[str, object] = {
        "version": 1,
        "purpose": "semantic covenant SFT without answer-key leakage",
        "ground_truth_used": False,
        "splits": {name: len(rows) for name, rows in split_examples.items()},
        "scenario_examples": scenario_counts,
        "sha256": hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
