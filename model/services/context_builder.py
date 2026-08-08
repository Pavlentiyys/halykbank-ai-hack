"""Build borrower contexts once per dataset from documents and the ledger."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from model.config import Settings
from model.domain.ledger import Txn
from model.domain.types import BorrowerContext, DatasetRef, DocumentRef, PageChunk
from model.ports.extraction import LedgerRepository, TextExtractor
from model.services.metrics import aggregate_metrics

SCENARIO_ACCOUNTS: Mapping[str, str] = {
    "P1": "ACC-7801",
    "P2": "ACC-7802",
    "P3": "ACC-7803",
    "P4": "ACC-7804",
    "P5": "ACC-7805",
    "P6": "ACC-7806",
    "P7": "ACC-7807",
    "P8": "ACC-7808",
    "P9": "ACC-7809",
    "P10": "ACC-7810",
    "B1": "ACC-7201",
    "B4": "ACC-7204",
}


@dataclass(frozen=True)
class _Document:
    name: str
    pages: Tuple[PageChunk, ...]
    text: str
    kind: str


def _compact(text: str) -> str:
    return " ".join(text.split())


def classify_document(pages: Sequence[PageChunk]) -> str:
    text = _compact("\n".join(page.text for page in pages))
    head = text[:5000]
    if "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ" in head or "НЕ ПРИМЕНЯЕТСЯ" in head:
        return "inactive_contract"
    if "ДОГОВОР БАНКОВСКОГО ЗАЙМА" in head or "Д О Г О В О Р Б А Н К О В С К О Г О З А Й М А" in head:
        return "active_contract"
    if "НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА" in head:
        return "interim_workpaper"
    if "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ" in text:
        return "final_audit"
    if "Доля голосующих прав" in text or ("Досье" in head and "KYC" in head):
        return "kyc"
    if "СЛУЖЕБНАЯ ЗАПИСКА КАЗНАЧЕЙСТВА" in text:
        return "treasury_memo"
    return "other"


def extract_covenants(text: str) -> Dict[str, str]:
    compact = _compact(text)
    start = compact.find("6.1")
    end = compact.find("Статья 7", start)
    if start < 0:
        return {}
    block = compact[start : end if end >= 0 else len(compact)]
    markers = {
        "6.1": 0,
        "6.2": block.find("Пункт 6.2"),
        "6.3": block.find("Пункт 6.3"),
    }
    if markers["6.2"] < 0 or markers["6.3"] < 0:
        return {}
    return {
        "6.1": block[: markers["6.2"]].strip(),
        "6.2": block[markers["6.2"] : markers["6.3"]].strip(),
        "6.3": block[markers["6.3"] :].strip(),
    }


def extract_related_parties(
    text: str, fallback_threshold: Decimal
) -> Tuple[str, ...]:
    compact = _compact(text)
    marker = compact.find("Доля голосующих прав")
    if marker < 0:
        return ()
    end = compact.find("Организации, в которых", marker)
    table = compact[marker + len("Доля голосующих прав") : end if end >= 0 else marker + 800]
    threshold_match = re.search(
        r"владеет\s+([0-9]+(?:\.[0-9]+)?)%\s+и более", compact[end : end + 400]
    )
    threshold = (
        Decimal(threshold_match.group(1)) / Decimal("100")
        if threshold_match
        else fallback_threshold
    )
    parties = []
    pattern = re.compile(r'(.+?)\s+([0-9]+(?:\.[0-9]+)?)%')
    cursor = 0
    for match in pattern.finditer(table):
        name = table[cursor : match.start(2)].strip().strip('"')
        name = re.sub(r"^.*?([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9&.,\-\" ]+)$", r"\1", name)
        share = Decimal(match.group(2)) / Decimal("100")
        if name and share >= threshold:
            parties.append(name.strip())
        cursor = match.end()
    return tuple(parties)


def _recover_missing_amount(txn: Txn, documents: Iterable[_Document]) -> Txn:
    if txn.amount is not None:
        return txn
    for document in documents:
        compact = _compact(document.text)
        index = compact.find(txn.txn_id)
        if index < 0:
            continue
        window = compact[index : index + 500]
        match = re.search(
            r"фактическая сумма операции составляет\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)",
            window,
            flags=re.IGNORECASE,
        )
        if match:
            return txn.with_amount(-Decimal(match.group(1).replace(",", "")))
    return txn


def _apply_disclosed_fx(
    txn: Txn,
    documents: Iterable[_Document],
    configured_fx: Decimal,
) -> Txn:
    if txn.amount is None or txn.currency != "EUR":
        return txn
    for document in documents:
        if document.kind != "final_audit":
            continue
        compact = _compact(document.text)
        if txn.counterparty not in compact:
            continue
        match = re.search(
            r"счёт на сумму\s*([0-9][0-9,]*\.[0-9]+)\s*EUR.{0,180}?в размере\s*\$([0-9][0-9,]*\.[0-9]+)",
            compact,
            flags=re.IGNORECASE,
        )
        if match:
            eur = Decimal(match.group(1).replace(",", ""))
            usd = Decimal(match.group(2).replace(",", ""))
            if eur and configured_fx:
                raw = txn.amount / configured_fx
                return txn.with_amount(raw * (usd / eur))
    return txn


def _excluded_transactions(documents: Iterable[_Document]) -> Tuple[str, ...]:
    excluded = []
    for document in documents:
        if document.kind != "final_audit":
            continue
        compact = _compact(document.text)
        for match in re.finditer(r"(TXN-[A-Z0-9]+-\d+)", compact):
            window = compact[match.start() : match.start() + 450].casefold()
            if (
                ("исключ" in window and ("2026" in window or "ковенантного периода" in window))
                or ("относится к услугам" in window and "2026" in window)
            ):
                excluded.append(match.group(1))
    return tuple(dict.fromkeys(excluded))


def _category_name(text: str) -> str:
    value = text.casefold()
    if "операцион" in value:
        return "operating"
    if "страхов" in value:
        return "insurance"
    if "процент" in value:
        return "interest"
    if "капитальн" in value:
        return "capex"
    if "оплат" in value and "труд" in value:
        return "personnel"
    return "other"


def _category_overrides(
    documents: Iterable[_Document], transactions: Sequence[Txn]
) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    by_amount: Dict[Decimal, Txn] = {
        abs(txn.amount): txn for txn in transactions if txn.amount is not None
    }
    for document in documents:
        if document.kind in {"inactive_contract", "interim_workpaper"}:
            continue
        compact = _compact(document.text)
        one_time_start = compact.find("Ниже приведены разовые статьи")
        one_time_end = compact.find("Разовыми для целей ковенантов", one_time_start)
        if one_time_start >= 0 and one_time_end > one_time_start:
            one_time_block = compact[one_time_start:one_time_end]
            for amount_text in re.findall(r"\$([0-9][0-9,]*\.[0-9]+)", one_time_block):
                txn = by_amount.get(Decimal(amount_text.replace(",", "")))
                if txn is not None:
                    overrides[txn.txn_id] = "operating"
        for match in re.finditer(r"(TXN-[A-Z0-9]+-\d+)", compact):
            window = compact[match.start() : match.start() + 600]
            target = re.search(
                r"переклассифицирован\w*[^.]{0,180}?как\s+([А-Яа-яЁё ]+?)(?:\.|;|,| Основание)",
                window,
                flags=re.IGNORECASE,
            )
            if target:
                overrides[match.group(1)] = _category_name(target.group(1))
        for match in re.finditer(
            r"Сумма в размере\s*\$([0-9][0-9,]*\.[0-9]+).{0,300}?переклассифицирован\w*[^.]{0,180}?как\s+([А-Яа-яЁё ]+?)(?:\.|;|,| Основание)",
            compact,
            flags=re.IGNORECASE,
        ):
            amount = Decimal(match.group(1).replace(",", ""))
            txn = by_amount.get(amount)
            if txn is not None:
                overrides[txn.txn_id] = _category_name(match.group(2))
    return {txn_id: category for txn_id, category in overrides.items() if category != "other"}


def _document_only_metrics(documents: Iterable[_Document]) -> Dict[str, Decimal]:
    result: Dict[str, Decimal] = {}
    for document in documents:
        if document.kind != "final_audit":
            continue
        compact = _compact(document.text)
        one_time_start = compact.find("Ниже приведены разовые статьи")
        one_time_end = compact.find("Разовыми для целей ковенантов", one_time_start)
        if one_time_start >= 0 and one_time_end > one_time_start:
            one_time_block = compact[one_time_start:one_time_end]
            threshold_match = re.search(
                r"Разовыми для целей ковенантов признаются статьи в сумме не менее\s*\$([0-9][0-9,]*\.[0-9]+)",
                compact[one_time_end : one_time_end + 300],
                flags=re.IGNORECASE,
            )
            if threshold_match:
                threshold = Decimal(threshold_match.group(1).replace(",", ""))
                amounts = [
                    Decimal(value.replace(",", ""))
                    for value in re.findall(r"\$([0-9][0-9,]*\.[0-9]+)", one_time_block)
                ]
                result["one_time_addback"] = sum(
                    (amount for amount in amounts if amount >= threshold),
                    Decimal("0"),
                )
        match = re.search(
            r"обязательство по программе выходных пособий в размере\s*\$([0-9][0-9,]*\.[0-9]+)",
            compact,
            flags=re.IGNORECASE,
        )
        if match:
            result["severance_obligation"] = Decimal(match.group(1).replace(",", ""))
        group_capex = re.search(
            r"капитальн(?:ые|ых) затрат(?:ы)? Группы.{0,120}?\$([0-9][0-9,]*\.[0-9]+)",
            compact,
            flags=re.IGNORECASE,
        )
        if group_capex:
            result["group_capex"] = Decimal(group_capex.group(1).replace(",", ""))
    return result


class BorrowerContextBuilder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_all(
        self,
        dataset: DatasetRef,
        documents: TextExtractor,
        ledger_repository: LedgerRepository,
    ) -> Dict[str, BorrowerContext]:
        ledger = ledger_repository.load(dataset.resolve("master_ledger_2025.csv"))
        document_dir = dataset.resolve("documents")
        extracted: List[_Document] = []
        for path in sorted(document_dir.glob("*.pdf")):
            pages = tuple(documents.extract(DocumentRef(path)))
            text = "\n".join(page.text for page in pages)
            extracted.append(_Document(path.name, pages, text, classify_document(pages)))

        contexts: Dict[str, BorrowerContext] = {}
        for scenario_id, account_id in SCENARIO_ACCOUNTS.items():
            borrower_docs = tuple(doc for doc in extracted if account_id in doc.text)
            pages = tuple(page for doc in borrower_docs for page in doc.pages)
            kinds = {doc.name: doc.kind for doc in borrower_docs}
            active = next((doc for doc in borrower_docs if doc.kind == "active_contract"), None)
            kyc = next((doc for doc in borrower_docs if doc.kind == "kyc"), None)
            covenant_texts = extract_covenants(active.text) if active else {}
            related_parties = extract_related_parties(
                kyc.text if kyc else "",
                self._settings.ensemble.related_party_threshold,
            )
            txns = tuple(
                _apply_disclosed_fx(
                    _recover_missing_amount(txn, borrower_docs),
                    borrower_docs,
                    self._settings.ensemble.fx_eur_usd,
                )
                for txn in ledger.for_account(account_id)
            )
            if not related_parties:
                related_parties = tuple(
                    txn.counterparty
                    for txn in txns
                    if "management advisory retainer" in txn.description.casefold()
                )
            excluded = _excluded_transactions(borrower_docs)
            overrides = _category_overrides(borrower_docs, txns)
            metrics = aggregate_metrics(
                txns,
                related_parties,
                overrides=overrides,
                excluded_txn_ids=excluded,
            )
            metrics.update(_document_only_metrics(borrower_docs))
            contexts[scenario_id] = BorrowerContext(
                scenario_id=scenario_id,
                account_id=account_id,
                pages=pages,
                metrics=metrics,
                transactions=txns,
                related_parties=related_parties,
                covenant_texts=covenant_texts,
                document_kinds=kinds,
                excluded_txn_ids=excluded,
                category_overrides=overrides,
            )
        return contexts
