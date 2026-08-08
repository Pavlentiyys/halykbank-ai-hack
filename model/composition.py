"""Default object graph assembly; the only core module selecting adapters."""

from __future__ import annotations

from typing import Optional

from model.adapters.ledger_csv import CsvLedgerRepository
from model.adapters.llm_cached import CachedLanguageModel
from model.adapters.llm_gemma import GemmaClient
from model.adapters.llm_null import NullLanguageModel
from model.adapters.pdf_pymupdf import PyMuPdfExtractor
from model.config import Settings
from model.ensemble.ensemble import CovenantEnsemble, make_default_answer
from model.ensemble.numeric import NumericEstimator
from model.ensemble.policy import NumericAuthoritativePolicy
from model.ensemble.semantic import SemanticEstimator
from model.ports.inference import LanguageModel
from model.services.context_builder import BorrowerContextBuilder
from model.services.pipeline import CovenantPipeline


def build_pipeline(
    settings: Settings,
    language_model: Optional[LanguageModel] = None,
) -> CovenantPipeline:
    llm: LanguageModel
    if language_model is not None:
        llm = language_model
    elif settings.ensemble.gemma.enabled:
        llm = GemmaClient(settings.ensemble.gemma)
    else:
        llm = NullLanguageModel()
    if settings.cache_dir is not None and settings.ensemble.gemma.enabled:
        llm = CachedLanguageModel(llm, settings.cache_dir)

    estimators = [
        SemanticEstimator(llm, settings.ensemble, settings.max_context_chars),
    ]
    if settings.ensemble.numeric_enabled:
        estimators.append(NumericEstimator(settings.ensemble))
    ensemble = CovenantEnsemble(
        estimators=estimators,
        policy=NumericAuthoritativePolicy(),
        fallback=make_default_answer,
    )
    return CovenantPipeline(
        documents=PyMuPdfExtractor(
            ocr_fallback_enabled=settings.ocr_fallback_enabled,
            ocr_language=settings.ocr_language,
        ),
        ledger=CsvLedgerRepository(settings.ensemble.fx_eur_usd),
        context_builder=BorrowerContextBuilder(settings),
        ensemble=ensemble,
        settings=settings,
    )
