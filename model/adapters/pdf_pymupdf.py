"""PDF text-layer adapter with a best-effort OCR fallback.

OCR is a bonus, never a precondition: a page that cannot be recognised yields
empty text so the pipeline keeps its default answers instead of dying.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Sequence

from model.domain.types import DocumentRef, PageChunk

log = logging.getLogger(__name__)


class PyMuPdfExtractor:
    def __init__(self, ocr_fallback_enabled: bool = True, ocr_language: str = "rus+eng") -> None:
        self._ocr_fallback_enabled = ocr_fallback_enabled
        self._ocr_language = ocr_language
        self._ocr_available = True

    def extract(self, source: DocumentRef) -> Sequence[PageChunk]:
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to read PDF documents") from exc

        pages = []
        with fitz.open(source.path) as document:
            for index, page in enumerate(document):
                text = page.get_text("text") or ""
                if self._needs_ocr(text, page):
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
                    text = self._recognise(pixmap.tobytes("png"), source, index + 1) or text
                pages.append(
                    PageChunk(
                        doc_name=source.path.name,
                        page=index + 1,
                        text=text,
                    )
                )
        return pages

    def _needs_ocr(self, text: str, page: object) -> bool:
        return (
            self._ocr_fallback_enabled
            and self._ocr_available
            and len(text.strip()) <= 3
            and bool(page.get_images(full=True))  # type: ignore[attr-defined]
        )

    def _recognise(self, image: bytes, source: DocumentRef, page_number: int) -> str:
        """Return recognised text, or an empty string when OCR is unavailable."""
        try:
            completed = subprocess.run(
                ["tesseract", "stdin", "stdout", "-l", self._ocr_language, "--psm", "6"],
                input=image,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            # Единственный раз на экстрактор: дальше просто отдаём пустой текст.
            self._ocr_available = False
            log.warning(
                "tesseract not found; pages without a text layer stay empty "
                "(install tesseract with the %s language data to enable OCR)",
                self._ocr_language,
            )
            return ""
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("OCR failed for %s page %s: %s", source.path.name, page_number, exc)
            return ""
        return completed.stdout.decode("utf-8", errors="replace")
