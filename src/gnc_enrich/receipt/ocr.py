"""Receipt OCR using Tesseract with optional LLM fallback for degraded images."""

from __future__ import annotations

import hashlib
import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytesseract
from PIL import Image

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import LineItem, ReceiptEvidence
from gnc_enrich.llm.client import LlmClient

logger = logging.getLogger(__name__)

_TOTAL_RE = re.compile(
    r"(?:total|grand\s*total|amount\s*due|balance\s*due|subtotal)"
    r"\s*[:=]?\s*£?\s*(\d{1,7}(?:[,]\d{3})*\.\d{2})",
    re.IGNORECASE,
)

_LINE_ITEM_RE = re.compile(
    r"^(.{2,40}?)\s+£?\s*(\d{1,5}\.\d{2})\s*$",
    re.MULTILINE,
)

_HEIC_EXTENSIONS = {".heic", ".heif"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def _open_image(path: Path) -> Image.Image:
    if path.suffix.lower() in _HEIC_EXTENSIONS:
        import pillow_heif

        heif_file = pillow_heif.open_heif(str(path))
        return heif_file.to_pillow()
    return Image.open(path)


def _extract_total(text: str) -> Decimal | None:
    """Find the receipt total, preferring 'Total' over 'Subtotal'."""
    best: Decimal | None = None
    for m in _TOTAL_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            val = Decimal(raw)
        except InvalidOperation:
            continue
        keyword = m.group(0).lower()
        if "subtotal" not in keyword:
            return val
        if best is None:
            best = val
    return best


def _extract_line_items(text: str) -> list[LineItem]:
    items: list[LineItem] = []
    for m in _LINE_ITEM_RE.finditer(text):
        desc = m.group(1).strip()
        raw_amount = m.group(2)
        try:
            amount = Decimal(raw_amount)
        except InvalidOperation:
            continue
        items.append(LineItem(description=desc, amount=amount))
    return items


class ReceiptOcrEngine:
    """Extracts text and totals from receipt images using Tesseract OCR."""

    def __init__(self, llm_config: LlmConfig | None = None) -> None:
        self._llm_config = llm_config

    def parse(self, receipt_path: Path) -> ReceiptEvidence:
        if not receipt_path.exists():
            raise FileNotFoundError(f"Receipt not found: {receipt_path}")

        if receipt_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {receipt_path.suffix}")

        img = _open_image(receipt_path)
        try:
            ocr_text = pytesseract.image_to_string(img)
        finally:
            img.close()

        parsed_total = _extract_total(ocr_text)
        line_items = _extract_line_items(ocr_text)

        evidence_id = hashlib.sha256(str(receipt_path).encode()).hexdigest()[:16]

        evidence = ReceiptEvidence(
            evidence_id=evidence_id,
            source_path=str(receipt_path),
            ocr_text=ocr_text,
            parsed_total=parsed_total,
            line_items=line_items,
        )

        if (
            self._llm_config
            and self._llm_config.mode != LlmMode.DISABLED
            and self._llm_config.endpoint
            and self._llm_config.model_name
            and parsed_total is None
        ):
            evidence = self._try_llm_fallback(evidence, receipt_path)

        return evidence

    def _try_llm_fallback(self, evidence: ReceiptEvidence, receipt_path: Path) -> ReceiptEvidence:
        """Attempt LLM-based extraction when Tesseract fails to find a total."""
        logger.info("Tesseract found no total; attempting LLM fallback for %s", receipt_path)
        try:
            import json

            client = LlmClient(self._llm_config)
            if not client.enabled:
                return evidence
            resp_data = client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract the total amount and line items from this receipt OCR text. "
                            "Return JSON with keys: total (string), items (list of {description, amount}).\n\n"
                            f"{evidence.ocr_text}"
                        ),
                    }
                ],
                max_tokens=self._llm_config.max_tokens,
                temperature=self._llm_config.temperature,
            )
            if not resp_data:
                return evidence
            # For backwards compatibility with existing tests/spec, assume an
            # OpenAI-style response shape and extract the message content.
            msg_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not msg_text:
                return evidence
            data = json.loads(msg_text)
            if "total" in data and data["total"]:
                evidence.parsed_total = Decimal(str(data["total"]))
            if "items" in data and isinstance(data["items"], list):
                evidence.line_items = []
                for it in data["items"]:
                    if not isinstance(it, dict):
                        continue
                    try:
                        desc = it.get("description") or ""
                        amt = it.get("amount")
                        if amt is not None:
                            evidence.line_items.append(
                                LineItem(description=str(desc), amount=Decimal(str(amt)))
                            )
                    except (InvalidOperation, ValueError, TypeError):
                        continue
        except Exception:
            logger.warning("LLM fallback failed for %s", receipt_path, exc_info=True)

        return evidence
