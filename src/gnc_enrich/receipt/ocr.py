"""Receipt OCR abstraction."""

from pathlib import Path

from gnc_enrich.domain.models import ReceiptEvidence


class ReceiptOcrEngine:
    """Extracts text and totals from JPG/JPEG/HEIC receipts."""

    def parse(self, receipt_path: Path) -> ReceiptEvidence:
        raise NotImplementedError
