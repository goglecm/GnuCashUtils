"""Receipt-to-transaction one-to-one matching with conflict safeguards."""

from __future__ import annotations

import logging
from decimal import Decimal

from gnc_enrich.domain.models import ReceiptEvidence, Transaction

logger = logging.getLogger(__name__)


class ReceiptMatcher:
    """Produces the best one-to-one receipt match for a transaction.

    Tracks previously assigned receipts to prevent double-matching.
    """

    def __init__(
        self,
        receipts: list[ReceiptEvidence],
        amount_tolerance: float = 0.50,
    ) -> None:
        self._receipts = list(receipts)
        self._tolerance = Decimal(str(amount_tolerance))
        self._assigned: set[str] = set()

    def match(self, tx: Transaction) -> ReceiptEvidence | None:
        """Find the best unassigned receipt for the transaction."""
        best: ReceiptEvidence | None = None
        best_score = -1.0

        for receipt in self._receipts:
            if receipt.evidence_id in self._assigned:
                continue
            score = self._score(tx, receipt)
            if score > best_score:
                best_score = score
                best = receipt

        if best is not None and best_score > 0:
            self._assigned.add(best.evidence_id)
            best = ReceiptEvidence(
                evidence_id=best.evidence_id,
                source_path=best.source_path,
                ocr_text=best.ocr_text,
                parsed_total=best.parsed_total,
                line_items=best.line_items,
                relevance_score=round(best_score, 4),
            )
            return best

        return None

    def _score(self, tx: Transaction, receipt: ReceiptEvidence) -> float:
        """Score a receipt against a transaction. Returns 0 if OCR found no total."""
        if receipt.parsed_total is None:
            return 0.0

        diff = abs(receipt.parsed_total - tx.amount)
        if diff <= self._tolerance:
            return 5.0 + (1.0 / (1.0 + float(diff)))
        return 0.0

    @property
    def assigned_ids(self) -> set[str]:
        return set(self._assigned)

    def is_amount_compatible(
        self, receipt: ReceiptEvidence, tx: Transaction
    ) -> bool:
        """Check whether receipt total is within tolerance of transaction amount."""
        if receipt.parsed_total is None:
            return False
        return abs(receipt.parsed_total - tx.amount) <= self._tolerance
