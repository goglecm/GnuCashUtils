"""Email-to-transaction matching using date/amount/text scoring."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from gnc_enrich.domain.models import EmailEvidence, Transaction
from gnc_enrich.email.index import EmailIndexRepository

logger = logging.getLogger(__name__)

_WEIGHT_AMOUNT = 5.0
_WEIGHT_DATE_PROXIMITY = 2.0
_WEIGHT_TEXT_TOKEN = 0.5


class EmailMatcher:
    """Matches transactions to email evidence using scored multi-signal ranking."""

    def __init__(
        self,
        index: EmailIndexRepository,
        date_window_days: int = 7,
        amount_tolerance: float = 0.50,
    ) -> None:
        self._index = index
        self._window = date_window_days
        self._tolerance = amount_tolerance

    def match(self, tx: Transaction) -> list[EmailEvidence]:
        """Return email evidence ranked by relevance for the given transaction.

        Only emails that have at least one parsed amount matching within
        tolerance are returned.  Date window is used as a pre-filter,
        not as a standalone matching signal.
        """
        date_from = tx.posted_date - timedelta(days=self._window)
        date_to = tx.posted_date + timedelta(days=self._window)

        candidates = self._index.search(
            amount=tx.amount,
            amount_tolerance=self._tolerance,
            date_from=date_from,
            date_to=date_to,
            limit=50,
        )

        scored: list[tuple[float, EmailEvidence]] = []
        desc_tokens = set(tx.description.lower().split())

        for ev in candidates:
            has_amount, score = self._score(tx, ev, desc_tokens)
            if not has_amount:
                continue
            ev_copy = EmailEvidence(
                evidence_id=ev.evidence_id,
                message_id=ev.message_id,
                sender=ev.sender,
                subject=ev.subject,
                sent_at=ev.sent_at,
                body_snippet=ev.body_snippet,
                full_body=ev.full_body,
                parsed_amounts=ev.parsed_amounts,
                relevance_score=round(score, 4),
            )
            scored.append((score, ev_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored]

    def _score(
        self,
        tx: Transaction,
        ev: EmailEvidence,
        desc_tokens: set[str],
    ) -> tuple[bool, float]:
        """Compute a weighted relevance score for one email against a transaction.

        Returns ``(has_amount_match, score)`` so the caller can drop
        emails that lack any amount overlap.
        """
        score = 0.0
        has_amount = False

        tol = Decimal(str(self._tolerance))
        for ea in ev.parsed_amounts:
            if abs(ea - tx.amount) <= tol:
                has_amount = True
                score += _WEIGHT_AMOUNT
                break

        ev_date = ev.sent_at.date() if isinstance(ev.sent_at, datetime) else ev.sent_at
        day_diff = abs((ev_date - tx.posted_date).days)
        if day_diff <= self._window:
            proximity = 1.0 - (day_diff / max(self._window, 1))
            score += _WEIGHT_DATE_PROXIMITY * proximity

        ev_text = f"{ev.sender} {ev.subject} {ev.body_snippet}".lower()
        matched = sum(1 for t in desc_tokens if len(t) > 2 and t in ev_text)
        score += matched * _WEIGHT_TEXT_TOKEN

        return has_amount, score
