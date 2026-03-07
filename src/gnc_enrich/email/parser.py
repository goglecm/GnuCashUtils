"""EML parsing primitives using Python stdlib email module."""

from __future__ import annotations

import email
import email.policy
import hashlib
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path

from gnc_enrich.domain.models import EmailEvidence

logger = logging.getLogger(__name__)

_GBP_AMOUNT_RE = re.compile(
    r"£\s?(\d{1,7}(?:[,]\d{3})*(?:\.\d{1,2})?)"
    r"|"
    r"GBP\s?(\d{1,7}(?:[,]\d{3})*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

_MAX_BODY_SNIPPET = 500


def _extract_amounts(text: str) -> list[Decimal]:
    amounts: list[Decimal] = []
    for m in _GBP_AMOUNT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        raw = raw.replace(",", "")
        try:
            amounts.append(Decimal(raw))
        except InvalidOperation:
            continue
    return amounts


def _get_body_text(msg: email.message.EmailMessage) -> str:
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = body.get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return content


class EmlParser:
    """Parses .eml files into searchable evidence records."""

    def parse(self, eml_path: Path) -> EmailEvidence:
        raw = eml_path.read_bytes()
        msg = email.message_from_bytes(raw, policy=email.policy.default)

        message_id = msg.get("Message-ID", "")
        sender = msg.get("From", "")
        subject = msg.get("Subject", "")

        date_str = msg.get("Date", "")
        try:
            sent_at = parsedate_to_datetime(date_str)
        except Exception:
            sent_at = datetime(1970, 1, 1, tzinfo=timezone.utc)

        body_text = _get_body_text(msg)
        body_snippet = body_text[:_MAX_BODY_SNIPPET]

        combined_text = f"{subject} {body_text}"
        parsed_amounts = _extract_amounts(combined_text)

        evidence_id = hashlib.sha256(
            (message_id or str(eml_path)).encode()
        ).hexdigest()[:16]

        return EmailEvidence(
            evidence_id=evidence_id,
            message_id=message_id,
            sender=sender,
            subject=subject,
            sent_at=sent_at,
            body_snippet=body_snippet,
            full_body=body_text,
            parsed_amounts=parsed_amounts,
            relevance_score=0.0,
        )
