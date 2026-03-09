"""EML parsing primitives using Python stdlib email module."""

from __future__ import annotations

import email
import email.policy
import hashlib
import html as html_mod
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path

from gnc_enrich.domain.models import EmailEvidence

logger = logging.getLogger(__name__)

_GBP_AMOUNT_RE = re.compile(
    r"£\s?(\d{1,7}(?:[,]\d{3})*(?:\.\d{1,2})?)" r"|" r"GBP\s?(\d{1,7}(?:[,]\d{3})*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

_MAX_BODY_SNIPPET = 500
_AMOUNT_CONTEXT_CHARS = 200


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple newlines and strip each line for readable display."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_amount_context(filtered_body: str, amount: Decimal) -> str:
    """Return a short snippet of body text around the first occurrence of the amount."""
    needle = str(amount)
    if needle not in filtered_body:
        needle_alt = str(int(amount)) if amount == int(amount) else needle
        if needle_alt not in filtered_body:
            return filtered_body[:_AMOUNT_CONTEXT_CHARS] if filtered_body else ""
    pos = filtered_body.find(needle)
    if pos < 0:
        pos = filtered_body.find(str(int(amount)))
    if pos < 0:
        return filtered_body[:_AMOUNT_CONTEXT_CHARS] if filtered_body else ""
    start = max(0, pos - _AMOUNT_CONTEXT_CHARS // 2)
    end = min(len(filtered_body), pos + len(needle) + _AMOUNT_CONTEXT_CHARS // 2)
    return _normalise_whitespace(filtered_body[start:end])


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_CSS_COMMENT_RE = re.compile(r"/\*\*?[^*]*\*+/", re.DOTALL)
_CSS_RULE_RE = re.compile(r"\.[\w-]+\s*\{[^}]*\}|[\w#.-]+\s*\{\s*[^}]*\}", re.IGNORECASE)

_SIGNATURE_MARKERS_STRIPPED = {"___", "---", "Sent from", "Get Outlook"}


def _extract_amounts(text: str) -> list[Decimal]:
    """Extract GBP amounts from text using regex."""
    amounts: list[Decimal] = []
    for m in _GBP_AMOUNT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        raw = raw.replace(",", "")
        try:
            amounts.append(Decimal(raw))
        except InvalidOperation:
            continue
    return amounts


def _strip_html(text: str) -> str:
    """Remove HTML tags (including <style> blocks), decode entities, strip CSS comments, collapse whitespace."""
    text = _STYLE_BLOCK_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_mod.unescape(text)
    text = _CSS_COMMENT_RE.sub(" ", text)
    text = _CSS_RULE_RE.sub(" ", text)
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    return text.strip()


def _filter_body(body_text: str) -> str:
    """Strip signatures, quoted replies, and boilerplate from an email body.

    Handles the RFC 3676 signature delimiter ``-- `` (dash-dash-space)
    as well as common informal markers like ``---`` or ``___``.
    """
    lines: list[str] = []
    for line in body_text.splitlines():
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if raw == "-- " or any(stripped.startswith(m) for m in _SIGNATURE_MARKERS_STRIPPED):
            break
        if stripped.startswith(">"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _get_body_text(msg: email.message.EmailMessage) -> str:
    """Extract the plain-text body from an email message."""
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = body.get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return _strip_html(content)


class EmlParser:
    """Parses .eml files into searchable evidence records."""

    def parse(self, eml_path: Path) -> EmailEvidence:
        """Parse an .eml file into an EmailEvidence record."""
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
        filtered_body = _filter_body(body_text)
        display_body = _normalise_whitespace(filtered_body)
        body_snippet = display_body[:_MAX_BODY_SNIPPET]
        parsed_amounts = _extract_amounts(f"{subject} {filtered_body}")
        amount_context = ""
        if parsed_amounts:
            amount_context = _extract_amount_context(filtered_body, parsed_amounts[0])
        if not amount_context and display_body:
            amount_context = display_body[: _AMOUNT_CONTEXT_CHARS * 2]

        evidence_id = hashlib.sha256((message_id or str(eml_path)).encode()).hexdigest()[:16]

        return EmailEvidence(
            evidence_id=evidence_id,
            message_id=message_id,
            sender=sender,
            subject=subject,
            sent_at=sent_at,
            body_snippet=body_snippet,
            full_body=body_text,
            filtered_body=display_body,
            amount_context=amount_context,
            parsed_amounts=parsed_amounts,
            relevance_score=0.0,
        )
