"""Persistent email index backed by JSONL for fast evidence retrieval."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from gnc_enrich.domain.models import EmailEvidence
from gnc_enrich.email.parser import EmlParser

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _serialize_evidence(ev: EmailEvidence) -> dict[str, Any]:
    """Convert an EmailEvidence to a JSON-serializable dict."""
    return {
        "evidence_id": ev.evidence_id,
        "message_id": ev.message_id,
        "sender": ev.sender,
        "subject": ev.subject,
        "sent_at": ev.sent_at.isoformat(),
        "body_snippet": ev.body_snippet,
        "full_body": ev.full_body,
        "parsed_amounts": [str(a) for a in ev.parsed_amounts],
        "relevance_score": ev.relevance_score,
    }


def _deserialize_evidence(d: dict) -> EmailEvidence:
    """Reconstruct an EmailEvidence from a JSON dict."""
    return EmailEvidence(
        evidence_id=d["evidence_id"],
        message_id=d["message_id"],
        sender=d["sender"],
        subject=d["subject"],
        sent_at=datetime.fromisoformat(d["sent_at"]),
        body_snippet=d.get("body_snippet", ""),
        full_body=d.get("full_body", ""),
        parsed_amounts=[Decimal(a) for a in d.get("parsed_amounts", [])],
        relevance_score=float(d.get("relevance_score", 0.0)),
    )


class EmailIndexRepository:
    """Stores and queries parsed email evidence using a JSONL index file."""

    def __init__(self) -> None:
        self._entries: list[EmailEvidence] = []
        self._indexed_files: set[str] = set()
        self._parser = EmlParser()

    def build_or_load(self, emails_dir: Path, state_dir: Path) -> None:
        """Load existing index from state_dir and index any new .eml files."""
        index_path = state_dir / "email_index.jsonl"
        manifest_path = state_dir / "email_index_manifest.json"
        state_dir.mkdir(parents=True, exist_ok=True)

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._indexed_files = set(manifest.get("indexed_files", []))

        if index_path.exists():
            for lineno, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt line %d in %s", lineno, index_path)
                    continue
                if "_schema_version" in d:
                    continue
                self._entries.append(_deserialize_evidence(d))

        if not index_path.exists() or index_path.stat().st_size == 0:
            index_path.write_text(
                json.dumps({"_schema_version": _SCHEMA_VERSION}) + "\n",
                encoding="utf-8",
            )

        new_count = 0
        eml_files = sorted(emails_dir.rglob("*.eml"))
        with index_path.open("a", encoding="utf-8") as idx_f:
            for eml_path in eml_files:
                fname = str(eml_path.relative_to(emails_dir))
                if fname in self._indexed_files:
                    continue
                try:
                    ev = self._parser.parse(eml_path)
                    self._entries.append(ev)
                    idx_f.write(json.dumps(_serialize_evidence(ev)) + "\n")
                    self._indexed_files.add(fname)
                    new_count += 1
                except Exception:
                    logger.warning("Failed to parse %s, skipping", eml_path, exc_info=True)

        manifest_data = {"_schema_version": _SCHEMA_VERSION, "indexed_files": sorted(self._indexed_files)}
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        logger.info(
            "Email index: %d total entries (%d new)", len(self._entries), new_count
        )

    def search(
        self,
        query_text: str = "",
        *,
        amount: Decimal | None = None,
        amount_tolerance: float = 0.50,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = 20,
    ) -> list[EmailEvidence]:
        """Search the in-memory index by text tokens, amount, and date range.

        When *amount* is provided, only emails containing at least one
        parsed amount within *amount_tolerance* of the target are returned.
        Date range is applied as a hard filter.
        """
        results: list[tuple[float, EmailEvidence]] = []
        query_tokens = set(query_text.lower().split()) if query_text else set()

        for ev in self._entries:
            score = 0.0

            if date_from or date_to:
                ev_date = ev.sent_at.date() if isinstance(ev.sent_at, datetime) else ev.sent_at
                if date_from and ev_date < date_from:
                    continue
                if date_to and ev_date > date_to:
                    continue
                score += 1.0

            amount_hit = False
            if amount is not None:
                tol = Decimal(str(amount_tolerance))
                for ea in ev.parsed_amounts:
                    if abs(ea - amount) <= tol:
                        amount_hit = True
                        score += 5.0
                        break
                if not amount_hit:
                    continue

            if query_tokens:
                searchable = f"{ev.sender} {ev.subject} {ev.body_snippet}".lower()
                matched = sum(1 for t in query_tokens if t in searchable)
                if matched:
                    score += matched * 0.5

            if score > 0:
                results.append((score, ev))

        results.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in results[:limit]]

    @property
    def entries(self) -> list[EmailEvidence]:
        return list(self._entries)

    def search_by_date_amount(
        self,
        tx_date: date,
        tx_amount: Decimal,
        *,
        window_days: int = 7,
        amount_tolerance: float = 0.50,
        limit: int = 20,
    ) -> list[EmailEvidence]:
        """Convenience search for transaction matching."""
        return self.search(
            amount=tx_amount,
            amount_tolerance=amount_tolerance,
            date_from=tx_date - timedelta(days=window_days),
            date_to=tx_date + timedelta(days=window_days),
            limit=limit,
        )
