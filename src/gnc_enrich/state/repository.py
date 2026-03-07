"""State persistence using JSON/JSONL flat files."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gnc_enrich.domain.models import (
    AuditEntry,
    EvidencePacket,
    LineItem,
    EmailEvidence,
    Proposal,
    ReceiptEvidence,
    ReviewDecision,
    SkipRecord,
    Split,
    Transaction,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_serialize(v) for v in obj)
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _parse_decimal(v: Any) -> Decimal:
    if v is None:
        return Decimal(0)
    return Decimal(str(v))


def _parse_datetime(v: str | None) -> datetime | None:
    if not v:
        return None
    return datetime.fromisoformat(v)


def _parse_date(v: str) -> date:
    return date.fromisoformat(v)


def _parse_split(d: dict) -> Split:
    return Split(
        account_path=d["account_path"],
        amount=_parse_decimal(d["amount"]),
        memo=d.get("memo", ""),
    )


def _parse_line_item(d: dict) -> LineItem:
    return LineItem(
        description=d["description"],
        amount=_parse_decimal(d["amount"]),
        quantity=d.get("quantity", 1),
    )


def _parse_email_evidence(d: dict) -> EmailEvidence:
    return EmailEvidence(
        evidence_id=d["evidence_id"],
        message_id=d["message_id"],
        sender=d["sender"],
        subject=d["subject"],
        sent_at=datetime.fromisoformat(d["sent_at"]),
        body_snippet=d.get("body_snippet", ""),
        parsed_amounts=[_parse_decimal(a) for a in d.get("parsed_amounts", [])],
        relevance_score=float(d.get("relevance_score", 0.0)),
    )


def _parse_receipt_evidence(d: dict | None) -> ReceiptEvidence | None:
    if d is None:
        return None
    return ReceiptEvidence(
        evidence_id=d["evidence_id"],
        source_path=d["source_path"],
        ocr_text=d["ocr_text"],
        parsed_total=_parse_decimal(d["parsed_total"]) if d.get("parsed_total") else None,
        line_items=[_parse_line_item(li) for li in d.get("line_items", [])],
        relevance_score=float(d.get("relevance_score", 0.0)),
    )


def _parse_transaction(d: dict) -> Transaction:
    return Transaction(
        tx_id=d["tx_id"],
        posted_date=_parse_date(d["posted_date"]),
        description=d["description"],
        currency=d["currency"],
        amount=_parse_decimal(d["amount"]),
        splits=[_parse_split(s) for s in d.get("splits", [])],
        account_name=d.get("account_name", ""),
        original_category=d.get("original_category", ""),
    )


def _parse_evidence_packet(d: dict | None) -> EvidencePacket | None:
    if d is None:
        return None
    return EvidencePacket(
        tx_id=d["tx_id"],
        emails=[_parse_email_evidence(e) for e in d.get("emails", [])],
        receipt=_parse_receipt_evidence(d.get("receipt")),
        similar_transactions=[_parse_transaction(t) for t in d.get("similar_transactions", [])],
    )


def _parse_proposal(d: dict) -> Proposal:
    return Proposal(
        proposal_id=d["proposal_id"],
        tx_id=d["tx_id"],
        suggested_description=d["suggested_description"],
        suggested_splits=[_parse_split(s) for s in d["suggested_splits"]],
        confidence=float(d["confidence"]),
        rationale=d["rationale"],
        evidence=_parse_evidence_packet(d.get("evidence")),
    )


def _parse_decision(d: dict) -> ReviewDecision:
    return ReviewDecision(
        tx_id=d["tx_id"],
        action=d["action"],
        final_description=d["final_description"],
        final_splits=[_parse_split(s) for s in d["final_splits"]],
        reviewer_note=d.get("reviewer_note", ""),
        decided_at=_parse_datetime(d.get("decided_at")),
    )


def _parse_skip(d: dict) -> SkipRecord:
    return SkipRecord(
        tx_id=d["tx_id"],
        reason=d.get("reason", ""),
        skipped_at=_parse_datetime(d.get("skipped_at")),
    )


class StateRepository:
    """Persists proposals, decisions, skips, and audit data as JSON/JSONL files."""

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- file paths ----------------------------------------------------------

    @property
    def _proposals_path(self) -> Path:
        return self._dir / "proposals.json"

    @property
    def _decisions_path(self) -> Path:
        return self._dir / "decisions.jsonl"

    @property
    def _skips_path(self) -> Path:
        return self._dir / "skip_state.json"

    @property
    def _audit_path(self) -> Path:
        return self._dir / "audit_log.jsonl"

    @property
    def _feedback_path(self) -> Path:
        return self._dir / "feedback_events.jsonl"

    # -- proposals ------------------------------------------------------------

    def save_proposals(self, proposals: list[Proposal]) -> None:
        data = {
            "schema_version": _SCHEMA_VERSION,
            "proposals": [_serialize(p) for p in proposals],
        }
        self._proposals_path.write_text(
            json.dumps(data, cls=_Encoder, indent=2), encoding="utf-8"
        )

    def load_proposals(self) -> list[Proposal]:
        if not self._proposals_path.exists():
            return []
        raw = json.loads(self._proposals_path.read_text(encoding="utf-8"))
        return [_parse_proposal(p) for p in raw.get("proposals", [])]

    # -- decisions ------------------------------------------------------------

    def save_decision(self, decision: ReviewDecision) -> None:
        line = json.dumps(_serialize(decision), cls=_Encoder) + "\n"
        with self._decisions_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_decisions(self) -> list[ReviewDecision]:
        if not self._decisions_path.exists():
            return []
        results: list[ReviewDecision] = []
        for line in self._decisions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(_parse_decision(json.loads(line)))
        return results

    # -- skips ----------------------------------------------------------------

    def save_skip(self, skip: SkipRecord) -> None:
        skips = self._load_skip_records()
        skips = [s for s in skips if s.tx_id != skip.tx_id]
        skips.append(skip)
        data = {
            "schema_version": _SCHEMA_VERSION,
            "skips": [_serialize(s) for s in skips],
        }
        self._skips_path.write_text(
            json.dumps(data, cls=_Encoder, indent=2), encoding="utf-8"
        )

    def _load_skip_records(self) -> list[SkipRecord]:
        if not self._skips_path.exists():
            return []
        raw = json.loads(self._skips_path.read_text(encoding="utf-8"))
        return [_parse_skip(s) for s in raw.get("skips", [])]

    def load_skipped_ids(self) -> set[str]:
        return {s.tx_id for s in self._load_skip_records()}

    # -- audit ----------------------------------------------------------------

    def append_audit(self, entry: AuditEntry) -> None:
        line = json.dumps(_serialize(entry), cls=_Encoder) + "\n"
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_audit_log(self) -> list[AuditEntry]:
        if not self._audit_path.exists():
            return []
        results: list[AuditEntry] = []
        for line in self._audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                results.append(AuditEntry(
                    entry_id=d["entry_id"],
                    tx_id=d["tx_id"],
                    action=d["action"],
                    proposed_description=d["proposed_description"],
                    proposed_splits=[_parse_split(s) for s in d["proposed_splits"]],
                    final_description=d["final_description"],
                    final_splits=[_parse_split(s) for s in d["final_splits"]],
                    confidence=float(d["confidence"]),
                    evidence_ids=d.get("evidence_ids", []),
                    timestamp=_parse_datetime(d.get("timestamp")),
                ))
        return results

    # -- feedback -------------------------------------------------------------

    def append_feedback(self, feedback: dict) -> None:
        line = json.dumps(feedback, cls=_Encoder) + "\n"
        with self._feedback_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_feedback(self) -> list[dict]:
        if not self._feedback_path.exists():
            return []
        results = []
        for line in self._feedback_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
        return results

    # -- metadata helpers -----------------------------------------------------

    def save_metadata(self, key: str, data: dict) -> None:
        """Save arbitrary keyed metadata (e.g. gnucash_path, run_config)."""
        path = self._dir / f"{key}.json"
        wrapped = {"schema_version": _SCHEMA_VERSION, **data}
        path.write_text(json.dumps(wrapped, cls=_Encoder, indent=2), encoding="utf-8")

    def load_metadata(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
