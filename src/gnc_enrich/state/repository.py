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
        full_body=d.get("full_body", ""),
        filtered_body=d.get("filtered_body", ""),
        amount_context=d.get("amount_context", ""),
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
        is_transfer=d.get("is_transfer", False),
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
    tx_date_raw = d.get("tx_date")
    return Proposal(
        proposal_id=d["proposal_id"],
        tx_id=d["tx_id"],
        suggested_description=d["suggested_description"],
        suggested_splits=[_parse_split(s) for s in d["suggested_splits"]],
        confidence=float(d["confidence"]),
        rationale=d["rationale"],
        evidence=_parse_evidence_packet(d.get("evidence")),
        tx_date=_parse_date(tx_date_raw) if tx_date_raw else None,
        tx_amount=_parse_decimal(d["tx_amount"]) if d.get("tx_amount") is not None else None,
        original_description=d.get("original_description", ""),
        original_splits=[_parse_split(s) for s in d.get("original_splits", [])],
        confidence_breakdown=d.get("confidence_breakdown", []),
        is_transfer=d.get("is_transfer", False),
    )


def _parse_decision(d: dict) -> ReviewDecision:
    return ReviewDecision(
        tx_id=d["tx_id"],
        action=d["action"],
        final_description=d["final_description"],
        final_splits=[_parse_split(s) for s in d["final_splits"]],
        reviewer_note=d.get("reviewer_note", ""),
        decided_at=_parse_datetime(d.get("decided_at")),
        approved_email_ids=d.get("approved_email_ids", []),
        approved_receipt=d.get("approved_receipt", False),
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

    @property
    def state_dir(self) -> Path:
        """Public accessor for the state directory path."""
        return self._dir

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
        """Persist all proposals to proposals.json."""
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
        try:
            raw = json.loads(self._proposals_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt proposals file %s; returning empty list", self._proposals_path)
            return []
        return [_parse_proposal(p) for p in raw.get("proposals", [])]

    # -- decisions ------------------------------------------------------------

    def save_decision(self, decision: ReviewDecision) -> None:
        """Append a review decision to decisions.jsonl."""
        self._ensure_jsonl_header(self._decisions_path)
        line = json.dumps(_serialize(decision), cls=_Encoder) + "\n"
        with self._decisions_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_decisions(self) -> list[ReviewDecision]:
        """Load all review decisions from decisions.jsonl."""
        if not self._decisions_path.exists():
            return []
        results: list[ReviewDecision] = []
        for lineno, line in enumerate(self._decisions_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line %d in %s", lineno, self._decisions_path)
                continue
            if "_schema_version" in d:
                continue
            results.append(_parse_decision(d))
        return results

    # -- skips ----------------------------------------------------------------

    def save_skip(self, skip: SkipRecord) -> None:
        """Persist a skip record, replacing any prior skip for the same tx."""
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
        """Return the set of transaction IDs that have been skipped."""
        return {s.tx_id for s in self._load_skip_records()}

    # -- audit ----------------------------------------------------------------

    def append_audit(self, entry: AuditEntry) -> None:
        self._ensure_jsonl_header(self._audit_path)
        line = json.dumps(_serialize(entry), cls=_Encoder) + "\n"
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_audit_log(self) -> list[AuditEntry]:
        """Load all audit entries from audit_log.jsonl."""
        if not self._audit_path.exists():
            return []
        results: list[AuditEntry] = []
        for lineno, line in enumerate(self._audit_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line %d in %s", lineno, self._audit_path)
                continue
            if "_schema_version" in d:
                continue
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
        self._ensure_jsonl_header(self._feedback_path)
        line = json.dumps(feedback, cls=_Encoder) + "\n"
        with self._feedback_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_feedback(self) -> list[dict]:
        """Load all feedback events from feedback_events.jsonl."""
        if not self._feedback_path.exists():
            return []
        results = []
        for lineno, line in enumerate(self._feedback_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line %d in %s", lineno, self._feedback_path)
                continue
            if "_schema_version" in d:
                continue
            results.append(d)
        return results

    # -- metadata helpers -----------------------------------------------------

    def _ensure_jsonl_header(self, path: Path) -> None:
        """Write a schema version header as the first line if the file is new."""
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"_schema_version": _SCHEMA_VERSION}) + "\n")

    def save_metadata(self, key: str, data: dict) -> None:
        """Save arbitrary keyed metadata (e.g. gnucash_path, run_config)."""
        path = self._dir / f"{key}.json"
        wrapped = {"schema_version": _SCHEMA_VERSION, **data}
        path.write_text(json.dumps(wrapped, cls=_Encoder, indent=2), encoding="utf-8")

    def load_metadata(self, key: str) -> dict | None:
        """Load keyed metadata JSON (e.g. run_config)."""
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
