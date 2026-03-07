"""Domain entities used by orchestration and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class ReviewAction(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    SKIP = "skip"


@dataclass(slots=True)
class Account:
    account_id: str
    name: str
    full_path: str
    account_type: str
    currency: str
    parent_id: str | None = None


@dataclass(slots=True)
class Split:
    account_path: str
    amount: Decimal
    memo: str = ""


@dataclass(slots=True)
class Transaction:
    tx_id: str
    posted_date: date
    description: str
    currency: str
    amount: Decimal
    splits: list[Split] = field(default_factory=list)
    account_name: str = ""
    original_category: str = ""


@dataclass(slots=True)
class LineItem:
    description: str
    amount: Decimal
    quantity: int = 1


@dataclass(slots=True)
class EmailEvidence:
    evidence_id: str
    message_id: str
    sender: str
    subject: str
    sent_at: datetime
    body_snippet: str = ""
    full_body: str = ""
    parsed_amounts: list[Decimal] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass(slots=True)
class ReceiptEvidence:
    evidence_id: str
    source_path: str
    ocr_text: str
    parsed_total: Decimal | None = None
    line_items: list[LineItem] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass(slots=True)
class EvidencePacket:
    """Aggregated evidence for a single transaction."""
    tx_id: str
    emails: list[EmailEvidence] = field(default_factory=list)
    receipt: ReceiptEvidence | None = None
    similar_transactions: list[Transaction] = field(default_factory=list)


@dataclass(slots=True)
class Proposal:
    proposal_id: str
    tx_id: str
    suggested_description: str
    suggested_splits: list[Split]
    confidence: float
    rationale: str
    evidence: EvidencePacket | None = None


@dataclass(slots=True)
class ReviewDecision:
    tx_id: str
    action: str  # approve|edit|skip
    final_description: str
    final_splits: list[Split]
    reviewer_note: str = ""
    decided_at: datetime | None = None
    approved_email_ids: list[str] = field(default_factory=list)
    approved_receipt: bool = False


@dataclass(slots=True)
class SkipRecord:
    tx_id: str
    reason: str = ""
    skipped_at: datetime | None = None


@dataclass(slots=True)
class AuditEntry:
    entry_id: str
    tx_id: str
    action: str
    proposed_description: str
    proposed_splits: list[Split]
    final_description: str
    final_splits: list[Split]
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
