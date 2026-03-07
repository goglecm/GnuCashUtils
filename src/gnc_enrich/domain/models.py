"""Domain entities used by orchestration and adapters."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


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


@dataclass(slots=True)
class EmailEvidence:
    evidence_id: str
    message_id: str
    sender: str
    subject: str
    sent_at: datetime
    parsed_amounts: list[Decimal] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass(slots=True)
class ReceiptEvidence:
    evidence_id: str
    source_path: str
    ocr_text: str
    parsed_total: Decimal | None = None
    relevance_score: float = 0.0


@dataclass(slots=True)
class Proposal:
    proposal_id: str
    tx_id: str
    suggested_description: str
    suggested_splits: list[Split]
    confidence: float
    rationale: str


@dataclass(slots=True)
class ReviewDecision:
    tx_id: str
    action: str  # approve|edit|skip
    final_description: str
    final_splits: list[Split]
    reviewer_note: str = ""
