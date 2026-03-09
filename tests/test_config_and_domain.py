from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from gnc_enrich.config import ApplyConfig, LlmConfig, LlmMode, ReviewConfig, RunConfig
from gnc_enrich.domain.models import (
    Account,
    AuditEntry,
    EmailEvidence,
    EvidencePacket,
    LineItem,
    Proposal,
    ReceiptEvidence,
    ReviewAction,
    ReviewDecision,
    SkipRecord,
    Split,
    Transaction,
)


def test_config_defaults() -> None:
    run = RunConfig(
        gnucash_path=Path("book.gnucash"),
        emails_dir=Path("emails"),
        receipts_dir=Path("receipts"),
        processed_receipts_dir=Path("receipts_done"),
        state_dir=Path("state"),
    )
    assert run.date_window_days == 7
    assert run.amount_tolerance == 0.50
    assert run.include_skipped is False
    assert run.use_llm_during_run is False
    assert run.llm.mode == LlmMode.DISABLED

    review = ReviewConfig(state_dir=Path("state"))
    assert review.host == "127.0.0.1"
    assert review.port == 7860

    apply_cfg = ApplyConfig(state_dir=Path("state"))
    assert apply_cfg.create_backup is True
    assert apply_cfg.dry_run is False


def test_llm_config() -> None:
    llm = LlmConfig(mode=LlmMode.ONLINE, endpoint="http://localhost:11434", model_name="llama3")
    assert llm.mode == LlmMode.ONLINE
    assert llm.temperature == 0.2
    assert llm.max_tokens == 1024


def test_domain_dataclass_roundtrip() -> None:
    split = Split(account_path="Expenses:Food", amount=Decimal("10.00"), memo="Lunch")
    tx = Transaction(
        tx_id="abc",
        posted_date=date(2025, 1, 1),
        description="Shop",
        currency="GBP",
        amount=Decimal("10.00"),
        splits=[split],
        account_name="Current Account",
        original_category="Unspecified",
    )
    assert tx.splits[0].account_path == "Expenses:Food"
    assert tx.account_name == "Current Account"
    assert tx.original_category == "Unspecified"


def test_account_model() -> None:
    acct = Account(
        account_id="a1",
        name="Food",
        full_path="Expenses:Food",
        account_type="EXPENSE",
        currency="GBP",
    )
    assert acct.full_path == "Expenses:Food"
    assert acct.parent_id is None


def test_line_item_defaults() -> None:
    item = LineItem(description="Coffee", amount=Decimal("3.50"))
    assert item.quantity == 1


def test_receipt_evidence_with_line_items() -> None:
    items = [
        LineItem(description="Milk", amount=Decimal("1.50")),
        LineItem(description="Bread", amount=Decimal("2.00")),
    ]
    receipt = ReceiptEvidence(
        evidence_id="r1",
        source_path="/tmp/receipt.jpg",
        ocr_text="Milk 1.50\nBread 2.00\nTotal 3.50",
        parsed_total=Decimal("3.50"),
        line_items=items,
    )
    assert len(receipt.line_items) == 2
    assert receipt.parsed_total == Decimal("3.50")


def test_evidence_packet() -> None:
    packet = EvidencePacket(
        tx_id="tx1",
        emails=[
            EmailEvidence(
                evidence_id="e1",
                message_id="<msg1@example.com>",
                sender="shop@example.com",
                subject="Order confirmation",
                sent_at=datetime(2025, 1, 1, 12, 0),
            )
        ],
    )
    assert len(packet.emails) == 1
    assert packet.receipt is None


def test_proposal_with_evidence() -> None:
    prop = Proposal(
        proposal_id="p1",
        tx_id="tx1",
        suggested_description="Groceries 01/01/2025",
        suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
        confidence=0.85,
        rationale="Matched email from shop@example.com",
        evidence=EvidencePacket(tx_id="tx1"),
    )
    assert prop.evidence is not None
    assert prop.confidence == 0.85


def test_review_decision_with_timestamp() -> None:
    now = datetime(2025, 6, 1, 10, 30)
    dec = ReviewDecision(
        tx_id="tx1",
        action=ReviewAction.APPROVE,
        final_description="Groceries",
        final_splits=[],
        decided_at=now,
    )
    assert dec.decided_at == now


def test_skip_record() -> None:
    skip = SkipRecord(tx_id="tx99", reason="Unclear evidence")
    assert skip.skipped_at is None


def test_audit_entry() -> None:
    entry = AuditEntry(
        entry_id="audit-1",
        tx_id="tx1",
        action="approve",
        proposed_description="Groceries",
        proposed_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
        final_description="Groceries 01/01/2025",
        final_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
        confidence=0.9,
        evidence_ids=["e1", "r1"],
    )
    assert entry.timestamp is None
    assert len(entry.evidence_ids) == 2


def test_review_action_enum() -> None:
    assert ReviewAction.APPROVE == "approve"
    assert ReviewAction.SKIP == "skip"
    assert ReviewAction.EDIT == "edit"


def test_review_action_validate_accepts_and_rejects() -> None:
    assert ReviewAction.validate("approve") == "approve"
    assert ReviewAction.validate("edit") == "edit"
    assert ReviewAction.validate("skip") == "skip"
    with pytest.raises(ValueError, match="Invalid review action"):
        ReviewAction.validate("invalid")
