"""Tests for JSON/JSONL state persistence."""

from datetime import datetime
from decimal import Decimal

from gnc_enrich.domain.models import (
    AuditEntry,
    EvidencePacket,
    EmailEvidence,
    Proposal,
    ReviewDecision,
    SkipRecord,
    Split,
)
from gnc_enrich.state.repository import StateRepository


def _sample_proposal(pid: str = "p1", tx_id: str = "tx1") -> Proposal:
    return Proposal(
        proposal_id=pid,
        tx_id=tx_id,
        suggested_description="Groceries 01/01/2025",
        suggested_splits=[
            Split(account_path="Expenses:Food", amount=Decimal("15.00"), memo="lunch"),
        ],
        confidence=0.82,
        rationale="Matched email from tesco@example.com",
        evidence=EvidencePacket(
            tx_id=tx_id,
            emails=[
                EmailEvidence(
                    evidence_id="e1",
                    message_id="<msg@tesco.com>",
                    sender="tesco@example.com",
                    subject="Your receipt",
                    sent_at=datetime(2025, 1, 2, 14, 30),
                    parsed_amounts=[Decimal("15.00")],
                    relevance_score=0.9,
                )
            ],
        ),
    )


def _sample_decision(tx_id: str = "tx1", action: str = "approve") -> ReviewDecision:
    return ReviewDecision(
        tx_id=tx_id,
        action=action,
        final_description="Groceries 01/01/2025",
        final_splits=[Split(account_path="Expenses:Food", amount=Decimal("15.00"))],
        reviewer_note="looks correct",
        decided_at=datetime(2025, 6, 1, 10, 0),
    )


# -- proposals ----------------------------------------------------------------


def test_save_and_load_proposals(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    proposals = [_sample_proposal("p1", "tx1"), _sample_proposal("p2", "tx2")]
    repo.save_proposals(proposals)

    loaded = repo.load_proposals()
    assert len(loaded) == 2
    assert loaded[0].proposal_id == "p1"
    assert loaded[0].confidence == 0.82
    assert loaded[0].suggested_splits[0].amount == Decimal("15.00")
    assert loaded[1].tx_id == "tx2"


def test_load_proposals_empty_dir(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    assert repo.load_proposals() == []


def test_proposals_evidence_roundtrip(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    prop = _sample_proposal()
    repo.save_proposals([prop])

    loaded = repo.load_proposals()[0]
    assert loaded.evidence is not None
    assert len(loaded.evidence.emails) == 1
    assert loaded.evidence.emails[0].sender == "tesco@example.com"
    assert loaded.evidence.emails[0].parsed_amounts == [Decimal("15.00")]


# -- decisions ----------------------------------------------------------------


def test_save_and_load_decisions(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    repo.save_decision(_sample_decision("tx1"))
    repo.save_decision(_sample_decision("tx2", action="edit"))

    loaded = repo.load_decisions()
    assert len(loaded) == 2
    assert loaded[0].tx_id == "tx1"
    assert loaded[0].action == "approve"
    assert loaded[1].action == "edit"


def test_load_decisions_empty(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    assert repo.load_decisions() == []


def test_decision_datetime_roundtrip(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    dec = _sample_decision()
    repo.save_decision(dec)
    loaded = repo.load_decisions()[0]
    assert loaded.decided_at == datetime(2025, 6, 1, 10, 0)


# -- skips --------------------------------------------------------------------


def test_save_and_load_skips(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    repo.save_skip(SkipRecord(tx_id="tx5", reason="Unclear"))
    repo.save_skip(SkipRecord(tx_id="tx6", reason="No evidence"))

    ids = repo.load_skipped_ids()
    assert ids == {"tx5", "tx6"}


def test_skip_deduplicates(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    repo.save_skip(SkipRecord(tx_id="tx5", reason="first"))
    repo.save_skip(SkipRecord(tx_id="tx5", reason="updated"))

    records = repo._load_skip_records()
    assert len(records) == 1
    assert records[0].reason == "updated"


def test_load_skipped_empty(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    assert repo.load_skipped_ids() == set()


# -- audit --------------------------------------------------------------------


def test_append_and_load_audit(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    entry = AuditEntry(
        entry_id="a1",
        tx_id="tx1",
        action="approve",
        proposed_description="Groceries",
        proposed_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
        final_description="Groceries 01/01/2025",
        final_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
        confidence=0.9,
        evidence_ids=["e1"],
        timestamp=datetime(2025, 6, 1, 12, 0),
    )
    repo.append_audit(entry)
    repo.append_audit(entry)

    loaded = repo.load_audit_log()
    assert len(loaded) == 2
    assert loaded[0].entry_id == "a1"
    assert loaded[0].confidence == 0.9


# -- feedback -----------------------------------------------------------------


def test_append_and_load_feedback(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    repo.append_feedback({"proposal_id": "p1", "accepted": True})
    repo.append_feedback({"proposal_id": "p2", "accepted": False, "note": "wrong"})

    loaded = repo.load_feedback()
    assert len(loaded) == 2
    assert loaded[1]["note"] == "wrong"


# -- metadata -----------------------------------------------------------------


def test_save_and_load_metadata(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    repo.save_metadata("run_config", {"gnucash_path": "/tmp/book.gnucash"})

    loaded = repo.load_metadata("run_config")
    assert loaded is not None
    assert loaded["gnucash_path"] == "/tmp/book.gnucash"
    assert loaded["schema_version"] == 1


def test_load_metadata_missing(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    assert repo.load_metadata("nonexistent") is None
