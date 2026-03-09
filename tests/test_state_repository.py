"""Tests for JSON/JSONL state persistence."""

import json
from datetime import date, datetime
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
        tx_date=date(2025, 1, 1),
        tx_amount=Decimal("15.00"),
        original_description="Card Payment TESCO",
        original_splits=[
            Split(account_path="Imbalance-GBP", amount=Decimal("15.00"), memo="orig memo"),
        ],
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


def test_proposal_original_fields_roundtrip(tmp_path) -> None:
    repo = StateRepository(tmp_path)
    prop = _sample_proposal()
    repo.save_proposals([prop])

    loaded = repo.load_proposals()[0]
    assert loaded.tx_date == date(2025, 1, 1)
    assert loaded.tx_amount == Decimal("15.00")
    assert loaded.original_description == "Card Payment TESCO"
    assert len(loaded.original_splits) == 1
    assert loaded.original_splits[0].account_path == "Imbalance-GBP"
    assert loaded.original_splits[0].memo == "orig memo"


def test_proposal_none_fields_roundtrip(tmp_path) -> None:
    """Proposals with None tx_date/tx_amount must survive serialization."""
    repo = StateRepository(tmp_path)
    prop = Proposal(
        proposal_id="pn",
        tx_id="txn",
        suggested_description="Unknown",
        suggested_splits=[Split(account_path="Expenses:Misc", amount=Decimal("5.00"))],
        confidence=0.5,
        rationale="test",
        tx_date=None,
        tx_amount=None,
        original_description="",
        original_splits=[],
    )
    repo.save_proposals([prop])
    loaded = repo.load_proposals()[0]
    assert loaded.tx_date is None
    assert loaded.tx_amount is None
    assert loaded.original_description == ""
    assert loaded.original_splits == []


def test_proposal_extraction_result_items_sanitized_on_load(tmp_path) -> None:
    """Loading proposals with extraction_result.items containing non-dicts sanitizes to dicts only."""
    import json

    repo = StateRepository(tmp_path)
    # Simulate legacy state: extraction_result with a list item (invalid)
    raw = {
        "schema_version": 1,
        "proposals": [
            {
                "proposal_id": "p1",
                "tx_id": "tx1",
                "suggested_description": "Test",
                "suggested_splits": [{"account_path": "Expenses:Food", "amount": "10", "memo": ""}],
                "confidence": 0.8,
                "rationale": "test",
                "tx_date": "2025-01-01",
                "tx_amount": "10",
                "original_description": "",
                "original_splits": [],
                "extraction_result": {
                    "seller_name": "Shop",
                    "items": [
                        {"description": "Item A", "amount": "5"},
                        ["list", "not", "dict"],
                        {"description": "Item B", "amount": "5"},
                    ],
                },
            }
        ],
    }
    (tmp_path / "proposals.json").write_text(json.dumps(raw), encoding="utf-8")
    loaded = repo.load_proposals()
    assert len(loaded) == 1
    assert loaded[0].extraction_result is not None
    items = loaded[0].extraction_result.get("items") or []
    assert len(items) == 2
    assert all(isinstance(x, dict) for x in items)
    assert items[0].get("description") == "Item A"
    assert items[1].get("description") == "Item B"


def test_corrupt_proposals_json_returns_empty(tmp_path) -> None:
    """Corrupt proposals.json should not crash; returns empty list."""
    repo = StateRepository(tmp_path)
    (tmp_path / "proposals.json").write_text("{invalid json", encoding="utf-8")
    loaded = repo.load_proposals()
    assert loaded == []


def test_load_proposals_skips_invalid_entries(tmp_path) -> None:
    """load_proposals skips proposals with missing/invalid keys and returns the rest."""
    repo = StateRepository(tmp_path)
    valid1 = {
        "proposal_id": "p1",
        "tx_id": "tx1",
        "suggested_description": "A",
        "suggested_splits": [{"account_path": "Expenses:Food", "amount": "10", "memo": ""}],
        "confidence": 0.8,
        "rationale": "r",
        "tx_date": "2025-01-01",
        "tx_amount": "10",
        "original_description": "",
        "original_splits": [],
    }
    invalid = {"proposal_id": "p2"}  # missing required keys
    valid3 = {
        "proposal_id": "p3",
        "tx_id": "tx3",
        "suggested_description": "B",
        "suggested_splits": [{"account_path": "Expenses:Misc", "amount": "5", "memo": ""}],
        "confidence": 0.5,
        "rationale": "r2",
        "tx_date": "2025-01-02",
        "tx_amount": "5",
        "original_description": "",
        "original_splits": [],
    }
    data = {"schema_version": 1, "proposals": [valid1, invalid, valid3]}
    (tmp_path / "proposals.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = repo.load_proposals()
    assert len(loaded) == 2
    assert loaded[0].proposal_id == "p1"
    assert loaded[1].proposal_id == "p3"


def test_load_metadata_returns_none_for_corrupt_file(tmp_path) -> None:
    """load_metadata returns None when the JSON file is corrupt."""
    repo = StateRepository(tmp_path)
    (tmp_path / "run_config.json").write_text("not valid json", encoding="utf-8")
    assert repo.load_metadata("run_config") is None


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


def test_load_feedback_skips_corrupt_and_header_lines(tmp_path) -> None:
    """load_feedback ignores schema header and corrupt lines; returns remaining events."""
    repo = StateRepository(tmp_path)
    repo._ensure_jsonl_header(repo._feedback_path)
    with repo._feedback_path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write('{"proposal_id": "p1", "accepted": true}\n')
        f.write('{"proposal_id": "p2"}\n')
    loaded = repo.load_feedback()
    assert len(loaded) == 2
    assert loaded[0]["proposal_id"] == "p1"


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


def test_load_decisions_skips_corrupt_and_invalid_lines(tmp_path) -> None:
    """load_decisions skips unparseable lines and lines with missing/invalid keys; returns the rest."""
    repo = StateRepository(tmp_path)
    repo._ensure_jsonl_header(repo._decisions_path)
    with repo._decisions_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"_schema_version": 1}) + "\n")
        f.write("{invalid json\n")
        f.write(
            json.dumps(
                {
                    "tx_id": "tx1",
                    "action": "approve",
                    "final_description": "Done",
                    "final_splits": [{"account_path": "Expenses:Food", "amount": "10", "memo": ""}],
                    "reviewer_note": "",
                    "decided_at": "2025-01-01T12:00:00",
                    "approved_email_ids": [],
                    "approved_receipt": False,
                }
            )
            + "\n"
        )
        f.write(json.dumps({"tx_id": "tx2"}) + "\n")  # missing required keys
    loaded = repo.load_decisions()
    assert len(loaded) == 1
    assert loaded[0].tx_id == "tx1"


def test_load_skipped_ids_returns_empty_when_skip_state_corrupt(tmp_path) -> None:
    """load_skipped_ids returns empty set when skip_state.json is corrupt (graceful, non-fatal)."""
    repo = StateRepository(tmp_path)
    (tmp_path / "skip_state.json").write_text("not valid json", encoding="utf-8")
    assert repo.load_skipped_ids() == set()


def test_load_skipped_ids_skips_invalid_skip_record(tmp_path) -> None:
    """When skip_state has one invalid skip record (missing required tx_id), the valid one is still loaded."""
    repo = StateRepository(tmp_path)
    data = {
        "schema_version": 1,
        "skips": [
            {"tx_id": "tx1", "reason": "Skipped", "skipped_at": "2025-01-01T12:00:00"},
            {"reason": "no tx_id"},  # missing required tx_id -> KeyError
        ],
    }
    (tmp_path / "skip_state.json").write_text(json.dumps(data), encoding="utf-8")
    ids = repo.load_skipped_ids()
    assert ids == {"tx1"}


def test_load_audit_log_skips_corrupt_and_malformed_lines(tmp_path) -> None:
    """load_audit_log skips unparseable lines and lines with missing keys; returns the rest."""
    repo = StateRepository(tmp_path)
    repo._ensure_jsonl_header(repo._audit_path)
    valid_entry = {
        "entry_id": "e1",
        "tx_id": "tx1",
        "action": "approve",
        "proposed_description": "P",
        "proposed_splits": [{"account_path": "Expenses:Food", "amount": "10", "memo": ""}],
        "final_description": "F",
        "final_splits": [{"account_path": "Expenses:Food", "amount": "10", "memo": ""}],
        "confidence": 0.9,
        "evidence_ids": [],
        "timestamp": "2025-01-01T12:00:00",
    }
    with repo._audit_path.open("a", encoding="utf-8") as f:
        f.write("{bad json\n")
        f.write(json.dumps(valid_entry) + "\n")
        f.write(json.dumps({"entry_id": "e2"}) + "\n")  # missing required keys
    loaded = repo.load_audit_log()
    assert len(loaded) == 1
    assert loaded[0].entry_id == "e1"
