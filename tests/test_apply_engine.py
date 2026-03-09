"""Tests for the apply engine: dry-run, apply, backup, rollback, audit."""

import gzip
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from gnc_enrich.apply.engine import ApplyEngine
from gnc_enrich.domain.models import (
    EvidencePacket,
    Proposal,
    ReceiptEvidence,
    ReviewDecision,
    Split,
)
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.state.repository import StateRepository
from tests.conftest import SAMPLE_GNUCASH_XML


def _setup_state(tmp_path: Path) -> tuple[Path, Path]:
    """Set up a GnuCash file and state dir with proposals + decisions."""
    gnucash = tmp_path / "book.gnucash"
    with gzip.open(gnucash, "wb") as f:
        f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))

    state_dir = tmp_path / "state"
    state = StateRepository(state_dir)

    state.save_metadata(
        "run_config",
        {
            "gnucash_path": str(gnucash),
            "processed_receipts_dir": str(tmp_path / "processed"),
        },
    )

    proposals = [
        Proposal(
            proposal_id="p1",
            tx_id="tx_unspec1",
            suggested_description="Card Payment - Tesco 15/01/2025",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.85,
            rationale="ML + email match",
            tx_date=date(2025, 1, 15),
            tx_amount=Decimal("25.00"),
            original_description="Card Payment",
            original_splits=[Split(account_path="Unspecified", amount=Decimal("25.00"))],
        ),
        Proposal(
            proposal_id="p2",
            tx_id="tx_unspec2",
            suggested_description="Direct Debit 01/02/2025",
            suggested_splits=[Split(account_path="Expenses:Utilities", amount=Decimal("9.50"))],
            confidence=0.6,
            rationale="Heuristic",
            tx_date=date(2025, 2, 1),
            tx_amount=Decimal("9.50"),
            original_description="Direct Debit",
            original_splits=[Split(account_path="Unspecified", amount=Decimal("9.50"))],
        ),
        Proposal(
            proposal_id="p3",
            tx_id="tx_imbalance1",
            suggested_description="POS Transaction 20/01/2025",
            suggested_splits=[
                Split(account_path="Expenses:Miscellaneous", amount=Decimal("32.00"))
            ],
            confidence=0.4,
            rationale="Low confidence",
            tx_date=date(2025, 1, 20),
            tx_amount=Decimal("32.00"),
            original_description="POS Transaction",
            original_splits=[Split(account_path="Imbalance-GBP", amount=Decimal("32.00"))],
        ),
        Proposal(
            proposal_id="p4",
            tx_id="tx_transfer",
            suggested_description="Transfer to Savings 25/01/2025",
            suggested_splits=[
                Split(account_path="Current Account", amount=Decimal("-500.00")),
                Split(account_path="Savings Account", amount=Decimal("500.00")),
            ],
            confidence=1.0,
            rationale="Transfer between own accounts",
            tx_date=date(2025, 1, 25),
            tx_amount=Decimal("500.00"),
            original_description="Transfer to Savings",
            original_splits=[
                Split(account_path="Current Account", amount=Decimal("-500.00")),
                Split(account_path="Savings Account", amount=Decimal("500.00")),
            ],
            is_transfer=True,
        ),
    ]
    state.save_proposals(proposals)

    state.save_decision(
        ReviewDecision(
            tx_id="tx_unspec1",
            action="approve",
            final_description="Tesco Groceries 15/01/2025",
            final_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            decided_at=datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
        )
    )
    state.save_decision(
        ReviewDecision(
            tx_id="tx_unspec2",
            action="edit",
            final_description="BT Broadband 01/02/2025",
            final_splits=[Split(account_path="Expenses:Utilities", amount=Decimal("9.50"))],
        )
    )
    state.save_decision(
        ReviewDecision(
            tx_id="tx_imbalance1",
            action="skip",
            final_description="",
            final_splits=[],
        )
    )
    state.save_decision(
        ReviewDecision(
            tx_id="tx_transfer",
            action="approve",
            final_description="Transfer to Savings 25/01/2025",
            final_splits=[
                Split(account_path="Expenses:Miscellaneous", amount=Decimal("500.00")),
            ],
            decided_at=datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
        )
    )

    return gnucash, state_dir


class TestDryRun:

    def test_generates_report_file(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        report = engine.generate_dry_run_report(state_dir)
        assert report.exists()
        assert report.name == "dry_run_report.txt"

    def test_report_contains_summary(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        report = engine.generate_dry_run_report(state_dir)
        content = report.read_text()
        assert "approve" in content.lower()
        assert "skip" in content.lower()
        assert "Summary" in content

    def test_report_lists_all_decisions(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        report = engine.generate_dry_run_report(state_dir)
        content = report.read_text()
        assert "tx_unspec1" in content
        assert "tx_unspec2" in content
        assert "tx_imbalance1" in content

    def test_dry_run_report_with_no_decisions(self, tmp_path: Path) -> None:
        """generate_dry_run_report with no decisions still writes report (0 approved, 0 skipped)."""
        gnucash = tmp_path / "book.gnucash"
        with gzip.open(gnucash, "wb") as f:
            f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(gnucash)})
        state.save_proposals([])
        report = ApplyEngine().generate_dry_run_report(state_dir)
        content = report.read_text()
        assert "DRY-RUN REPORT" in content
        assert "0 approved" in content or "Summary" in content


class TestApply:

    def test_apply_modifies_gnucash_file(self, tmp_path: Path) -> None:
        gnucash, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        loader = GnuCashLoader()
        txs = loader.load_transactions(gnucash)
        tx_map = {t.tx_id: t for t in txs}
        assert tx_map["tx_unspec1"].description == "Tesco Groceries 15/01/2025"
        assert tx_map["tx_unspec2"].description == "BT Broadband 01/02/2025"

    def test_apply_creates_backup(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        backup_dir = state_dir / "backups"
        assert backup_dir.exists()
        backups = list(backup_dir.glob("*.gnucash"))
        assert len(backups) >= 1

    def test_apply_writes_journal(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        journal_path = state_dir / "apply_journal.jsonl"
        assert journal_path.exists()
        entries = [
            json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()
        ]
        data_entries = [e for e in entries if "_schema_version" not in e]
        assert len(data_entries) == 3  # approved + edited + transfer
        tx_ids = {e["tx_id"] for e in data_entries}
        assert "tx_unspec1" in tx_ids
        assert "tx_unspec2" in tx_ids
        assert "tx_transfer" in tx_ids
        assert "tx_imbalance1" not in tx_ids  # skipped

    def test_apply_writes_audit_log(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        state = StateRepository(state_dir)
        audit = state.load_audit_log()
        assert len(audit) == 3
        assert audit[0].action in ("approve", "edit")

    def test_apply_transfer_updates_splits_like_other_transactions(self, tmp_path: Path) -> None:
        """Transfers with proposals/decisions are treated like any other transaction: description and splits follow the decision."""
        gnucash, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        loader = GnuCashLoader()
        txs = loader.load_transactions(gnucash)
        tx_map = {t.tx_id: t for t in txs}
        transfer = tx_map["tx_transfer"]
        assert transfer.description == "Transfer to Savings 25/01/2025"
        # Decision in _setup_state sets a single Expenses:Miscellaneous split; writer rewrites splits accordingly.
        split_paths = {sp.account_path for sp in transfer.splits}
        assert "Expenses:Miscellaneous" in split_paths

    def test_journal_stores_original_description(self, tmp_path: Path) -> None:
        _, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        journal_path = state_dir / "apply_journal.jsonl"
        entries = [
            json.loads(line)
            for line in journal_path.read_text().splitlines()
            if line.strip() and '"_schema_version"' not in line
        ]
        tx1_entry = next(e for e in entries if e["tx_id"] == "tx_unspec1")
        assert tx1_entry["original_description"] == "Card Payment"

    def test_apply_skips_when_no_decisions(self, tmp_path: Path) -> None:
        gnucash = tmp_path / "book.gnucash"
        with gzip.open(gnucash, "wb") as f:
            f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))

        state_dir = tmp_path / "state"
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(gnucash)})
        state.save_proposals([])

        engine = ApplyEngine()
        engine.apply(state_dir)

    def test_apply_updates_split_accounts(self, tmp_path: Path) -> None:
        """Writer must re-point target splits to the approved category."""
        gnucash, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir)

        loader = GnuCashLoader()
        txs = loader.load_transactions(gnucash)
        tx_map = {t.tx_id: t for t in txs}
        food_splits = [s for s in tx_map["tx_unspec1"].splits if s.account_path == "Expenses:Food"]
        assert len(food_splits) == 1
        assert food_splits[0].amount == Decimal("25.00")

    def test_apply_no_backup_flag(self, tmp_path: Path) -> None:
        gnucash, state_dir = _setup_state(tmp_path)
        engine = ApplyEngine()
        engine.apply(state_dir, create_backup=False)
        backup_dir = state_dir / "backups"
        assert not backup_dir.exists()

    def test_apply_missing_proposal_only_updates_description(self, tmp_path: Path) -> None:
        """When a decision has no matching proposal, only description is applied (splits unchanged)."""
        gnucash = tmp_path / "book.gnucash"
        with gzip.open(gnucash, "wb") as f:
            f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))

        state_dir = tmp_path / "state"
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(gnucash)})
        state.save_proposals([])

        state.save_decision(
            ReviewDecision(
                tx_id="tx_unspec1",
                action="approve",
                final_description="Updated without proposal",
                final_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
                decided_at=datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
            )
        )

        engine = ApplyEngine()
        engine.apply(state_dir)

        loader = GnuCashLoader()
        txs = loader.load_transactions(gnucash)
        tx = next(t for t in txs if t.tx_id == "tx_unspec1")
        assert tx.description == "Updated without proposal"
        assert any(s.account_path == "Unspecified" for s in tx.splits)

    def test_receipt_not_moved_when_total_mismatch(self, tmp_path: Path) -> None:
        """Spec 7.3: receipt with materially different total is not moved to processed."""
        gnucash = tmp_path / "book.gnucash"
        with gzip.open(gnucash, "wb") as f:
            f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))
        receipts_dir = tmp_path / "receipts"
        receipts_dir.mkdir()
        processed_dir = tmp_path / "processed"
        receipt_file = receipts_dir / "receipt_mismatch.jpg"
        receipt_file.write_bytes(b"dummy image")

        state_dir = tmp_path / "state"
        state = StateRepository(state_dir)
        state.save_metadata(
            "run_config",
            {
                "gnucash_path": str(gnucash),
                "processed_receipts_dir": str(processed_dir),
            },
        )
        # Proposal for tx_unspec1 (amount £25) but receipt parsed_total £99 — mismatch
        prop = Proposal(
            proposal_id="p1",
            tx_id="tx_unspec1",
            suggested_description="Tesco 15/01/2025",
            suggested_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
            confidence=0.8,
            rationale="ML",
            tx_date=date(2025, 1, 15),
            tx_amount=Decimal("25.00"),
            original_description="Card Payment",
            original_splits=[Split(account_path="Unspecified", amount=Decimal("25.00"))],
            evidence=EvidencePacket(
                tx_id="tx_unspec1",
                receipt=ReceiptEvidence(
                    evidence_id="r1",
                    source_path=str(receipt_file),
                    ocr_text="Total 99.00",
                    parsed_total=Decimal("99.00"),
                ),
            ),
        )
        state.save_proposals([prop])
        state.save_decision(
            ReviewDecision(
                tx_id="tx_unspec1",
                action="approve",
                final_description="Tesco 15/01/2025",
                final_splits=[Split(account_path="Expenses:Food", amount=Decimal("25.00"))],
                decided_at=datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
            )
        )

        ApplyEngine().apply(state_dir)

        assert receipt_file.exists()
        assert not any(processed_dir.glob("*")), "Receipt must not be moved when total mismatches"

    def test_apply_fails_without_metadata(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        engine = ApplyEngine()
        with pytest.raises(RuntimeError, match="No run_config"):
            engine.apply(state_dir)

    def test_apply_raises_when_gnucash_has_no_book(self, tmp_path: Path) -> None:
        """When the GnuCash file has no <gnc:book>, apply raises (invalid file)."""
        invalid = tmp_path / "invalid.gnucash"
        invalid.write_text('<?xml version="1.0"?><root><other/></root>', encoding="utf-8")
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(invalid)})
        state.save_proposals([])
        state.save_decision(
            ReviewDecision(
                tx_id="tx1",
                action="approve",
                final_description="Desc",
                final_splits=[Split(account_path="Expenses:Food", amount=Decimal("10.00"))],
            )
        )
        engine = ApplyEngine()
        with pytest.raises(ValueError, match="gnc:book|No.*book"):
            engine.apply(state_dir)


class TestRollback:

    def test_rollback_restores_original(self, tmp_path: Path) -> None:
        gnucash, state_dir = _setup_state(tmp_path)

        original_loader = GnuCashLoader()
        original_txs = original_loader.load_transactions(gnucash)
        original_desc = {t.tx_id: t.description for t in original_txs}

        engine = ApplyEngine()
        engine.apply(state_dir)

        mod_loader = GnuCashLoader()
        mod_txs = mod_loader.load_transactions(gnucash)
        assert {t.tx_id: t.description for t in mod_txs}["tx_unspec1"] != original_desc[
            "tx_unspec1"
        ]

        engine.rollback(state_dir)

        restored_loader = GnuCashLoader()
        restored_txs = restored_loader.load_transactions(gnucash)
        assert {t.tx_id: t.description for t in restored_txs}["tx_unspec1"] == original_desc[
            "tx_unspec1"
        ]

    def test_rollback_fails_without_backup(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(tmp_path / "book.gnucash")})
        engine = ApplyEngine()
        with pytest.raises(RuntimeError, match="No backup"):
            engine.rollback(state_dir)

    def test_rollback_raises_when_backup_dir_empty(self, tmp_path: Path) -> None:
        """When backup directory exists but has no matching backup files, rollback raises."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "backups").mkdir()
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(tmp_path / "book.gnucash")})
        engine = ApplyEngine()
        with pytest.raises(RuntimeError, match="No backup files found"):
            engine.rollback(state_dir)

    def test_rollback_with_specific_backup_name(self, tmp_path: Path) -> None:
        """Rollback honours an explicit backup filename when provided."""
        # Create original book and run_config metadata
        gnucash = tmp_path / "book.gnucash"
        with gzip.open(gnucash, "wb") as f:
            f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))
        state_dir = tmp_path / "state"
        state = StateRepository(state_dir)
        state.save_metadata("run_config", {"gnucash_path": str(gnucash)})

        backup_dir = state_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create a backup using the real writer
        from gnc_enrich.gnucash.loader import GnuCashWriter

        writer = GnuCashWriter()
        # Backup of the original SAMPLE_GNUCASH_XML
        backup1 = writer.create_backup(gnucash, backup_dir)
        original_bytes = backup1.read_bytes()

        # Mutate the book file so we can see rollback effect clearly
        gnucash.write_bytes(SAMPLE_GNUCASH_XML.encode("utf-8") + b"<!-- modified -->")
        assert gnucash.read_bytes() != original_bytes

        # Roll back explicitly to the first backup by name
        engine = ApplyEngine()
        engine.rollback(state_dir, backup_name=backup1.name)

        # The restored file should match the first backup's contents exactly
        assert gnucash.read_bytes() == original_bytes


class TestBackupRetention:

    def test_prune_backups_keeps_most_recent_n(self, tmp_path: Path) -> None:
        """_prune_backups keeps at most N backups and deletes the oldest ones."""
        gnucash = tmp_path / "book.gnucash"
        gnucash.write_text("dummy", encoding="utf-8")
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create four fake backups with lexicographically increasing timestamps
        stem = gnucash.stem
        suffix = gnucash.suffix
        names = [
            f"{stem}.20250101T120000Z{suffix}",
            f"{stem}.20250102T120000Z{suffix}",
            f"{stem}.20250103T120000Z{suffix}",
            f"{stem}.20250104T120000Z{suffix}",
        ]
        for name in names:
            (backup_dir / name).write_text(name, encoding="utf-8")

        engine = ApplyEngine()
        engine._prune_backups(gnucash, backup_dir, retention=2)

        remaining = sorted(p.name for p in backup_dir.glob(f"{stem}.*{suffix}"))
        assert remaining == names[-2:]
