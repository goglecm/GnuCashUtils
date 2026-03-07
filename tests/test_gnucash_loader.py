"""Tests for GnuCash XML loading, candidate filtering, and writing."""

from datetime import date
from decimal import Decimal
from pathlib import Path

from gnc_enrich.gnucash.loader import GnuCashLoader, GnuCashWriter


class TestGnuCashLoader:

    def test_load_all_transactions(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        assert len(txs) == 8

    def test_load_from_uncompressed_xml(self, sample_gnucash_xml_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_xml_path)
        assert len(txs) == 8

    def test_transaction_fields(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        tx_map = {t.tx_id: t for t in txs}

        food_tx = tx_map["tx_normal_food"]
        assert food_tx.description == "Tesco Groceries"
        assert food_tx.currency == "GBP"
        assert food_tx.posted_date == date(2025, 1, 10)
        assert food_tx.amount == Decimal("15.00")
        assert len(food_tx.splits) == 2

    def test_split_amounts(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        tx_map = {t.tx_id: t for t in txs}

        unspec_tx = tx_map["tx_unspec1"]
        amounts = sorted(s.amount for s in unspec_tx.splits)
        assert amounts == [Decimal("-25.00"), Decimal("25.00")]

    def test_account_paths_resolved(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        tx_map = {t.tx_id: t for t in txs}

        food_tx = tx_map["tx_normal_food"]
        paths = {s.account_path for s in food_tx.splits}
        assert "Expenses:Food" in paths
        assert "Current Account" in paths

    def test_load_accounts(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        accounts = loader.load_accounts(sample_gnucash_path)
        names = {a.name for a in accounts}
        assert "Current Account" in names
        assert "Unspecified" in names
        assert "Imbalance-GBP" in names
        assert "Food" in names


class TestCandidateFiltering:

    def test_filters_to_target_accounts_only(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)

        candidate_ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" in candidate_ids
        assert "tx_unspec2" in candidate_ids
        assert "tx_imbalance1" in candidate_ids

    def test_excludes_normal_categorised(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        ids = {c.tx_id for c in candidates}
        assert "tx_normal_food" not in ids
        assert "tx_normal_food2" not in ids

    def test_excludes_non_gbp(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        ids = {c.tx_id for c in candidates}
        assert "tx_eur" not in ids

    def test_excludes_future_dated(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        ids = {c.tx_id for c in candidates}
        assert "tx_future" not in ids

    def test_excludes_transfers_when_requested(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, include_transfers=False)
        ids = {c.tx_id for c in candidates}
        assert "tx_transfer" not in ids

    def test_includes_transfers_in_separate_queue_by_default(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, include_transfers=True)
        ids = {c.tx_id for c in candidates}
        assert "tx_transfer" in ids
        transfer = next(c for c in candidates if c.tx_id == "tx_transfer")
        assert transfer.is_transfer is True
        expense = next(c for c in candidates if c.tx_id != "tx_transfer")
        assert expense.is_transfer is False

    def test_candidate_count(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, include_transfers=True)
        assert len(candidates) == 4

    def test_skipped_ids_excluded(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, skipped_ids={"tx_unspec1"})
        ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" not in ids
        assert len(candidates) == 3

    def test_include_skipped(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(
            txs, include_skipped=True, skipped_ids={"tx_unspec1"}
        )
        ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" in ids


class TestGnuCashWriter:

    def test_create_backup(self, sample_gnucash_path: Path, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backups"
        writer = GnuCashWriter()
        backup = writer.create_backup(sample_gnucash_path, backup_dir)
        assert backup.exists()
        assert backup.parent == backup_dir
        assert backup.stat().st_size == sample_gnucash_path.stat().st_size

    def test_write_changes_updates_description(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tree = loader.get_tree()

        writer = GnuCashWriter()
        changes = {
            "tx_unspec1": {"description": "Updated: Card Payment at Tesco 15/01/2025"},
        }
        output = writer.write_changes(sample_gnucash_path, tree, changes, in_place=True)

        reloaded_loader = GnuCashLoader()
        txs = reloaded_loader.load_transactions(output)
        tx_map = {t.tx_id: t for t in txs}
        assert tx_map["tx_unspec1"].description == "Updated: Card Payment at Tesco 15/01/2025"

    def test_write_out_of_place(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tree = loader.get_tree()

        writer = GnuCashWriter()
        output = writer.write_changes(sample_gnucash_path, tree, {}, in_place=False)
        assert output != sample_gnucash_path
        assert output.exists()
