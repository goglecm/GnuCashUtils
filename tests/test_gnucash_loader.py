"""Tests for GnuCash XML loading, candidate filtering, and writing."""

import dataclasses
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from gnc_enrich.domain.models import Split, Transaction
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

    def test_get_tree_returns_none_before_load(self) -> None:
        """get_tree() returns None when no file has been loaded (graceful accessor)."""
        loader = GnuCashLoader()
        assert loader.get_tree() is None

    def test_get_account_path_returns_empty_for_unknown_id(self, sample_gnucash_path: Path) -> None:
        """get_account_path(unknown_id) returns empty string (non-fatal, spec-aligned)."""
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        assert loader.get_account_path("unknown-guid") == ""
        assert loader.get_account_path("") == ""

    def test_load_transactions_raises_when_no_gnc_book(self, tmp_path: Path) -> None:
        """load_transactions raises ValueError when XML has no <gnc:book> (invalid file)."""
        invalid_xml = tmp_path / "no_book.xml"
        invalid_xml.write_text(
            '<?xml version="1.0"?><gnc-v2 xmlns:gnc="http://www.gnucash.org/XML/gnc"><gnc:count-data>1</gnc:count-data></gnc-v2>',
            encoding="utf-8",
        )
        loader = GnuCashLoader()
        with pytest.raises(ValueError, match="No.*gnc:book"):
            loader.load_transactions(invalid_xml)


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

    def test_excludes_settled_transfers(self, sample_gnucash_path: Path) -> None:
        """Settled transfers (both legs to bank/asset) are never candidates."""
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        ids = {c.tx_id for c in candidates}
        assert "tx_transfer" not in ids

    def test_unsettled_transfers_only_when_description_suggests_transfer(
        self, sample_gnucash_path: Path
    ) -> None:
        """Only 2-split (target + bank) with 'transfer' etc. in description are is_transfer=True; else expense."""
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" in ids
        expense_like = next(c for c in candidates if c.tx_id == "tx_unspec1")
        assert expense_like.is_transfer is False
        assert getattr(expense_like, "is_unsettled_transfer", False) is False

    def test_candidate_count(self, sample_gnucash_path: Path) -> None:
        """Only Unspecified/Imbalance-GBP; settled transfer excluded → 3 candidates."""
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        assert len(candidates) == 3

    def test_skipped_ids_excluded(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, skipped_ids={"tx_unspec1"})
        ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" not in ids
        assert len(candidates) == 2

    def test_include_skipped(self, sample_gnucash_path: Path) -> None:
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs, include_skipped=True, skipped_ids={"tx_unspec1"})
        ids = {c.tx_id for c in candidates}
        assert "tx_unspec1" in ids
        assert len(candidates) == 3

    def test_settled_transfer_not_unsettled(self, sample_gnucash_path: Path) -> None:
        """Settled transfer (both legs to bank/asset) is excluded and _is_unsettled_transfer returns False."""
        loader = GnuCashLoader()
        txs = loader.load_transactions(sample_gnucash_path)
        candidates = loader.filter_candidates(txs)
        assert "tx_transfer" not in {c.tx_id for c in candidates}
        tx_transfer = next(t for t in txs if t.tx_id == "tx_transfer")
        assert loader._is_unsettled_transfer(tx_transfer) is False

    def test_has_target_account_matches_any_path_segment(self) -> None:
        """Unspecified/Imbalance-GBP under parent (e.g. Expenses:Unspecified) must count as candidate."""
        loader = GnuCashLoader()
        base = {
            "tx_id": "t",
            "posted_date": date(2020, 1, 1),
            "description": "x",
            "currency": "GBP",
            "amount": Decimal("1"),
        }
        tx_unspec_under_expenses = Transaction(
            **base,
            splits=[Split(account_path="Expenses:Unspecified", amount=Decimal("1"), memo="")],
        )
        tx_imbalance_nested = Transaction(
            **base,
            splits=[
                Split(account_path="Assets:Current:Imbalance-GBP", amount=Decimal("1"), memo="")
            ],
        )
        tx_food_not_target = Transaction(
            **base,
            splits=[Split(account_path="Expenses:Food", amount=Decimal("1"), memo="")],
        )
        assert loader._has_target_account(tx_unspec_under_expenses) is True
        assert loader._has_target_account(tx_imbalance_nested) is True
        assert loader._has_target_account(tx_food_not_target) is False

    def test_description_with_transfer_keyword_marked_unsettled_transfer(
        self, sample_gnucash_path: Path
    ) -> None:
        """2-split (one target, one bank) with 'transfer' in description is is_unsettled_transfer=True."""
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tx = Transaction(
            tx_id="t1",
            posted_date=date(2025, 1, 1),
            description="Transfer to Savings",
            currency="GBP",
            amount=Decimal("100.00"),
            splits=[
                Split(account_path="Current Account", amount=Decimal("-100.00"), memo=""),
                Split(account_path="Unspecified", amount=Decimal("100.00"), memo=""),
            ],
        )
        assert loader._is_unsettled_transfer(tx) is True
        tx_expense = dataclasses.replace(tx, description="Card Payment")
        assert loader._is_unsettled_transfer(tx_expense) is False

    def test_three_splits_with_one_target_not_unsettled_transfer(
        self, sample_gnucash_path: Path
    ) -> None:
        """Only 2-split transactions (one target, one own account) are marked unsettled transfer."""
        loader = GnuCashLoader()
        loader.load_transactions(sample_gnucash_path)
        tx_three = Transaction(
            tx_id="t3",
            posted_date=date(2025, 1, 1),
            description="Split payment",
            currency="GBP",
            amount=Decimal("50.00"),
            splits=[
                Split(account_path="Current Account", amount=Decimal("-50.00"), memo=""),
                Split(account_path="Unspecified", amount=Decimal("25.00"), memo=""),
                Split(account_path="Expenses:Food", amount=Decimal("25.00"), memo=""),
            ],
        )
        assert loader._has_target_account(tx_three) is True
        assert loader._is_unsettled_transfer(tx_three) is False


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

    def test_write_changes_raises_when_tree_has_no_book(self, tmp_path: Path) -> None:
        """write_changes raises ValueError when tree has no <gnc:book> (graceful failure)."""
        from lxml import etree

        root = etree.fromstring(
            '<?xml version="1.0"?><gnc-v2 xmlns:gnc="http://www.gnucash.org/XML/gnc"></gnc-v2>'
        )
        tree = etree.ElementTree(root)
        source = tmp_path / "dummy.gnucash"
        source.write_text("", encoding="utf-8")
        writer = GnuCashWriter()
        with pytest.raises(ValueError, match="No.*gnc:book"):
            writer.write_changes(source, tree, {}, in_place=True)
