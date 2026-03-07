from datetime import date
from decimal import Decimal
from pathlib import Path

from gnc_enrich.config import ApplyConfig, ReviewConfig, RunConfig
from gnc_enrich.domain.models import Split, Transaction


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

    review = ReviewConfig(state_dir=Path("state"))
    assert review.host == "127.0.0.1"
    assert review.port == 7860

    apply = ApplyConfig(state_dir=Path("state"))
    assert apply.create_backup is True


def test_domain_dataclass_roundtrip() -> None:
    split = Split(account_path="Expenses:Food", amount=Decimal("10.00"), memo="Lunch")
    tx = Transaction(
        tx_id="abc",
        posted_date=date(2025, 1, 1),
        description="Shop",
        currency="GBP",
        amount=Decimal("10.00"),
        splits=[split],
    )
    assert tx.splits[0].account_path == "Expenses:Food"
