"""GnuCash loading and writing adapters for XML files (.gnucash)."""

from __future__ import annotations

import gzip
import logging
import shutil
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from lxml import etree

from gnc_enrich.domain.models import Account, Split, Transaction

logger = logging.getLogger(__name__)

_NS = {
    "gnc": "http://www.gnucash.org/XML/gnc",
    "act": "http://www.gnucash.org/XML/act",
    "book": "http://www.gnucash.org/XML/book",
    "cd": "http://www.gnucash.org/XML/cd",
    "cmdty": "http://www.gnucash.org/XML/cmdty",
    "slot": "http://www.gnucash.org/XML/slot",
    "split": "http://www.gnucash.org/XML/split",
    "trn": "http://www.gnucash.org/XML/trn",
    "ts": "http://www.gnucash.org/XML/ts",
}

_TARGET_ACCOUNTS = {"Unspecified", "Imbalance-GBP"}

_TRANSFER_TYPES = {"BANK", "ASSET", "LIABILITY", "CASH", "CREDIT", "EQUITY"}


def _text(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _parse_fraction(value_str: str) -> Decimal:
    """Parse GnuCash fraction like '1500/100' -> Decimal('15.00')."""
    if "/" in value_str:
        num, denom = value_str.split("/")
        return Decimal(num) / Decimal(denom)
    return Decimal(value_str)


def _parse_date(date_el: etree._Element | None) -> date:
    if date_el is None:
        raise ValueError("Missing date element")
    ts_el = date_el.find("ts:date", _NS)
    raw = _text(ts_el)
    # GnuCash date format: "2025-01-15 00:00:00 +0000"
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unparseable date: {raw}")


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def _parse_tree(source: Path) -> etree._ElementTree:
    if _is_gzip(source):
        with gzip.open(source, "rb") as f:
            return etree.parse(f)
    return etree.parse(str(source))


class GnuCashLoader:
    """Loads transactions and account metadata from GnuCash XML files."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._account_paths: dict[str, str] = {}
        self._tree: etree._ElementTree | None = None

    def load_transactions(self, source: Path) -> list[Transaction]:
        """Parse a .gnucash XML file and return all transactions."""
        self._tree = _parse_tree(source)
        root = self._tree.getroot()
        book = root.find("gnc:book", _NS)
        if book is None:
            raise ValueError("No <gnc:book> element found")

        self._build_account_map(book)
        return self._extract_transactions(book)

    def load_accounts(self, source: Path) -> list[Account]:
        """Return all accounts from a .gnucash file."""
        if not self._accounts:
            self.load_transactions(source)
        return list(self._accounts.values())

    def get_account_path(self, account_id: str) -> str:
        return self._account_paths.get(account_id, "")

    def get_tree(self) -> etree._ElementTree | None:
        return self._tree

    def filter_candidates(
        self,
        transactions: list[Transaction],
        *,
        include_skipped: bool = False,
        skipped_ids: set[str] | None = None,
    ) -> list[Transaction]:
        """Filter to only actionable target transactions per domain rules."""
        today = date.today()
        skipped = skipped_ids or set()
        candidates = []

        for tx in transactions:
            if not include_skipped and tx.tx_id in skipped:
                continue
            if tx.posted_date > today:
                continue
            if tx.currency != "GBP":
                continue
            if self._is_transfer(tx):
                continue
            if not self._has_target_account(tx):
                continue
            candidates.append(tx)

        return candidates

    def _build_account_map(self, book: etree._Element) -> None:
        self._accounts.clear()
        self._account_paths.clear()

        for acct_el in book.findall("gnc:account", _NS):
            acct_id = _text(acct_el.find("act:id", _NS))
            name = _text(acct_el.find("act:name", _NS))
            acct_type = _text(acct_el.find("act:type", _NS))

            parent_el = acct_el.find("act:parent", _NS)
            parent_id = _text(parent_el) if parent_el is not None else None

            cmdty_el = acct_el.find("act:commodity", _NS)
            currency = ""
            if cmdty_el is not None:
                currency = _text(cmdty_el.find("cmdty:id", _NS))

            self._accounts[acct_id] = Account(
                account_id=acct_id,
                name=name,
                full_path="",
                account_type=acct_type,
                currency=currency,
                parent_id=parent_id,
            )

        for acct_id in self._accounts:
            self._account_paths[acct_id] = self._resolve_path(acct_id)
            self._accounts[acct_id].full_path = self._account_paths[acct_id]

    def _resolve_path(self, acct_id: str) -> str:
        parts: list[str] = []
        current = acct_id
        seen: set[str] = set()
        while current and current in self._accounts:
            if current in seen:
                break
            seen.add(current)
            acct = self._accounts[current]
            if acct.account_type == "ROOT":
                break
            parts.append(acct.name)
            current = acct.parent_id or ""
        parts.reverse()
        return ":".join(parts)

    def _extract_transactions(self, book: etree._Element) -> list[Transaction]:
        transactions: list[Transaction] = []

        for trn_el in book.findall("gnc:transaction", _NS):
            tx_id = _text(trn_el.find("trn:id", _NS))
            description = _text(trn_el.find("trn:description", _NS))

            currency_el = trn_el.find("trn:currency", _NS)
            currency = _text(currency_el.find("cmdty:id", _NS)) if currency_el is not None else ""

            posted_date = _parse_date(trn_el.find("trn:date-posted", _NS))

            splits: list[Split] = []
            total_amount = Decimal(0)

            splits_el = trn_el.find("trn:splits", _NS)
            if splits_el is not None:
                for sp_el in splits_el.findall("trn:split", _NS):
                    acct_id = _text(sp_el.find("split:account", _NS))
                    value_str = _text(sp_el.find("split:value", _NS))
                    memo = _text(sp_el.find("split:memo", _NS))
                    amount = _parse_fraction(value_str) if value_str else Decimal(0)
                    acct_path = self._account_paths.get(acct_id, acct_id)

                    splits.append(Split(account_path=acct_path, amount=amount, memo=memo))
                    if amount > 0:
                        total_amount += amount

            account_name = ""
            original_category = ""
            for sp in splits:
                acct = self._find_account_by_path(sp.account_path)
                if acct and acct.account_type in _TRANSFER_TYPES:
                    account_name = sp.account_path
                else:
                    original_category = sp.account_path

            transactions.append(Transaction(
                tx_id=tx_id,
                posted_date=posted_date,
                description=description,
                currency=currency,
                amount=total_amount,
                splits=splits,
                account_name=account_name,
                original_category=original_category,
            ))

        return transactions

    def _find_account_by_path(self, path: str) -> Account | None:
        for acct in self._accounts.values():
            if acct.full_path == path:
                return acct
        return None

    def _has_target_account(self, tx: Transaction) -> bool:
        for sp in tx.splits:
            top_level = sp.account_path.split(":")[0]
            if top_level in _TARGET_ACCOUNTS:
                return True
        return False

    def _is_transfer(self, tx: Transaction) -> bool:
        """A transfer has all splits going to asset/liability/bank accounts."""
        if not tx.splits:
            return False
        for sp in tx.splits:
            acct = self._find_account_by_path(sp.account_path)
            if acct is None:
                return False
            if acct.account_type not in _TRANSFER_TYPES:
                return False
        return True


class GnuCashWriter:
    """Applies approved mutations to GnuCash XML files with backup support."""

    def create_backup(self, source: Path, backup_dir: Path) -> Path:
        """Copy the source file to backup_dir with a timestamped name."""
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"{source.stem}.{ts}{source.suffix}"
        dest = backup_dir / backup_name
        shutil.copy2(source, dest)
        logger.info("Backup created: %s", dest)
        return dest

    def write_changes(
        self,
        source: Path,
        tree: etree._ElementTree,
        changes: dict[str, dict],
        *,
        in_place: bool = True,
    ) -> Path:
        """Apply description/split changes to the XML tree and write out.

        changes is a mapping of tx_id -> {"description": str, "splits": [{"account_path": str, "amount": str, "memo": str}]}
        """
        root = tree.getroot()
        book = root.find("gnc:book", _NS)
        if book is None:
            raise ValueError("No <gnc:book> element found")

        for trn_el in book.findall("gnc:transaction", _NS):
            tx_id = _text(trn_el.find("trn:id", _NS))
            if tx_id not in changes:
                continue

            change = changes[tx_id]

            if "description" in change:
                desc_el = trn_el.find("trn:description", _NS)
                if desc_el is not None:
                    desc_el.text = change["description"]

        output = source if in_place else source.with_suffix(".enriched.gnucash")

        xml_bytes = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
        with gzip.open(output, "wb") as f:
            f.write(xml_bytes)

        logger.info("Wrote changes to %s", output)
        return output
