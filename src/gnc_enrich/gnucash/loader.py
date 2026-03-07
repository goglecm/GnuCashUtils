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
    """Parse GnuCash fraction like '1500/100' -> Decimal('15.00').

    Handles malformed values gracefully by returning Decimal(0) and logging
    a warning rather than crashing on corrupt GnuCash data.
    """
    try:
        if "/" in value_str:
            parts = value_str.split("/", maxsplit=1)
            num, denom = Decimal(parts[0]), Decimal(parts[1])
            if denom == 0:
                logger.warning("Zero denominator in fraction: %s", value_str)
                return Decimal(0)
            return num / denom
        return Decimal(value_str)
    except Exception:
        logger.warning("Unparseable split value: %s", value_str)
        return Decimal(0)


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
    """Check if a file is gzip-compressed by reading its magic bytes."""
    with open(path, "rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def _parse_tree(source: Path) -> etree._ElementTree:
    """Parse a GnuCash XML file, handling both gzip and plain XML."""
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
        """Return the colon-separated path for an account ID."""
        return self._account_paths.get(account_id, "")

    def get_tree(self) -> etree._ElementTree | None:
        """Return the parsed lxml ElementTree, or None if not yet loaded."""
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
        if hasattr(self, "_path_to_account"):
            del self._path_to_account

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
        """Look up an account by its colon-separated path (uses cached reverse map)."""
        if not hasattr(self, "_path_to_account"):
            self._path_to_account: dict[str, Account] = {
                acct.full_path: acct for acct in self._accounts.values()
            }
        return self._path_to_account.get(path)

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

        loader = GnuCashLoader()
        loader._accounts = {}
        loader._account_paths = {}
        loader._build_account_map(book)

        new_categories: set[str] = set()
        for change in changes.values():
            for sp in change.get("splits", []):
                new_categories.add(sp["account_path"])

        if new_categories:
            self._ensure_accounts_exist(book, new_categories, loader)

        account_guid_by_path: dict[str, str] = {
            acct.full_path: aid for aid, acct in loader._accounts.items()
        }

        for trn_el in book.findall("gnc:transaction", _NS):
            tx_id = _text(trn_el.find("trn:id", _NS))
            if tx_id not in changes:
                continue

            change = changes[tx_id]

            if "description" in change:
                desc_el = trn_el.find("trn:description", _NS)
                if desc_el is not None:
                    desc_el.text = change["description"]

            if "splits" in change and change["splits"]:
                splits_el = trn_el.find("trn:splits", _NS)
                if splits_el is not None:
                    self._apply_split_changes(
                        splits_el, change["splits"], account_guid_by_path, loader,
                    )

        output = source if in_place else source.with_suffix(".enriched.gnucash")

        xml_bytes = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
        with gzip.open(output, "wb") as f:
            f.write(xml_bytes)

        logger.info("Wrote changes to %s", output)
        return output

    @staticmethod
    def _decimal_to_fraction(amount_str: str) -> str:
        """Convert a decimal amount string to GnuCash fraction format (e.g. '25.00' -> '2500/100')."""
        d = Decimal(amount_str)
        num = int(d * 100)
        return f"{num}/100"

    def _apply_split_changes(
        self,
        splits_el: etree._Element,
        sp_changes: list[dict],
        account_guid_by_path: dict[str, str],
        loader: GnuCashLoader,
    ) -> None:
        """Replace target splits with user-specified splits, updating amounts."""
        import uuid as _uuid

        target_els = []
        for sp_el in splits_el.findall("trn:split", _NS):
            acct_el = sp_el.find("split:account", _NS)
            if acct_el is None:
                continue
            acct_path = loader._account_paths.get(_text(acct_el), "")
            top = acct_path.split(":")[0] if acct_path else ""
            if top in _TARGET_ACCOUNTS:
                target_els.append(sp_el)

        applied = 0
        for sp_change in sp_changes:
            target_guid = account_guid_by_path.get(sp_change["account_path"])
            if not target_guid:
                logger.warning("No GUID for account %s; skipping split", sp_change["account_path"])
                continue

            frac = self._decimal_to_fraction(sp_change["amount"])

            if applied < len(target_els):
                sp_el = target_els[applied]
                acct_el = sp_el.find("split:account", _NS)
                acct_el.text = target_guid
                val_el = sp_el.find("split:value", _NS)
                if val_el is not None:
                    val_el.text = frac
                qty_el = sp_el.find("split:quantity", _NS)
                if qty_el is not None:
                    qty_el.text = frac
            else:
                sp_el = etree.SubElement(splits_el, "{%s}split" % _NS["trn"])
                id_el = etree.SubElement(sp_el, "{%s}id" % _NS["split"], type="guid")
                id_el.text = _uuid.uuid4().hex
                state_el = etree.SubElement(sp_el, "{%s}reconciled-state" % _NS["split"])
                state_el.text = "n"
                val_el = etree.SubElement(sp_el, "{%s}value" % _NS["split"])
                val_el.text = frac
                qty_el = etree.SubElement(sp_el, "{%s}quantity" % _NS["split"])
                qty_el.text = frac
                acct_el = etree.SubElement(sp_el, "{%s}account" % _NS["split"], type="guid")
                acct_el.text = target_guid

            if sp_change.get("memo"):
                memo_el = sp_el.find("split:memo", _NS)
                if memo_el is None:
                    memo_el = etree.SubElement(sp_el, "{%s}memo" % _NS["split"])
                memo_el.text = sp_change["memo"]

            applied += 1

        for j in range(applied, len(target_els)):
            splits_el.remove(target_els[j])

    def _ensure_accounts_exist(
        self, book: etree._Element, category_paths: set[str], loader: GnuCashLoader
    ) -> None:
        """Create new account elements for category paths that don't exist yet."""
        existing: set[str] = {acct.full_path for acct in loader._accounts.values()}

        for cat_path in category_paths:
            if cat_path in existing:
                continue
            self._create_account_chain(book, cat_path, existing, loader)

    def _create_account_chain(
        self,
        book: etree._Element,
        full_path: str,
        existing: set[str],
        loader: GnuCashLoader,
    ) -> None:
        """Create any missing account nodes along a path like 'Expenses:Food:Takeaway'."""
        import uuid as _uuid

        parts = full_path.split(":")
        for i in range(1, len(parts) + 1):
            partial = ":".join(parts[:i])
            if partial in existing:
                continue

            parent_path = ":".join(parts[: i - 1]) if i > 1 else ""
            parent_id = ""
            if parent_path:
                for acct in loader._accounts.values():
                    if acct.full_path == parent_path:
                        parent_id = acct.account_id
                        break
            else:
                for acct in loader._accounts.values():
                    if acct.account_type == "ROOT":
                        parent_id = acct.account_id
                        break

            new_id = _uuid.uuid4().hex
            acct_el = etree.SubElement(book, "{%s}account" % _NS["gnc"], version="2.0.0")
            name_el = etree.SubElement(acct_el, "{%s}name" % _NS["act"])
            name_el.text = parts[i - 1]
            id_el = etree.SubElement(acct_el, "{%s}id" % _NS["act"], type="guid")
            id_el.text = new_id
            type_el = etree.SubElement(acct_el, "{%s}type" % _NS["act"])
            type_el.text = "EXPENSE"

            cmdty_el = etree.SubElement(acct_el, "{%s}commodity" % _NS["act"])
            space_el = etree.SubElement(cmdty_el, "{%s}space" % _NS["cmdty"])
            space_el.text = "ISO4217"
            cid_el = etree.SubElement(cmdty_el, "{%s}id" % _NS["cmdty"])
            cid_el.text = "GBP"

            if parent_id:
                par_el = etree.SubElement(acct_el, "{%s}parent" % _NS["act"], type="guid")
                par_el.text = parent_id

            new_acct = Account(
                account_id=new_id,
                name=parts[i - 1],
                full_path=partial,
                account_type="EXPENSE",
                currency="GBP",
                parent_id=parent_id or None,
            )
            loader._accounts[new_id] = new_acct
            loader._account_paths[new_id] = partial
            existing.add(partial)
            logger.info("Created new account: %s (id=%s)", partial, new_id)
