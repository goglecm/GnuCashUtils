"""Apply approved changes with backup, undo journal, and audit trail."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.domain.models import AuditEntry, ReviewDecision
from gnc_enrich.gnucash.loader import GnuCashLoader, GnuCashWriter
from gnc_enrich.receipt.repository import ReceiptRepository
from gnc_enrich.state.repository import StateRepository

logger = logging.getLogger(__name__)


class ApplyEngine:
    """Applies approved review decisions to the GnuCash file."""

    def generate_dry_run_report(self, state_dir: Path) -> Path:
        """Produce a human-readable diff report without modifying any files."""
        state = StateRepository(state_dir)
        proposals = state.load_proposals()
        decisions = state.load_decisions()

        decision_map = {d.tx_id: d for d in decisions}
        proposal_map = {p.tx_id: p for p in proposals}

        report_lines: list[str] = []
        report_lines.append("=" * 70)
        report_lines.append("DRY-RUN REPORT")
        report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report_lines.append("=" * 70)
        report_lines.append("")

        approved = 0
        skipped = 0
        edited = 0

        for tx_id, dec in decision_map.items():
            prop = proposal_map.get(tx_id)
            report_lines.append(f"Transaction: {tx_id}")
            report_lines.append(f"  Action: {dec.action}")

            if prop:
                report_lines.append(f"  Proposed description: {prop.suggested_description}")
                report_lines.append(f"  Confidence: {prop.confidence:.0%}")

            report_lines.append(f"  Final description: {dec.final_description}")
            for sp in dec.final_splits:
                report_lines.append(f"  Split: {sp.account_path} = £{sp.amount}")

            report_lines.append("")

            if dec.action == "approve":
                approved += 1
            elif dec.action == "skip":
                skipped += 1
            elif dec.action == "edit":
                edited += 1

        report_lines.append("-" * 70)
        report_lines.append(f"Summary: {approved} approved, {edited} edited, {skipped} skipped")

        report_path = state_dir / "dry_run_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info("Dry-run report written to %s", report_path)
        return report_path

    def apply(
        self,
        state_dir: Path,
        *,
        create_backup: bool = True,
        backup_dir: Path | None = None,
        in_place: bool = True,
    ) -> None:
        """Apply all approved/edited decisions to the GnuCash file.

        Parameters
        ----------
        state_dir:
            Directory containing proposals, decisions, and run metadata.
        create_backup:
            Whether to create a backup before writing (default True).
        backup_dir:
            Where to store the backup.  Falls back to ``state_dir / "backups"``.
        in_place:
            Write changes back to the original file when True (default);
            otherwise write to ``<original>.enriched.gnucash``.
        """
        state = StateRepository(state_dir)

        meta = state.load_metadata("run_config")
        if not meta or "gnucash_path" not in meta:
            raise RuntimeError("No run_config metadata found; run the pipeline first")

        gnucash_path = Path(meta["gnucash_path"])
        raw_receipts_dir = meta.get("processed_receipts_dir", "")
        processed_receipts_dir = Path(raw_receipts_dir) if raw_receipts_dir else None

        proposals = state.load_proposals()
        decisions = state.load_decisions()
        decision_map = {d.tx_id: d for d in decisions}
        proposal_map = {p.tx_id: p for p in proposals}

        approved_decisions = {
            tx_id: dec
            for tx_id, dec in decision_map.items()
            if dec.action in ("approve", "edit")
        }

        if not approved_decisions:
            logger.info("No approved decisions to apply")
            return

        resolved_backup_dir = backup_dir or (state_dir / "backups")
        writer = GnuCashWriter()
        backup_path = None
        if create_backup:
            backup_path = writer.create_backup(gnucash_path, resolved_backup_dir)

        journal: list[dict] = []
        loader = GnuCashLoader()
        loader.load_transactions(gnucash_path)
        tree = loader.get_tree()
        if tree is None:
            raise RuntimeError(f"Failed to parse GnuCash file: {gnucash_path}")

        changes: dict[str, dict] = {}
        for tx_id, dec in approved_decisions.items():
            prop = proposal_map.get(tx_id)
            is_transfer = prop and getattr(prop, "is_transfer", False)
            splits_for_change = (
                []
                if is_transfer
                else [
                    {"account_path": sp.account_path, "amount": str(sp.amount), "memo": sp.memo}
                    for sp in dec.final_splits
                ]
            )
            changes[tx_id] = {
                "description": dec.final_description,
                "splits": splits_for_change,
            }

            original_desc = ""
            if prop:
                original_desc = prop.original_description or prop.suggested_description

            journal.append({
                "tx_id": tx_id,
                "original_description": original_desc,
                "new_description": dec.final_description,
                "action": dec.action,
                "backup_path": str(backup_path) if backup_path else "",
            })

            state.append_audit(AuditEntry(
                entry_id=uuid.uuid4().hex[:12],
                tx_id=tx_id,
                action=dec.action,
                proposed_description=prop.suggested_description if prop else "",
                proposed_splits=prop.suggested_splits if prop else [],
                final_description=dec.final_description,
                final_splits=dec.final_splits,
                confidence=prop.confidence if prop else 0.0,
                evidence_ids=self._collect_evidence_ids(prop),
                timestamp=datetime.now(timezone.utc),
            ))

        writer.write_changes(gnucash_path, tree, changes, in_place=in_place)

        if processed_receipts_dir:
            self._move_compatible_receipts(
                proposals, approved_decisions, processed_receipts_dir
            )

        journal_path = state_dir / "apply_journal.jsonl"
        if not journal_path.exists() or journal_path.stat().st_size == 0:
            journal_path.write_text(
                json.dumps({"_schema_version": 1}) + "\n", encoding="utf-8"
            )
        with journal_path.open("a", encoding="utf-8") as f:
            for entry in journal:
                f.write(json.dumps(entry) + "\n")

        logger.info(
            "Applied %d changes; journal at %s", len(approved_decisions), journal_path
        )

    def rollback(self, state_dir: Path) -> None:
        """Restore from the most recent backup."""
        state = StateRepository(state_dir)
        meta = state.load_metadata("run_config")
        if not meta or "gnucash_path" not in meta:
            raise RuntimeError("No run_config metadata found")

        gnucash_path = Path(meta["gnucash_path"])
        backup_dir = state_dir / "backups"

        if not backup_dir.exists():
            raise RuntimeError("No backup directory found")

        backups = sorted(backup_dir.glob(f"{gnucash_path.stem}.*{gnucash_path.suffix}"))
        if not backups:
            raise RuntimeError("No backup files found")

        latest_backup = backups[-1]
        import shutil
        shutil.copy2(latest_backup, gnucash_path)
        logger.info("Rolled back %s from backup %s", gnucash_path, latest_backup)

    def _collect_evidence_ids(self, prop) -> list[str]:
        """Gather evidence IDs from a proposal for the audit trail."""
        if not prop or not prop.evidence:
            return []
        ids = [e.evidence_id for e in (prop.evidence.emails or [])]
        if prop.evidence.receipt:
            ids.append(prop.evidence.receipt.evidence_id)
        return ids

    def _move_compatible_receipts(
        self,
        proposals,
        approved_decisions: dict[str, ReviewDecision],
        processed_dir: Path,
        amount_tolerance: float = 0.50,
    ) -> None:
        """Move receipts whose total matches the approved transaction amount."""
        receipt_repo = ReceiptRepository()
        tol = Decimal(str(amount_tolerance))

        for prop in proposals:
            if prop.tx_id not in approved_decisions:
                continue
            if not prop.evidence or not prop.evidence.receipt:
                continue

            receipt = prop.evidence.receipt
            receipt_path = Path(receipt.source_path)
            if not receipt_path.exists():
                continue
            if receipt.parsed_total is None:
                logger.debug("Receipt %s has no parsed total; not moved", receipt_path)
                continue

            dec = approved_decisions[prop.tx_id]
            tx_amount = sum(
                sp.amount for sp in dec.final_splits if sp.amount > 0
            ) or Decimal(0)

            if abs(receipt.parsed_total - tx_amount) <= tol:
                receipt_repo.mark_processed(receipt_path, processed_dir)
                logger.info("Moved compatible receipt %s", receipt_path)
            else:
                logger.info(
                    "Receipt %s not amount-compatible (receipt=£%s, tx=£%s); not moved",
                    receipt_path,
                    receipt.parsed_total,
                    tx_amount,
                )
