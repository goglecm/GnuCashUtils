"""Review workflow service managing proposal queue and decision capture."""

from __future__ import annotations

import dataclasses
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import (
    EmailEvidence,
    Proposal,
    ReceiptEvidence,
    ReviewDecision,
    SkipRecord,
    Split,
    Transaction,
)
from gnc_enrich.ml.predictor import CategoryPredictor, FeedbackTrainer
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.state.repository import StateRepository

logger = logging.getLogger(__name__)


def _queue_order(proposals: list[Proposal]) -> list[Proposal]:
    """Return proposals sorted by tx_date for consistent queue/navigation order."""
    return sorted(proposals, key=lambda p: (p.tx_date or date.min, p.proposal_id))


class ReviewQueueService:
    """Serves one-by-one proposals to the reviewer and stores decisions."""

    def __init__(self, state_repo: StateRepository) -> None:
        self._state = state_repo
        self._feedback = FeedbackTrainer(state_dir=state_repo.state_dir)
        self._llm_config = self._load_llm_config()
        self._proposals: list[Proposal] = []
        self._decided_ids: set[str] = set()
        self._reload()

    def _load_llm_config(self) -> LlmConfig:
        """Load LLM config from run metadata, falling back to disabled."""
        meta = self._state.load_metadata("run_config")
        if not meta:
            return LlmConfig()
        try:
            return LlmConfig(
                mode=LlmMode(meta.get("llm_mode", "disabled")),
                endpoint=meta.get("llm_endpoint", ""),
                model_name=meta.get("llm_model", ""),
                use_web=meta.get("llm_use_web", False),
                extraction_endpoint=meta.get("llm_extraction_endpoint", ""),
                extraction_model=meta.get("llm_extraction_model", ""),
                extraction_api_key=meta.get("llm_extraction_api_key", ""),
            )
        except (ValueError, KeyError):
            return LlmConfig()

    def _reload(self) -> None:
        self._proposals = self._state.load_proposals()
        decisions = self._state.load_decisions()
        self._decided_ids = {d.tx_id for d in decisions}

    @property
    def total_count(self) -> int:
        return len(self._proposals)

    @property
    def pending_count(self) -> int:
        return sum(1 for p in self._proposals if p.tx_id not in self._decided_ids)

    @property
    def decided_count(self) -> int:
        return len(self._decided_ids)

    def next_proposal(self) -> Proposal | None:
        """Return the next unreviewed proposal, or None if all are reviewed."""
        for p in self._proposals:
            if p.tx_id not in self._decided_ids:
                return p
        return None

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        """Look up a specific proposal by ID."""
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    def get_account_paths(self) -> list[str]:
        """Return sorted list of GnuCash account paths for category dropdown.

        Prefer re-loading from the GnuCash book on each call using the gnucash_path
        from run_config metadata, falling back to cached account_paths metadata when
        the book cannot be loaded.
        """
        run_meta = self._state.load_metadata("run_config")
        gnucash_path = (run_meta or {}).get("gnucash_path")
        if gnucash_path:
            try:
                loader = GnuCashLoader()
                txs = loader.load_transactions(Path(gnucash_path))
                # GnuCashLoader exposes account paths via its internal mapping after load.
                account_paths = sorted({s.account_path for tx in txs for s in tx.splits})
                if account_paths:
                    return account_paths
            except Exception:
                logger.warning("Failed to reload account paths from GnuCash book", exc_info=True)
        meta = self._state.load_metadata("account_paths")
        if not meta or "paths" not in meta:
            return []
        return list(meta.get("paths", []))

    @property
    def llm_enabled(self) -> bool:
        """Return True when the main LLM is configured for this review session."""
        return self._llm_config.mode != LlmMode.DISABLED

    def all_proposals(self) -> list[Proposal]:
        return list(self._proposals)

    def get_proposal_by_tx(self, tx_id: str) -> Proposal | None:
        for p in self._proposals:
            if p.tx_id == tx_id:
                return p
        return None

    def queue_ordered_proposals(self) -> list[Proposal]:
        """Proposals in queue order (by date) for navigation and display."""
        return _queue_order(self._proposals)

    def get_next_proposal_id(self, current_proposal_id: str) -> str | None:
        """Return the next proposal ID in queue order, or None if at end."""
        ordered = self.queue_ordered_proposals()
        for i, p in enumerate(ordered):
            if p.proposal_id == current_proposal_id:
                if i + 1 < len(ordered):
                    return ordered[i + 1].proposal_id
                return None
        return None

    def get_prev_proposal_id(self, current_proposal_id: str) -> str | None:
        """Return the previous proposal ID in queue order, or None if at start."""
        ordered = self.queue_ordered_proposals()
        for i, p in enumerate(ordered):
            if p.proposal_id == current_proposal_id:
                if i > 0:
                    return ordered[i - 1].proposal_id
                return None
        return None

    def approved_decisions(self) -> list[ReviewDecision]:
        """Decisions that resulted in approve or edit (for display in queue)."""
        decisions = self._state.load_decisions()
        return [d for d in decisions if d.action in ("approve", "edit")]

    def get_email_category_hint(
        self, sender: str, subject: str, body: str, account_paths: list[str] | None = None
    ) -> str:
        """Suggest a category from email content (keyword heuristic) for UI hint.
        If account_paths is provided, may return a leaf from that list under the heuristic."""
        predictor = CategoryPredictor(
            llm_config=self._llm_config,
            prompts_dir=self._state.state_dir / "prompts",
        )
        return predictor.suggest_category_from_email(sender, subject, body, account_paths)

    def run_llm_check(
        self, proposal_id: str, selected_email_ids: list[str] | None = None
    ) -> dict | None:
        """Run extraction + category LLM for this proposal; update and persist proposal; return result dict or None.
        If selected_email_ids is provided, only those emails are used for the LLM; otherwise all matched emails are used.
        Works with or without evidence: with emails uses per-email extraction then merge; without emails
        uses extraction (or main) LLM on the transaction description to infer supplier/items, then category steps."""
        try:
            proposal = self.get_proposal(proposal_id)
            if not proposal:
                return None
            if self._llm_config.mode == LlmMode.DISABLED:
                return None
            account_paths = self.get_account_paths()
            if not account_paths:
                return None
            tx = Transaction(
                tx_id=proposal.tx_id,
                posted_date=proposal.tx_date or date.today(),
                description=proposal.original_description,
                currency="GBP",
                amount=proposal.tx_amount or Decimal(0),
            )
            predictor = CategoryPredictor(
                historical_transactions=[],
                llm_config=self._llm_config,
                prompts_dir=self._state.state_dir / "prompts",
            )
            all_emails = (proposal.evidence.emails if proposal.evidence else []) or []
            if selected_email_ids is not None:
                id_set = set(selected_email_ids)
                emails = [e for e in all_emails if e.evidence_id in id_set]
            else:
                emails = all_emails
            result = predictor.run_llm_check(
                tx,
                emails,
                proposal.evidence.receipt if proposal.evidence else None,
                account_paths,
            )
            if not result:
                return None
            # Update proposal with LLM result (do not overwrite ML suggested_description / suggested_splits)
            updates: dict = {
                "extraction_result": result.get("extraction"),
                "llm_confidence": result.get("confidence", 0.0),
                "llm_category": result.get("category") or "",
                "llm_description": result.get("description") or "",
            }
            updated = dataclasses.replace(proposal, **updates)
            for i, p in enumerate(self._proposals):
                if p.proposal_id == proposal_id:
                    self._proposals[i] = updated
                    break
            self._state.save_proposals(self._proposals)
            return result
        except Exception:
            logger.warning("run_llm_check failed for proposal %s", proposal_id, exc_info=True)
            return None

    def submit_decision(self, decision: ReviewDecision) -> None:
        """Persist a review decision and update internal state."""
        if not decision.decided_at:
            decision = ReviewDecision(
                tx_id=decision.tx_id,
                action=decision.action,
                final_description=decision.final_description,
                final_splits=decision.final_splits,
                reviewer_note=decision.reviewer_note,
                decided_at=datetime.now(timezone.utc),
                approved_email_ids=decision.approved_email_ids,
                approved_receipt=decision.approved_receipt,
            )

        if decision.action in ("approve", "edit") and (
            decision.approved_email_ids or decision.approved_receipt
        ):
            try:
                decision = self._enrich_from_approved_evidence(decision)
            except Exception:
                logger.warning("Evidence enrichment failed for tx %s; saving un-enriched decision", decision.tx_id, exc_info=True)

        self._state.save_decision(decision)
        self._decided_ids.add(decision.tx_id)

        proposal = self.get_proposal_by_tx(decision.tx_id)
        if proposal:
            self._feedback.record_feedback(
                proposal,
                accepted=decision.action in ("approve", "edit"),
                note=decision.reviewer_note,
            )

        if decision.action == "skip":
            self._state.save_skip(SkipRecord(
                tx_id=decision.tx_id,
                reason=decision.reviewer_note or "Skipped during review",
                skipped_at=datetime.now(timezone.utc),
            ))

        logger.info("Decision recorded: tx=%s action=%s", decision.tx_id, decision.action)

    def _enrich_from_approved_evidence(self, decision: ReviewDecision) -> ReviewDecision:
        """Enrich the decision description using only user-approved evidence."""
        proposal = self.get_proposal_by_tx(decision.tx_id)
        if not proposal or not proposal.evidence:
            return decision

        approved_emails: list[EmailEvidence] = []
        if decision.approved_email_ids and proposal.evidence.emails:
            id_set = set(decision.approved_email_ids)
            approved_emails = [e for e in proposal.evidence.emails if e.evidence_id in id_set]

        approved_receipt: ReceiptEvidence | None = None
        if decision.approved_receipt and proposal.evidence.receipt:
            approved_receipt = proposal.evidence.receipt

        if not approved_emails and not approved_receipt:
            return decision

        predictor = CategoryPredictor(
            llm_config=self._llm_config,
            prompts_dir=self._state.state_dir / "prompts",
        )

        if approved_receipt and approved_receipt.line_items:
            from copy import deepcopy
            approved_receipt = deepcopy(approved_receipt)
            expanded = predictor.describe_terse_items(approved_receipt)
            if expanded != [it.description for it in approved_receipt.line_items]:
                for it, desc in zip(approved_receipt.line_items, expanded):
                    it.description = desc

        enriched = predictor.enrich_description_from_evidence(
            decision.final_description, approved_emails, approved_receipt
        )
        return ReviewDecision(
            tx_id=decision.tx_id,
            action=decision.action,
            final_description=enriched,
            final_splits=decision.final_splits,
            reviewer_note=decision.reviewer_note,
            decided_at=decision.decided_at,
            approved_email_ids=decision.approved_email_ids,
            approved_receipt=decision.approved_receipt,
        )

    def is_decided(self, tx_id: str) -> bool:
        return tx_id in self._decided_ids
