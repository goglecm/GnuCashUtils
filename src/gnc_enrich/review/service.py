"""Review workflow service managing proposal queue and decision capture."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from gnc_enrich.config import LlmConfig, LlmMode
from gnc_enrich.domain.models import EmailEvidence, Proposal, ReceiptEvidence, ReviewDecision, SkipRecord
from gnc_enrich.ml.predictor import CategoryPredictor, FeedbackTrainer
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
        """Return sorted list of GnuCash account paths for category dropdown (from last run)."""
        meta = self._state.load_metadata("account_paths")
        if not meta or "paths" not in meta:
            return []
        return list(meta.get("paths", []))

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

    def get_email_category_hint(self, sender: str, subject: str, body: str) -> str:
        """Suggest a category from email content (keyword heuristic) for UI hint."""
        predictor = CategoryPredictor(llm_config=self._llm_config)
        return predictor.suggest_category_from_email(sender, subject, body)

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
            decision = self._enrich_from_approved_evidence(decision)

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

        predictor = CategoryPredictor(llm_config=self._llm_config)

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
