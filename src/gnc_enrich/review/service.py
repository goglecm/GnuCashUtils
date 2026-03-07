"""Review workflow service managing proposal queue and decision capture."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from gnc_enrich.domain.models import Proposal, ReviewDecision, SkipRecord
from gnc_enrich.state.repository import StateRepository

logger = logging.getLogger(__name__)


class ReviewQueueService:
    """Serves one-by-one proposals to the reviewer and stores decisions."""

    def __init__(self, state_repo: StateRepository) -> None:
        self._state = state_repo
        self._proposals: list[Proposal] = []
        self._decided_ids: set[str] = set()
        self._reload()

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

    def get_proposal_by_tx(self, tx_id: str) -> Proposal | None:
        for p in self._proposals:
            if p.tx_id == tx_id:
                return p
        return None

    def all_proposals(self) -> list[Proposal]:
        return list(self._proposals)

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
            )

        self._state.save_decision(decision)
        self._decided_ids.add(decision.tx_id)

        if decision.action == "skip":
            self._state.save_skip(SkipRecord(
                tx_id=decision.tx_id,
                reason=decision.reviewer_note or "Skipped during review",
                skipped_at=datetime.now(timezone.utc),
            ))

        logger.info("Decision recorded: tx=%s action=%s", decision.tx_id, decision.action)

    def is_decided(self, tx_id: str) -> bool:
        return tx_id in self._decided_ids
