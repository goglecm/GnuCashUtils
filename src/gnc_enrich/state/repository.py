"""State persistence contracts for indexed artifacts and review state."""

from gnc_enrich.domain.models import Proposal, ReviewDecision


class StateRepository:
    def save_proposals(self, proposals: list[Proposal]) -> None:
        raise NotImplementedError

    def load_proposals(self) -> list[Proposal]:
        raise NotImplementedError

    def save_decision(self, decision: ReviewDecision) -> None:
        raise NotImplementedError

    def load_skipped_ids(self) -> set[str]:
        raise NotImplementedError
