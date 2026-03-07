"""Review workflow service used by the web application."""

from gnc_enrich.domain.models import Proposal, ReviewDecision


class ReviewQueueService:
    """Serves one-by-one proposals to the reviewer and stores decisions."""

    def next_proposal(self) -> Proposal | None:
        raise NotImplementedError

    def submit_decision(self, decision: ReviewDecision) -> None:
        raise NotImplementedError
