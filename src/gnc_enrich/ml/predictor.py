"""ML/AI inference interfaces for category and description proposal."""

from gnc_enrich.domain.models import EmailEvidence, Proposal, ReceiptEvidence, Transaction


class CategoryPredictor:
    """Predicts split categories with confidence and rationale."""

    def propose(
        self,
        tx: Transaction,
        emails: list[EmailEvidence],
        receipt: ReceiptEvidence | None,
    ) -> Proposal:
        raise NotImplementedError


class FeedbackTrainer:
    """Captures user-approved outcomes for future model improvement."""

    def record_feedback(self, proposal: Proposal, accepted: bool, note: str = "") -> None:
        raise NotImplementedError
