"""Receipt-to-transaction matching logic."""

from gnc_enrich.domain.models import ReceiptEvidence, Transaction


class ReceiptMatcher:
    """Produces best one-to-one receipt candidate for a transaction."""

    def match(self, tx: Transaction) -> ReceiptEvidence | None:
        raise NotImplementedError
