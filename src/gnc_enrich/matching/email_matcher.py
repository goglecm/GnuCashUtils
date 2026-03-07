"""Email-to-transaction matching logic."""

from gnc_enrich.domain.models import EmailEvidence, Transaction


class EmailMatcher:
    """Matches transactions to email evidence using date/amount/text signals."""

    def match(self, tx: Transaction) -> list[EmailEvidence]:
        raise NotImplementedError
