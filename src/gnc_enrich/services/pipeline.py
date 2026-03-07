"""High-level orchestration for candidate selection and proposal generation."""

from dataclasses import dataclass

from gnc_enrich.config import RunConfig
from gnc_enrich.domain.models import Proposal


@dataclass(slots=True)
class PipelineResult:
    proposal_count: int
    skipped_count: int


class EnrichmentPipeline:
    """Coordinates loading, matching, inference, and proposal persistence."""

    def run(self, config: RunConfig) -> PipelineResult:
        raise NotImplementedError

    def build_proposals(self, config: RunConfig) -> list[Proposal]:
        raise NotImplementedError
