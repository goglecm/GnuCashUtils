"""Flask-based one-by-one transaction review web application."""

from __future__ import annotations

import logging
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

from datetime import date

from flask import Flask, redirect, render_template, request, url_for

from gnc_enrich.config import ReviewConfig
from gnc_enrich.domain.models import ReviewAction, ReviewDecision, Split
from gnc_enrich.review.service import ReviewQueueService

logger = logging.getLogger(__name__)


def create_app(service: ReviewQueueService) -> Flask:
    """Create and configure the Flask application for transaction review."""
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.secret_key = os.urandom(32)

    @app.route("/")
    def index():
        proposal = service.next_proposal()
        if proposal is None:
            return render_template("done.html", total=service.total_count, decided=service.decided_count, pending=0)
        return redirect(url_for("review", proposal_id=proposal.proposal_id))

    @app.route("/review/<proposal_id>")
    def review(proposal_id: str):
        proposal = service.get_proposal(proposal_id)
        if proposal is None:
            return redirect(url_for("index"))

        decided = service.is_decided(proposal.tx_id)
        return render_template(
            "review.html",
            proposal=proposal,
            decided=decided,
            pending=service.pending_count,
            total=service.total_count,
        )

    @app.route("/review/<proposal_id>/decide", methods=["POST"])
    def decide(proposal_id: str):
        proposal = service.get_proposal(proposal_id)
        if proposal is None:
            return redirect(url_for("index"))

        raw_action = request.form.get("action", "skip")
        action = ReviewAction.validate(raw_action)
        description = request.form.get("description", proposal.suggested_description)
        note = request.form.get("note", "")

        approved_email_ids = request.form.getlist("approved_email")
        approved_receipt = "approved_receipt" in request.form

        split_paths = request.form.getlist("split_path")
        split_amounts = request.form.getlist("split_amount")
        final_splits: list[Split] = []
        for path, amt in zip(split_paths, split_amounts):
            if path.strip():
                try:
                    final_splits.append(Split(account_path=path.strip(), amount=Decimal(amt)))
                except (InvalidOperation, ValueError):
                    continue

        if not final_splits:
            final_splits = proposal.suggested_splits

        decision = ReviewDecision(
            tx_id=proposal.tx_id,
            action=action,
            final_description=description,
            final_splits=final_splits,
            reviewer_note=note,
            approved_email_ids=approved_email_ids,
            approved_receipt=approved_receipt,
        )
        service.submit_decision(decision)

        return redirect(url_for("index"))

    @app.route("/queue")
    def queue():
        proposals = service.all_proposals()
        proposals.sort(key=lambda p: p.tx_date or date.min)
        return render_template(
            "queue.html",
            proposals=proposals,
            decided_ids={p.tx_id for p in proposals if service.is_decided(p.tx_id)},
            pending=service.pending_count,
            total=service.total_count,
        )

    return app


class ReviewWebApp:
    """Lifecycle wrapper around the Flask review application."""

    def __init__(self, service: ReviewQueueService, config: ReviewConfig) -> None:
        self._service = service
        self._config = config
        self._app: Flask | None = None

    def get_app(self) -> Flask:
        """Lazily create and return the Flask app instance."""
        if self._app is None:
            self._app = create_app(self._service)
        return self._app

    def run(self, host: str, port: int) -> None:
        """Start the Flask development server."""
        app = self.get_app()
        logger.info("Starting review web app at http://%s:%d", host, port)
        app.run(host=host, port=port, debug=False)
