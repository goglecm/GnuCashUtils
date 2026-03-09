"""Flask-based one-by-one transaction review web application."""

from __future__ import annotations

import logging
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

from datetime import date

from flask import Flask, jsonify, redirect, render_template, request, url_for

from gnc_enrich.config import ReviewConfig
from gnc_enrich.domain.models import ReviewAction, ReviewDecision, Split
from gnc_enrich.ml.predictor import CategoryPredictor
from gnc_enrich.review.service import ReviewQueueService

logger = logging.getLogger(__name__)


def _serialize_extraction_for_json(extraction: dict) -> dict:
    """Make extraction dict JSON-serializable (Decimal -> str; recurse into nested dicts/lists)."""

    def _serialize_value(val):  # noqa: C901
        if isinstance(val, Decimal):
            return str(val)
        if isinstance(val, dict):
            return {kk: _serialize_value(vv) for kk, vv in val.items()}
        if isinstance(val, list):
            return [_serialize_value(x) for x in val]
        return val

    return {k: _serialize_value(v) for k, v in extraction.items()}


def _money_filter(val) -> str:
    """Format a monetary amount with exactly two decimal places for display."""
    if val is None:
        return ""
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val)


def create_app(service: ReviewQueueService) -> Flask:
    """Create and configure the Flask application for transaction review."""
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.secret_key = os.urandom(32)
    app.jinja_env.filters["money"] = _money_filter

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
        next_id = service.get_next_proposal_id(proposal_id)
        prev_id = service.get_prev_proposal_id(proposal_id)
        account_paths = service.get_account_paths()
        emails_for_display: list[tuple] = []
        email_category_hints: list[str] = []
        if proposal.evidence and proposal.evidence.emails:
            amount = proposal.tx_amount if proposal.tx_amount is not None else Decimal(0)
            try:
                emails_for_display = CategoryPredictor.get_emails_for_display(
                    proposal.evidence.emails, amount
                )
                email_category_hints = [
                    service.get_email_category_hint(em.sender, em.subject, display_ctx, account_paths)
                    for em, display_ctx in emails_for_display
                ]
            except Exception:
                logger.warning("Failed to build emails for display for proposal %s", proposal_id, exc_info=True)
        has_description = bool(proposal.original_description)
        has_emails = bool(emails_for_display)
        has_receipt = bool(proposal.evidence and proposal.evidence.receipt)
        show_llm_button = (not decided) and service.llm_enabled and (has_description or has_emails or has_receipt)
        return render_template(
            "review.html",
            proposal=proposal,
            decided=decided,
            pending=service.pending_count,
            total=service.total_count,
            next_proposal_id=next_id,
            prev_proposal_id=prev_id,
            emails_for_display=emails_for_display,
            email_category_hints=email_category_hints,
            account_paths=account_paths,
            show_llm_button=show_llm_button,
        )

    @app.route("/review/<proposal_id>/decide", methods=["POST"])
    def decide(proposal_id: str):
        proposal = service.get_proposal(proposal_id)
        if proposal is None:
            return redirect(url_for("index"))

        raw_action = request.form.get("action", "skip")
        try:
            action = ReviewAction.validate(raw_action)
        except ValueError as e:
            return (
                str(e),
                400,
                {"Content-Type": "text/plain; charset=utf-8"},
            )
        description = request.form.get("description", proposal.suggested_description)
        note = request.form.get("note", "")

        approved_email_ids = request.form.getlist("approved_email")
        approved_receipt = "approved_receipt" in request.form

        split_paths = request.form.getlist("split_path")
        split_paths_new = request.form.getlist("split_path_new")
        split_amounts = request.form.getlist("split_amount")
        final_splits: list[Split] = []
        for i, amt in enumerate(split_amounts):
            path = (split_paths[i] if i < len(split_paths) else "").strip()
            if not path and i < len(split_paths_new):
                path = (split_paths_new[i] or "").strip()
            if not path:
                continue
            try:
                final_splits.append(Split(account_path=path, amount=Decimal(amt)))
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
        try:
            service.submit_decision(decision)
        except Exception:
            logger.exception("Failed to save decision for proposal %s", proposal_id)
            return (
                "Failed to save your decision. Please try again or check the state directory.",
                503,
                {"Content-Type": "text/plain; charset=utf-8"},
            )
        return redirect(url_for("index"))

    @app.route("/review/<proposal_id>/llm-check", methods=["POST"])
    def llm_check(proposal_id: str):
        """Run extraction + category LLM for this proposal; return JSON result.
        Request body may include selected_email_ids: list of evidence_id to use (only ticked emails)."""
        proposal = service.get_proposal(proposal_id)
        if proposal is None:
            return jsonify({"ok": False, "error": "Proposal not found"}), 404
        selected_email_ids: list[str] | None = None
        if request.is_json:
            data = request.get_json(silent=True) or {}
            raw_ids = data.get("selected_email_ids")
            if isinstance(raw_ids, list):
                selected_email_ids = [str(x) for x in raw_ids]
            elif raw_ids is not None:
                # Badly-typed value provided; treat as "no selected emails"
                selected_email_ids = []
        result = service.run_llm_check(proposal_id, selected_email_ids=selected_email_ids)
        if result is None:
            return jsonify({"ok": False, "error": "LLM disabled or check failed"}), 400
        extraction = result.get("extraction")
        if extraction is not None:
            try:
                extraction = _serialize_extraction_for_json(extraction)
            except Exception:
                logger.warning("Failed to serialize extraction for JSON", exc_info=True)
                extraction = None
        return jsonify({
            "ok": True,
            "extraction": extraction,
            "category": result.get("category", ""),
            "description": result.get("description", ""),
            "confidence": result.get("confidence", 0.0),
        })

    @app.route("/queue")
    def queue():
        ordered = service.queue_ordered_proposals()
        decided_ids = {p.tx_id for p in ordered if service.is_decided(p.tx_id)}
        approved = service.approved_decisions()
        return render_template(
            "queue.html",
            proposals=ordered,
            decided_ids=decided_ids,
            approved_decisions=approved,
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
