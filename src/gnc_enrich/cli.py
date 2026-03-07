"""CLI entrypoints for run/review/apply workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnc_enrich.config import ApplyConfig, ReviewConfig, RunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnc-enrich")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Build candidate proposals from source data")
    run.add_argument("--gnucash-path", type=Path, required=True)
    run.add_argument("--emails-dir", type=Path, required=True)
    run.add_argument("--receipts-dir", type=Path, required=True)
    run.add_argument("--processed-receipts-dir", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--date-window-days", type=int, default=7)
    run.add_argument("--amount-tolerance", type=float, default=0.50)
    run.add_argument("--include-skipped", action="store_true")

    review = sub.add_parser("review", help="Run local web review app")
    review.add_argument("--state-dir", type=Path, required=True)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=7860)

    apply = sub.add_parser("apply", help="Apply approved changes to GnuCash")
    apply.add_argument("--state-dir", type=Path, required=True)
    apply.add_argument("--create-backup", action="store_true")
    apply.add_argument("--backup-dir", type=Path)
    apply.add_argument("--in-place", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _ = RunConfig(
            gnucash_path=args.gnucash_path,
            emails_dir=args.emails_dir,
            receipts_dir=args.receipts_dir,
            processed_receipts_dir=args.processed_receipts_dir,
            state_dir=args.state_dir,
            date_window_days=args.date_window_days,
            amount_tolerance=args.amount_tolerance,
            include_skipped=args.include_skipped,
        )
        return 0

    if args.command == "review":
        _ = ReviewConfig(state_dir=args.state_dir, host=args.host, port=args.port)
        return 0

    if args.command == "apply":
        _ = ApplyConfig(
            state_dir=args.state_dir,
            create_backup=args.create_backup,
            backup_dir=args.backup_dir,
            in_place=args.in_place,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
