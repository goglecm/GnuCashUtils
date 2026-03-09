"""Tests for gnc_enrich.__main__ entrypoint."""

import runpy
import sys
from unittest.mock import patch

import pytest


def test_main_module_calls_main_and_exits() -> None:
    """When run as __main__, the module calls main() and raises SystemExit with its return value."""
    with patch("gnc_enrich.cli.main", return_value=0) as mock_main:
        with patch.object(
            sys,
            "argv",
            [
                "gnc_enrich",
                "run",
                "--gnucash-path",
                "/x",
                "--emails-dir",
                "/e",
                "--receipts-dir",
                "/r",
                "--processed-receipts-dir",
                "/p",
                "--state-dir",
                "/s",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module("gnc_enrich", run_name="__main__")
            assert exc_info.value.code == 0
        mock_main.assert_called_once()
