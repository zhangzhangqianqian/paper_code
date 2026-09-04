from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_repository_regression_v4 import REPO_ROOT, _summary, run_regression


def test_summary_parser_extracts_terminal_counts():
    counts = _summary("900 passed, 2 failed, 5 skipped, 1 xfailed in 1.2s")
    assert counts == {"passed": 900, "failed": 2, "skipped": 5, "xfailed": 1, "xpassed": 0, "errors": 0}


def test_summary_parser_handles_clean_output():
    counts = _summary("42 passed in 0.3s")
    assert counts["passed"] == 42
    assert sum(counts[key] for key in ("failed", "skipped", "xfailed", "xpassed", "errors")) == 0


def test_regression_runner_records_root_and_is_immutable(tmp_path: Path):
    output = tmp_path / "REGRESSION_RECEIPT.json"
    # Use a tiny explicit test selection for a fast contract test; the runner
    # still resolves cwd, PYTHONPATH and paths exactly as the full suite does.
    receipt = run_regression(output_path=output, pytest_args=("frame/tests/test_joint_dispatch_formal_v4_access.py",))
    assert receipt["status"] == "pass"
    assert Path(receipt["working_directory"]).resolve() == REPO_ROOT.resolve()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["output_sha256"] == receipt["output_sha256"]
    with pytest.raises(FileExistsError):
        run_regression(output_path=output, pytest_args=("frame/tests/test_joint_dispatch_formal_v4_access.py",))
