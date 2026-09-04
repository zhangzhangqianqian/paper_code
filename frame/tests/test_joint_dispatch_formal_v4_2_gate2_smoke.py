from __future__ import annotations

from pathlib import Path
from runpy import run_path


SMOKE = run_path(str(Path(__file__).parents[1] / "scripts" / "smoke_rsc_pf_formal_v4_2_gate2.py"))


def test_smoke_runs_every_family_without_formal_transition(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    receipt = SMOKE["run_gate2_smoke"](output)
    assert receipt["paper_eligible"] is False
    assert receipt["families"] == ["rsc", "direct", "pto", "itransformer", "diff_lp", "deterministic"]
    assert all(item["finite"] for item in receipt["results"])
    assert not (output / "GATE2_TRANSITION.json").exists()
