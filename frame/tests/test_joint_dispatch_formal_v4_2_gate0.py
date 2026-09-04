from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import runpy


PROBE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate0.py"))


def _receipts(tmp_path: Path) -> SimpleNamespace:
    paths = {}
    for name in ("itransformer_receipt", "diffopt_receipt", "capacity_receipt", "source_manifest"):
        path = tmp_path / f"{name}.json"; path.write_text("{}", encoding="utf-8"); paths[name] = path
    return SimpleNamespace(output_root=tmp_path, run_id="formal_v4_2_gate0_test", **paths)


def test_resource_probe_calls_real_components(monkeypatch, tmp_path):
    calls = Counter()
    def counted(name):
        def operation():
            calls[name] += 1
            return {"identity": name}
        return operation
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_rsc_forward", counted("rsc_forward"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_rsc_backward", counted("rsc_backward"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_highs_lp", counted("highs_lp"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_diff_lp", counted("diff_lp"))
    receipt = PROBE["run_gate0_probe"](_receipts(tmp_path))
    assert calls == Counter(rsc_forward=1, rsc_backward=1, highs_lp=1, diff_lp=1)
    assert receipt["synthetic_probe"] is False
    assert receipt["authorized_pilot"] is True


def test_gate0_fails_when_source_or_dependency_receipt_is_missing(tmp_path):
    fixture = _receipts(tmp_path)
    fixture.itransformer_receipt.unlink()
    receipt = PROBE["run_gate0_probe"](fixture)
    assert receipt["authorized_pilot"] is False


def test_gate0_uses_train_years_for_fit_and_never_evaluation(tmp_path):
    receipt = PROBE["run_gate0_probe"](_receipts(tmp_path))
    assert receipt["capacity_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["normalization_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["materialized_years"] == [2015, 2016, 2017, 2018, 2019]
    assert receipt["evaluation_year_accessed"] is False
