from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_rsc_pf_formal_v4_gate0_evidence as orchestrator


def test_orchestrator_runs_the_exact_nine_stage_order_without_gate1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []
    def fake_inventory(root: Path):
        (root / "audit").mkdir(parents=True, exist_ok=True)
        (root / "audit" / "RECEIPT_INVENTORY.json").write_text("{}", encoding="utf-8")
        return {"status": "pass"}

    monkeypatch.setattr(orchestrator, "_inventory", fake_inventory)

    def stage_runner(stage: str, root: Path) -> None:
        seen.append(stage)
        root.mkdir(parents=True, exist_ok=True)

    summary = orchestrator.run_evidence_orchestrator(
        reports_root=tmp_path / "reports",
        diffopt_python=Path("D:/isolated/python.exe"),
        main_python=Path("D:/main/python.exe"),
        data_dir=tmp_path / "dataset",
        stage_runner=stage_runner,
    )
    assert seen == list(orchestrator.STAGE_ORDER)
    assert summary["status"] == "pass"
    assert summary["test_set_accessed"] is False
    assert not (Path(summary["run_root"]) / "gate0" / "GATE0_AUTHORIZATION.json").exists()


def test_orchestrator_stops_after_first_failed_stage_and_keeps_partial_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []
    failure_stage = orchestrator.STAGE_ORDER[3]

    def stage_runner(stage: str, root: Path) -> None:
        seen.append(stage)
        root.mkdir(parents=True, exist_ok=True)
        if stage == failure_stage:
            raise RuntimeError("synthetic producer failure")

    with pytest.raises(RuntimeError, match="synthetic producer failure"):
        orchestrator.run_evidence_orchestrator(
            reports_root=tmp_path / "reports",
            diffopt_python=Path("D:/isolated/python.exe"),
            main_python=Path("D:/main/python.exe"),
            data_dir=tmp_path / "dataset",
            stage_runner=stage_runner,
        )
    assert seen == list(orchestrator.STAGE_ORDER[:4])
    roots = list((tmp_path / "reports").iterdir())
    assert len(roots) == 1
    root = roots[0]
    assert (root / "ORCHESTRATOR_FAILURE.json").is_file()
    assert not any(path.name == "GATE0_AUTHORIZATION.json" for path in root.rglob("*"))
