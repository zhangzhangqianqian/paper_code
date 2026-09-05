from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_4_pilot_materializer import V44WindowCollection
from src.joint_dispatch.formal_v4_4_teacher import build_same_information_teacher_v44
from src.joint_dispatch.formal_v4_4_pilot_materializer import MaterializedV44PilotData
from src.scheduling.dispatch_schema import VARIABLES


class _ForecastModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, **inputs):
        assert "target_physical" not in inputs
        batch = inputs["load_history"].shape[0]
        return SimpleNamespace(forecast_physical=torch.ones(batch, 4, 4))


class _Result:
    success = True
    status = "optimal"
    objective = 1.0
    message = ""

    def __init__(self):
        self.values = {name: np.zeros(4, dtype=np.float64) for name in VARIABLES}


def _teacher_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.test_joint_dispatch_formal_v4_4_pilot_materializer import _sources
    from src.joint_dispatch.formal_v4_4_pilot_materializer import materialize_v44_pilot_data

    sources = _sources(tmp_path)
    for path in (sources["train_data"], sources["selection_data"]):
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        arrays["load_and_exog"] = arrays["load_and_exog"].copy()
        arrays["load_and_exog"][:, 2] = 0.0
        np.savez_compressed(path, **arrays)
    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_pilot_materializer.generate_settled_device_trajectory", lambda base, parameters, *, capacity_receipt, trajectory_id: SimpleNamespace(
        settled_dispatch=np.zeros((len(base.timestamps), 21)), settled_mask=np.ones(len(base.timestamps), dtype=bool), trajectory_sha256="b" * 64, trajectory_id=trajectory_id,
    ))
    materialized = materialize_v44_pilot_data(**sources, artifact_root=tmp_path / "data_run")
    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_teacher.solve_dispatch_lp", lambda _inputs: _Result())
    return sources, materialized


def test_teacher_uses_forecasts_not_realized_future_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources, materialized = _teacher_fixture(tmp_path, monkeypatch)
    indices = np.arange(len(materialized.train), dtype=np.int64)
    first = build_same_information_teacher_v44(
        model=_ForecastModel(), materialized=materialized, indices=indices,
        benchmark=sources["benchmark"], capacity_receipt=sources["capacity_receipt"], artifact_root=tmp_path / "teacher_a",
        contract=sources["contract"], seed=2026,
    )
    changed_split = replace(materialized.train.split, forecast_target=materialized.train.forecast_target + 1000.0)
    changed = replace(materialized, train=V44WindowCollection(changed_split, "train"))
    second = build_same_information_teacher_v44(
        model=_ForecastModel(), materialized=changed, indices=indices,
        benchmark=sources["benchmark"], capacity_receipt=sources["capacity_receipt"], artifact_root=tmp_path / "teacher_b",
        contract=sources["contract"], seed=2026,
    )
    np.testing.assert_allclose(first.dispatch, second.dispatch)


def test_teacher_rejects_failed_or_nonfinite_lp_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources, materialized = _teacher_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_teacher.solve_dispatch_lp", lambda _inputs: None)
    with pytest.raises(RuntimeError, match="LP"):
        build_same_information_teacher_v44(
            model=_ForecastModel(), materialized=materialized, indices=np.asarray([0]),
            benchmark=sources["benchmark"], capacity_receipt=sources["capacity_receipt"], artifact_root=tmp_path / "teacher_failed",
            contract=sources["contract"], seed=2026,
        )
