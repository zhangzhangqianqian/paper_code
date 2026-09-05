from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.joint_dispatch.formal_v4_4_pilot_materializer import materialize_v44_pilot_data


def _base_arrays(years: tuple[int, ...], hours: int) -> tuple[np.ndarray, ...]:
    times = np.concatenate([
        np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year}-01-01") + np.timedelta64(hours, "h"), np.timedelta64(1, "h"))
        for year in years
    ]).astype("datetime64[ns]")
    load = np.zeros((len(times), 16), dtype=np.float64)
    load[:, 0] = 10.0
    load[:, 1] = 4.0
    load[:, 2] = 3.0
    load[:, 3] = 1.0
    return load, np.zeros((len(times), 2)), np.zeros((len(times), 2)), np.zeros((len(times), 3)), times


def _sources(tmp_path: Path) -> dict[str, object]:
    train = tmp_path / "train.npz"
    selection = tmp_path / "selection.npz"
    for path, years, label in ((train, (2015, 2016, 2017, 2018), "train"), (selection, (2019,), "selection")):
        arrays = _base_arrays(years, 40)
        np.savez_compressed(path, load_and_exog=arrays[0], renewable_forecast=arrays[1], renewable_realized=arrays[2], prices_and_weights=arrays[3], timestamps=arrays[4], split=np.asarray(label))
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(yaml.safe_dump({"values": {"bess_energy_capacity": 10.0}}), encoding="utf-8")
    capacity = tmp_path / "capacity.json"
    capacity.write_text('{"gate0_authorized": true, "capacity_multiplier": 1.0}', encoding="utf-8")
    contract = SimpleNamespace(validate=lambda: None, contract_sha256="a" * 64, pilot_train_windows=2)
    return {
        "train_data": train,
        "selection_data": selection,
        "benchmark": benchmark,
        "capacity_receipt": capacity,
        "split": {
            "train": np.asarray([0, 1]),
            "early_stop": np.asarray([2]),
            "selection_full": np.arange(13),
            "selection_stress": np.asarray([0]),
        },
        "contract": contract,
    }


def test_materializer_keeps_device_history_and_split_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _sources(tmp_path)

    def fake_trajectory(base, parameters, *, capacity_receipt, trajectory_id):
        return SimpleNamespace(
            settled_dispatch=np.zeros((len(base.timestamps), 21), dtype=np.float64),
            settled_mask=np.ones(len(base.timestamps), dtype=bool),
            trajectory_sha256=f"{trajectory_id:0<64}"[:64],
            trajectory_id=trajectory_id,
        )

    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_pilot_materializer.generate_settled_device_trajectory", fake_trajectory)
    result = materialize_v44_pilot_data(**sources, artifact_root=tmp_path / "run")
    assert result.train.load_history.shape[1:] == (24, 4)
    assert result.train.device_history.shape[1:] == (24, 17)
    assert result.train.activity_history.shape[1:] == (24, 6)
    assert set(result.selection.timestamps.astype("datetime64[Y]").astype(int) + 1970) == {2019}
    assert len(result.normalization_source) > len(result.train)


def test_materializer_rejects_mismatched_resume_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = _sources(tmp_path)

    def fake_trajectory(base, parameters, *, capacity_receipt, trajectory_id):
        return SimpleNamespace(
            settled_dispatch=np.zeros((len(base.timestamps), 21), dtype=np.float64),
            settled_mask=np.ones(len(base.timestamps), dtype=bool),
            trajectory_sha256="b" * 64,
            trajectory_id=trajectory_id,
        )

    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_pilot_materializer.generate_settled_device_trajectory", fake_trajectory)
    run = tmp_path / "run"
    materialize_v44_pilot_data(**sources, artifact_root=run)
    other_capacity = tmp_path / "other_capacity.json"
    other_capacity.write_text('{"gate0_authorized": true, "capacity_multiplier": 1.1}', encoding="utf-8")
    with pytest.raises(ValueError, match="lineage"):
        materialize_v44_pilot_data(**{**sources, "capacity_receipt": other_capacity}, artifact_root=run)
