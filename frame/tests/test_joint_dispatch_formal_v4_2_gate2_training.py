from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization
from src.joint_dispatch.formal_v4_2_gate2_training import (
    iter_gate2_batches,
    load_gate2_data,
    train_direct_policy,
    train_differentiable_lp,
    train_official_itransformer_pto,
    train_rsc_family,
)
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit
from src.joint_dispatch.formal_v4_models import DirectPolicyModel
from src.models import Scheme2RModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json"
FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)


def _split(times: np.ndarray, split: str) -> FormalV4WindowSplit:
    n = len(times)
    renewable_history = np.ones((n, 24, 2), dtype=np.float64)
    return FormalV4WindowSplit(
        load_history=np.ones((n, 24, 4)),
        exog_history=np.ones((n, 24, 12)),
        renewable_history=renewable_history,
        device_history=np.ones((n, 24, 17)),
        activity_history=np.zeros((n, 24, 6)),
        forecast_target=np.ones((n, 4, 4)),
        rigid_demand=np.ones((n, 4, 3)),
        renewable_forecast=np.ones((n, 4, 2)),
        renewable_realized=np.ones((n, 4, 2)),
        prices_and_weights=np.ones((n, 4, 3)),
        initial_soc=np.full((n, 1), 0.5),
        previous_chp=np.zeros((n, 1)),
        target_times=times,
        trajectory_ids=np.asarray([f"{split}-{i}" for i in range(n)]),
        state_hashes=np.asarray([f"state-{split}-{i}" for i in range(n)]),
        split=split,
    )


def _save_split(path: Path, split: FormalV4WindowSplit) -> None:
    np.savez_compressed(path, **{name: getattr(split, name) for name in FIELDS}, split=np.asarray(split.split))


def _run_root(tmp_path: Path):
    contract = load_formal_v4_2_contract(CONFIG)
    root = tmp_path / "run"
    gate1 = root / "gate1"
    protocol = root / "protocol"
    gate1.mkdir(parents=True)
    protocol.mkdir(parents=True)
    train_times = np.asarray([
        "2015-01-01T00:00", "2016-01-01T00:00",
        "2017-01-01T00:00", "2018-01-01T00:00",
    ], dtype="datetime64[ns]")
    selection_times = np.datetime64("2019-01-01T00:00") + np.arange(1000).astype("timedelta64[h]")
    train = _split(train_times, "train")
    selection = _split(selection_times, "selection")
    _save_split(gate1 / "TRAIN_WINDOWS.npz", train)
    _save_split(gate1 / "SELECTION_WINDOWS.npz", selection)
    normalization = fit_train_normalization(train)
    write_once_json(gate1 / "NORMALIZATION.json", normalization.to_payload())
    write_once_json(gate1 / "GATE1_ORIGIN_MANIFEST.json", {"origin_indices": list(range(1000))})
    evidence = {
        "evaluation_year_accessed": False,
        "train_windows_sha256": sha256_file(gate1 / "TRAIN_WINDOWS.npz"),
        "selection_windows_sha256": sha256_file(gate1 / "SELECTION_WINDOWS.npz"),
    }
    write_once_json(gate1 / "GATE1_EVIDENCE.json", evidence)
    write_once_json(protocol / "GATE1_TRANSITION.json", {
        "authorized_gate2": True,
        "contract_sha256": contract.contract_sha256,
        "gate1_evidence_sha256": sha256_file(gate1 / "GATE1_EVIDENCE.json"),
    })
    return root, contract


def test_gate2_uses_all_eligible_train_and_evaluation_windows(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    bundle = load_gate2_data(root, contract)
    assert len(bundle.train) == 4
    assert len(bundle.calibration) == 1000
    assert len(bundle.evaluation) == 1000
    assert np.all(np.diff(bundle.evaluation.target_times) == np.timedelta64(1, "h"))


def test_micro_batches_preserve_effective_batch(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    bundle = load_gate2_data(root, contract)
    parts = list(iter_gate2_batches(
        bundle.evaluation,
        bundle.normalization,
        effective_batch_size=64,
        micro_batch_size=8,
    ))
    first = [part for part in parts if part.effective_batch_index == 0]
    assert len(first) == 8
    assert sum(part.accumulation_weight for part in first) == pytest.approx(1.0)
    assert sum(len(part.indices) for part in parts) == len(bundle.evaluation)


def test_gate2_rejects_gate1_contract_mismatch(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    transition = root / "protocol" / "GATE1_TRANSITION.json"
    payload = json.loads(transition.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    transition.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="contract lineage"):
        load_gate2_data(root, contract)


def _parameters() -> dict[str, float]:
    return {
        "grid_import_capacity": 20.0, "chp_electric_capacity": 10.0,
        "chp_heat_capacity": 12.0, "gas_boiler_capacity": 20.0,
        "electric_chiller_capacity": 20.0, "absorption_chiller_capacity": 20.0,
        "bess_power_capacity": 5.0, "bess_energy_capacity": 20.0,
        "chp_electric_efficiency": 0.4, "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.0,
        "absorption_chiller_cop": 0.8, "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1.0e-6, "unserved_penalty": 100.0,
        "chp_ramp_fraction": 1.0, "surplus_penalty": 0.1,
        "grid_energy_price": 1.0, "gas_energy_price": 1.0,
        "carbon_price": 0.0,
    }


def _tiny_bundle(tmp_path: Path):
    root, contract = _run_root(tmp_path)
    data = load_gate2_data(root, contract)
    # Four train examples are sufficient to prove the execution identities.
    return data, {
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": "1" * 64,
        "selected_candidate_value": 1.0,
    }


def _tiny_budget() -> StageBudgetV42:
    return StageBudgetV42(max_epochs=1, minimum_epochs=1, ramp_epochs=1)


def test_rsc_pair_has_identical_stage_s_parent(tmp_path: Path) -> None:
    data, freeze = _tiny_bundle(tmp_path)
    teacher = np.zeros((len(data.train), 4, 21), dtype=np.float64)
    rows = train_rsc_family(
        2026, data, freeze, _parameters(), tmp_path / "rsc",
        budget=_tiny_budget(), teacher_dispatch=teacher,
    )
    assert rows["RSC-PF"].stage_s_parent_sha256 == rows["Decoupled-RSC-PF"].stage_s_parent_sha256
    assert rows["RSC-PF"].decision_forecaster_gradient_norm > 0.0
    assert rows["Decoupled-RSC-PF"].decision_forecaster_gradient_norm == 0.0
    assert "State-Conditioned-PTO" in rows


def test_direct_policy_has_no_forecast_training(tmp_path: Path) -> None:
    data, freeze = _tiny_bundle(tmp_path)
    teacher = np.zeros((len(data.train), 4, 21), dtype=np.float64)
    row = train_direct_policy(
        2026, data, freeze, _parameters(), tmp_path / "direct",
        budget=_tiny_budget(), teacher_dispatch=teacher,
    )
    assert row.training_receipt["forecast_loss_applicable"] is False
    assert row.training_receipt["optimizer_steps"] > 0


def test_direct_policy_projection_preserves_chp_ramp_feasibility_and_gradient() -> None:
    parameters = _parameters()
    parameters["chp_ramp_fraction"] = 0.2
    model = DirectPolicyModel(decoder_parameters=parameters, dropout=0.0)
    planning = torch.zeros((2, 4, 3), dtype=torch.float32, requires_grad=True)
    previous_chp = torch.tensor([[9.0], [0.0]], dtype=torch.float64)
    projected = model._project_planning_demand(planning, previous_chp)
    # The first sample must retain enough electric demand for the 9 -> 7
    # ramp-down interval; the second sample remains at the numerical margin.
    assert projected[0, 0, 0].item() >= 7.0
    assert projected[1, 0, 0].item() > 0.0
    projected.sum().backward()
    assert planning.grad is not None
    assert torch.isfinite(planning.grad).all().item()


class _TinyITransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.projection(history[:, -4:, :])


def test_itransformer_records_upstream_adaptation(tmp_path: Path) -> None:
    data, freeze = _tiny_bundle(tmp_path)
    receipt = {"commit": "c2426e68ca13f74aaec08045c5c724d8ad328124"}
    row = train_official_itransformer_pto(
        2026, data, freeze, receipt, tmp_path / "itransformer",
        budget=_tiny_budget(), model=_TinyITransformer(),
    )
    assert row.training_receipt["upstream_commit"] == receipt["commit"]
    assert row.training_receipt["method_label"] == "official_backbone_adaptation"


class _TinyDiffLayer(nn.Module):
    def forward(
        self,
        demand: torch.Tensor,
        renewable: torch.Tensor,
        prices: torch.Tensor,
        initial_soc: torch.Tensor,
        previous_chp: torch.Tensor,
    ) -> torch.Tensor:
        zero = demand[..., 0] * 0.0
        fields = [zero for _ in range(21)]
        fields[0] = demand[..., 0]
        fields[2] = renewable[..., 0]
        fields[4] = renewable[..., 1]
        fields[18] = demand[..., 1]
        fields[19] = demand[..., 2]
        return torch.stack(fields, dim=-1).to(torch.float64)


def test_diff_lp_training_is_real_and_complete(tmp_path: Path) -> None:
    data, freeze = _tiny_bundle(tmp_path)
    row = train_differentiable_lp(
        2026, data, freeze, {"eligible_for_gate0": True}, _parameters(),
        tmp_path / "difflp", budget=_tiny_budget(),
        model=Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0),
        layer=_TinyDiffLayer(), micro_batch_size=1,
    )
    evidence = row.training_receipt
    assert evidence["gradient_norm"] > 0.0
    assert evidence["optimizer_steps"] > 0
    assert evidence["sample_exposures"] == evidence["expected_sample_exposures"]
    assert evidence["failed_solves"] == 0
    assert evidence["effective_batch_size"] == 64
    assert evidence["training_solver_calls"] > 0
    assert evidence["solver_inaccurate_warning_count"] == 0
    assert evidence["solver_quality_status"] == "no_inaccurate_warning"
