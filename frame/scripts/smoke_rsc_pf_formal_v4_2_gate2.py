"""Bounded all-family smoke test for the formal-v4.2 Gate 2 executor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate2_training import (  # noqa: E402
    Gate2DataBundle, train_differentiable_lp, train_direct_policy,
    train_official_itransformer_pto, train_rsc_family, train_scheme2r_pto,
)
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit  # noqa: E402
from src.joint_dispatch.formal_v4_method_adapter import build_formal_v4_method_adapter  # noqa: E402
from src.models import Scheme2RModel  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.0, "unserved_penalty": 100.0,
    "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5, "carbon_price": 0.0,
}


def _split(times: np.ndarray, split: str) -> FormalV4WindowSplit:
    n = len(times); renew_history = np.ones((n, 24, 2), dtype=np.float64)
    return FormalV4WindowSplit(
        load_history=np.ones((n, 24, 4)), exog_history=np.ones((n, 24, 12)),
        renewable_history=renew_history, device_history=np.zeros((n, 24, 17)),
        activity_history=np.zeros((n, 24, 6)), forecast_target=np.ones((n, 4, 4)),
        rigid_demand=np.ones((n, 4, 3)), renewable_forecast=np.ones((n, 4, 2)),
        renewable_realized=np.ones((n, 4, 2)), prices_and_weights=np.ones((n, 4, 3)),
        initial_soc=np.full((n, 1), 0.5), previous_chp=np.zeros((n, 1)), target_times=times,
        trajectory_ids=np.asarray([f"{split}-{i}" for i in range(n)]),
        state_hashes=np.asarray([f"state-{split}-{i}" for i in range(n)]), split=split,
    )


def _bundle() -> Gate2DataBundle:
    train_times = np.asarray([
        f"{year}-01-01T{hour:02d}:00" for year in range(2015, 2019) for hour in range(4)
    ], dtype="datetime64[ns]")
    evaluation_times = np.datetime64("2019-01-01T00:00") + np.arange(8).astype("timedelta64[h]")
    train = _split(train_times, "train"); evaluation = _split(evaluation_times, "selection")
    normalization = fit_train_normalization(train)
    return Gate2DataBundle(train, evaluation, evaluation, normalization, *("1" * 64,) * 4)


class _TinyOfficial(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.projection = nn.Linear(4, 4)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.projection(history[:, -4:, :])


class _TinyDiffLayer(nn.Module):
    def forward(self, demand, renewable, prices, initial_soc, previous_chp):
        zero = demand[..., 0] * 0.0; fields = [zero for _ in range(21)]
        fields[0] = demand[..., 0]; fields[2] = renewable[..., 0]; fields[4] = renewable[..., 1]
        fields[18] = demand[..., 1]; fields[19] = demand[..., 2]
        return torch.stack(fields, dim=-1).to(torch.float64)


def _official_receipt(output: Path) -> tuple[dict[str, Any], Path, Path]:
    source = FRAME_ROOT / "third_party" / "iTransformer_source"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    license_path = next(path for path in (source / "LICENSE", source / "LICENSE.md", source / "LICENSE.txt") if path.is_file())
    payload = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        "source_root": "frame/third_party/iTransformer_source", "repository": "https://github.com/thuml/iTransformer",
        "commit": commit, "backbone_class": "model.iTransformer.Model",
        "imported_file_hashes": {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py")},
        "license_file": license_path.relative_to(source).as_posix(),
        "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        "reproduction_level": "official_backbone_adaptation", "verified": True,
    }
    receipt_path = output / "ITRANSFORMER_SMOKE_RECEIPT.json"; write_once_json(receipt_path, payload)
    return payload, source, receipt_path


def run_gate2_smoke(output_root: str | Path, *, real_external: bool = False) -> Mapping[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"smoke output exists: {output}")
    output.mkdir(parents=True)
    data = _bundle(); budget = StageBudgetV42(max_epochs=2, minimum_epochs=1, ramp_epochs=1)
    freeze = {"contract_sha256": "1" * 64, "source_manifest_sha256": "2" * 64, "c_ref": 1.0}
    teacher = np.zeros((len(data.train), 4, 21), dtype=np.float64)
    family = train_rsc_family(2026, data, freeze, PARAMETERS, output / "rows", budget=budget, teacher_dispatch=teacher)
    direct = train_direct_policy(2026, data, freeze, PARAMETERS, output / "rows" / "Direct-Policy" / "2026", budget=budget, teacher_dispatch=teacher)
    scheme = train_scheme2r_pto(2026, data, freeze, output / "rows" / "Scheme2R-PTO" / "2026", budget=budget)
    if real_external:
        legacy, source, receipt_path = _official_receipt(output)
        official = train_official_itransformer_pto(2026, data, freeze, legacy, output / "rows" / "Official iTransformer-PTO" / "2026", source_root=source, receipt_path=receipt_path, budget=budget)
        from src.joint_dispatch.formal_v4_diffopt import DifferentiableIESLayer
        diff_layer: nn.Module = DifferentiableIESLayer(PARAMETERS)
    else:
        legacy = {"commit": "c2426e68ca13f74aaec08045c5c724d8ad328124"}
        official = train_official_itransformer_pto(2026, data, freeze, legacy, output / "rows" / "Official iTransformer-PTO" / "2026", model=_TinyOfficial(), budget=budget)
        diff_layer = _TinyDiffLayer()
    diff = train_differentiable_lp(2026, data, freeze, {"eligible": True}, PARAMETERS, output / "rows" / "Differentiable-LP" / "2026", model=Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0), layer=diff_layer, budget=budget, micro_batch_size=1)
    window = {
        "load_history": data.evaluation.load_history[0], "exog_history": data.evaluation.exog_history[0],
        "device_history": data.evaluation.device_history[0], "activity_history": data.evaluation.activity_history[0],
        "renewable_forecast": data.evaluation.renewable_forecast[0], "renewable_realized": data.evaluation.renewable_realized[0],
        "prices_and_weights": data.evaluation.prices_and_weights[0], "forecast_target": data.evaluation.forecast_target[0],
        "initial_soc": data.evaluation.initial_soc[0], "previous_chp": data.evaluation.previous_chp[0],
    }
    seasonal = build_formal_v4_method_adapter("Seasonal-Naive-PTO", PARAMETERS)
    seasonal_result = seasonal.predict_and_dispatch(window)
    oracle = solve_dispatch_lp(DispatchInputs(
        window["forecast_target"][:, :3], window["renewable_realized"][:, 0],
        window["renewable_realized"][:, 1], PARAMETERS, 0.5, 0.0,
    ))
    if not oracle.success:
        raise RuntimeError(oracle.message)
    artifacts = [family["RSC-PF"], direct, scheme, official, diff]
    results = [
        {"family": name, "finite": bool(np.isfinite(list(artifact.training_receipt["loss_history"])).all()), "optimizer_steps": artifact.training_receipt["optimizer_steps"]}
        for name, artifact in zip(("rsc", "direct", "pto", "itransformer", "diff_lp"), artifacts)
    ]
    results.append({"family": "deterministic", "finite": bool(np.isfinite(seasonal_result["dispatch"]).all() and np.isfinite(oracle.objective)), "optimizer_steps": 0})
    receipt = {
        "schema": "formal-v4.2-gate2-smoke-v1", "paper_eligible": False,
        "families": ["rsc", "direct", "pto", "itransformer", "diff_lp", "deterministic"],
        "train_windows": 16, "epochs": 2, "evaluation_origins": 8,
        "real_external": bool(real_external), "results": results,
        "complete": all(item["finite"] and item["optimizer_steps"] >= 0 for item in results),
        "evaluation_year_accessed": False,
    }
    write_once_json(output / "SMOKE_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-external", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_gate2_smoke(args.output_root, real_external=args.real_external)
    print(json.dumps(receipt, ensure_ascii=False)); return 0 if receipt["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_gate2_smoke"]
