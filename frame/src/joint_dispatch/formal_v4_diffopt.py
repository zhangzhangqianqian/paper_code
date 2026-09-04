"""CVXPYlayers differentiable-LP baseline for formal-v4.

The module keeps the formulation auditable even when the optional solver stack
is not installed.  Importing this file therefore never silently substitutes a
neural scheduler for the public differentiable-optimization baseline.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


DIFFOPT_METHOD_ID = "Differentiable-LP"
HORIZON = 4
RIGID_TASKS = ("electricity", "cooling", "heating")


@dataclass(frozen=True)
class DifferentiableLPProblemSpec:
    """Dependency-independent contract for the convex dispatch layer."""

    predicted_parameters: tuple[str, ...] = RIGID_TASKS
    horizon: int = HORIZON
    online_optimizer_calls_per_window: int = 1
    deployable: bool = True
    optimizer_at_inference: bool = True
    output_dimension: int = 21
    training_regularizer: float = 0.0
    tie_break_throughput_cost: float = 1.0e-5

    def validate(self) -> None:
        if self.predicted_parameters != RIGID_TASKS:
            raise ValueError("Differentiable-LP must parameterize the three rigid demand RHSs")
        if self.horizon != HORIZON or self.output_dimension != 21:
            raise ValueError("formal-v4 Differentiable-LP contract is fixed at H=4 and 21 outputs")
        if self.online_optimizer_calls_per_window != 1:
            raise ValueError("Differentiable-LP uses exactly one optimizer call per window")
        if not self.deployable or not self.optimizer_at_inference:
            raise ValueError("Differentiable-LP must disclose its optimizer-at-inference status")
        if self.training_regularizer < 0.0:
            raise ValueError("training_regularizer must be non-negative")
        if self.tie_break_throughput_cost < 0.0:
            raise ValueError("tie_break_throughput_cost must be non-negative")


@dataclass(frozen=True)
class DifferentiableLPGateReceipt:
    """Machine-readable eligibility record; false is a hard stop."""

    method_id: str = DIFFOPT_METHOD_ID
    eligible: bool = False
    dpp_passed: bool = False
    finite_solves: int = 0
    parity_passed: bool = False
    physical_residual_passed: bool = False
    gradient_passed: bool = False
    native_probe_passed: bool = False
    memory_margin_fraction: float = 0.0
    projected_p95_hours: float = float("inf")
    source_environment: str = "unresolved"
    reason: str = "solver environment not prepared"

    def validate(self) -> None:
        if self.method_id != DIFFOPT_METHOD_ID:
            raise ValueError("unexpected differentiable-LP method id")
        if self.eligible and not (
            self.dpp_passed and self.finite_solves >= 100 and self.parity_passed
            and self.physical_residual_passed and self.gradient_passed and self.native_probe_passed
            and self.memory_margin_fraction >= 0.20
            and self.projected_p95_hours <= 24.0
        ):
            raise ValueError("eligible differentiable-LP receipt lacks mandatory gate evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_diffopt_gate_receipt(payload: Mapping[str, Any], *, run_root: str | Path) -> None:
    """Validate the Gate 0 receipt and bind its artifacts to one run root."""

    if payload.get("schema_version") != "formal-v4.1-differentiable-lp-gate-v1":
        raise ValueError("DiffLP receipt schema must be formal-v4.1")
    if payload.get("test_set_accessed") is not False:
        raise ValueError("DiffLP receipt reports test-set access")
    if int(payload.get("windows", 0)) < 100 or list(payload.get("training_years", ())) != [2015, 2016, 2017, 2018]:
        raise ValueError("DiffLP receipt is not a 100-window 2015-2018 train-only probe")
    root = Path(run_root).resolve()
    for field in ("benchmark_path", "train_archive_path"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"DiffLP receipt is missing {field}")
        path = (root / value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"DiffLP {field} escapes run root") from exc
        if not path.is_file() or _sha256_file(path) != payload.get(field.replace("_path", "_sha256")):
            raise ValueError(f"DiffLP {field} hash does not match run root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_backend() -> tuple[Any, Any]:
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except Exception as exc:  # pragma: no cover - exercised by environment gate
        raise ImportError(
            "Differentiable-LP requires the isolated cvxpy/cvxpylayers/diffcp environment"
        ) from exc
    return cp, CvxpyLayer


def _scalar(value: Tensor | float) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32).reshape(())


class DifferentiableIESLayer(nn.Module):
    """A DPP-compliant four-hour standard-IES layer.

    Inputs are one window at a time (``[H,...]``); the wrapper accepts a batch
    and loops over windows so the public contract remains one optimizer call per
    rolling window.  Prices are ``[H,4]`` in the order grid, gas, grid-emission,
    gas-emission.  Physical constants are supplied as a mapping and are fixed
    for the run, never learned through this baseline.
    """

    def __init__(self, parameters: Mapping[str, float], *, training_regularizer: float = 0.0, tie_break_throughput_cost: float = 1.0e-5) -> None:
        super().__init__()
        cp, CvxpyLayer = _require_backend()
        self.spec = DifferentiableLPProblemSpec(
            training_regularizer=float(training_regularizer),
            tie_break_throughput_cost=float(tie_break_throughput_cost),
        )
        self.spec.validate()
        self.parameters = dict(parameters)
        self.tie_break_throughput_cost = float(tie_break_throughput_cost)
        required = (
            "grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity",
            "gas_boiler_capacity", "electric_chiller_capacity", "absorption_chiller_capacity",
            "bess_power_capacity", "bess_energy_capacity", "chp_electric_efficiency",
            "chp_heat_efficiency", "gas_boiler_efficiency", "electric_chiller_cop",
            "absorption_chiller_cop", "bess_roundtrip_efficiency", "bess_throughput_cost",
            "unserved_penalty", "chp_ramp_fraction",
        )
        missing = [name for name in required if name not in self.parameters]
        if missing:
            raise ValueError(f"missing differentiable-LP physical parameters: {missing}")
        H = HORIZON
        demand = cp.Parameter((H, 3), name="rigid_demand")
        renewable = cp.Parameter((H, 2), nonneg=True, name="renewable_forecast")
        prices = cp.Parameter((H, 4), name="prices")
        initial_soc = cp.Parameter(name="initial_soc")
        previous_chp = cp.Parameter(name="previous_chp")
        values = {name: cp.Variable(H, nonneg=True, name=name) for name in self._variable_names()}
        eta = float(self.parameters["bess_roundtrip_efficiency"]) ** 0.5
        constraints = []
        p = self.parameters
        for t in range(H):
            constraints += [
                values["grid"][t] + values["pv_use"][t] + values["wt_use"][t] + values["p_chp"][t] + values["p_discharge"][t] + values["slack_e"][t] - values["p_ec"][t] - values["p_charge"][t] == demand[t, 0],
                values["q_ec"][t] + values["q_ac"][t] + values["slack_c"][t] == demand[t, 1],
                values["q_chp"][t] + values["q_gb"][t] + values["slack_h"][t] - values["q_ac_in"][t] - values["q_dump"][t] == demand[t, 2],
                values["pv_use"][t] + values["pv_curt"][t] == renewable[t, 0],
                values["wt_use"][t] + values["wt_curt"][t] == renewable[t, 1],
                values["p_chp"][t] == float(p["chp_electric_efficiency"]) * values["g_chp"][t],
                values["q_chp"][t] == float(p["chp_heat_efficiency"]) * values["g_chp"][t],
                values["q_gb"][t] == float(p["gas_boiler_efficiency"]) * values["g_gb"][t],
                values["q_ec"][t] == float(p["electric_chiller_cop"]) * values["p_ec"][t],
                values["q_ac"][t] == float(p["absorption_chiller_cop"]) * values["q_ac_in"][t],
            ]
            prior_soc = initial_soc * float(p["bess_energy_capacity"]) if t == 0 else values["soc"][t - 1]
            constraints += [values["soc"][t] - eta * values["p_charge"][t] + values["p_discharge"][t] / eta == prior_soc]
            prior_chp = previous_chp if t == 0 else values["p_chp"][t - 1]
            ramp = float(p["chp_ramp_fraction"]) * float(p["chp_electric_capacity"])
            constraints += [values["p_chp"][t] - prior_chp <= ramp, prior_chp - values["p_chp"][t] <= ramp]
        constraints += [
            values["grid"] <= float(p["grid_import_capacity"]),
            values["g_chp"] <= float(p["chp_electric_capacity"]) / float(p["chp_electric_efficiency"]),
            values["g_gb"] <= float(p["gas_boiler_capacity"]) / float(p["gas_boiler_efficiency"]),
            values["p_chp"] <= float(p["chp_electric_capacity"]), values["q_chp"] <= float(p["chp_heat_capacity"]),
            values["q_gb"] <= float(p["gas_boiler_capacity"]), values["p_ec"] <= float(p["electric_chiller_capacity"]) / float(p["electric_chiller_cop"]),
            values["q_ec"] <= float(p["electric_chiller_capacity"]), values["q_ac_in"] <= float(p["absorption_chiller_capacity"]) / float(p["absorption_chiller_cop"]),
            values["q_ac"] <= float(p["absorption_chiller_capacity"]), values["p_charge"] <= float(p["bess_power_capacity"]),
            values["p_discharge"] <= float(p["bess_power_capacity"]), values["soc"] <= float(p["bess_energy_capacity"]),
            values["soc"][-1] == initial_soc * float(p["bess_energy_capacity"]),
        ]
        grid_unit = prices[:, 0] + prices[:, 2]
        gas_unit = prices[:, 1] + prices[:, 3]
        objective = cp.sum(cp.multiply(grid_unit, values["grid"])) + cp.sum(cp.multiply(gas_unit, values["g_chp"] + values["g_gb"]))
        # SCS can return a degenerate simultaneous charge/discharge plan when
        # the benchmark's 1e-6 throughput cost is below its numerical scale.
        # A fixed 1e-5 tie-break is used only inside this differentiable layer;
        # all reported decisions are evaluated with the unregularized common
        # objective, and the gate records the resulting parity explicitly.
        throughput_cost = float(p["bess_throughput_cost"]) + self.tie_break_throughput_cost
        objective += throughput_cost * cp.sum(values["p_charge"] + values["p_discharge"])
        objective += float(p["unserved_penalty"]) * cp.sum(values["slack_e"] + values["slack_c"] + values["slack_h"])
        if training_regularizer:
            objective += float(training_regularizer) * sum(cp.sum_squares(values[name]) for name in self._variable_names())
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise ValueError("formal-v4 Differentiable-LP formulation is not DPP")
        self.problem = problem
        self._layer = CvxpyLayer(problem, parameters=[demand, renewable, prices, initial_soc, previous_chp], variables=[values[name] for name in self._variable_names()])

    @staticmethod
    def _variable_names() -> tuple[str, ...]:
        from ..scheduling.dispatch_schema import VARIABLES
        return tuple(VARIABLES)

    def forward(self, rigid_demand: Tensor, renewable_forecast: Tensor, prices: Tensor, initial_soc: Tensor, previous_chp: Tensor) -> Tensor:
        # Keep the cone-program solve in float64.  With the four-hour SOC
        # recursion, converting the solver result to float32 before the
        # physical audit can turn a sub-micro-unit residual into a visible
        # 1e-6--1e-5 violation.  Casting is differentiable, so gradients still
        # reach a float32 forecaster used by the surrounding network.
        rigid_demand = rigid_demand.to(dtype=torch.float64)
        renewable_forecast = renewable_forecast.to(dtype=torch.float64)
        prices = prices.to(dtype=torch.float64)
        initial_soc = initial_soc.to(dtype=torch.float64)
        previous_chp = previous_chp.to(dtype=torch.float64)
        if rigid_demand.ndim == 2:
            rigid_demand, renewable_forecast, prices = rigid_demand.unsqueeze(0), renewable_forecast.unsqueeze(0), prices.unsqueeze(0)
        if rigid_demand.shape[1:] != (HORIZON, 3) or renewable_forecast.shape[1:] != (HORIZON, 2) or prices.shape[1:] != (HORIZON, 4):
            raise ValueError("formal-v4 Differentiable-LP inputs have shapes [B,4,3], [B,4,2], [B,4,4]")
        batches = []
        for b in range(rigid_demand.shape[0]):
            soc = initial_soc[b] if initial_soc.ndim else initial_soc
            chp = previous_chp[b] if previous_chp.ndim else previous_chp
            # SCS is the explicitly pinned differentiable backend for the
            # Windows reference environment.  ECOS has produced native
            # access-violation crashes on some CPU builds; silently falling
            # back to it would make Gate 0 non-reproducible.
            solved = self._layer(
                rigid_demand[b], renewable_forecast[b], prices[b], soc, chp,
                solver_args={"solve_method": "SCS", "eps": 1e-10, "max_iters": 100000},
            )
            batches.append(torch.stack(solved, dim=-1))
        return torch.stack(batches, dim=0)


class DifferentiableLPForecasterAdapter(nn.Module):
    """Forecast-then-differentiable-LP wrapper with no neural scheduler."""

    def __init__(self, forecaster: nn.Module, layer: DifferentiableIESLayer) -> None:
        super().__init__()
        self.forecaster = forecaster
        self.layer = layer

    def forward(self, history: Tensor, renewable_forecast: Tensor, prices: Tensor, initial_soc: Tensor, previous_chp: Tensor) -> Tensor:
        forecast = self.forecaster(history)
        if isinstance(forecast, (tuple, list)):
            forecast = forecast[0]
        if forecast.shape[-1] == 4:
            forecast = forecast[..., :3]
        return self.layer(forecast, renewable_forecast, prices, initial_soc, previous_chp)

    def forecaster_parameters(self):
        return self.forecaster.parameters()

    def scheduler_parameters(self):
        return ()


def write_diffopt_lock(output: str | Path, requirements: str | Path) -> dict[str, Any]:
    """Record exact package versions when run inside the isolated environment."""
    output_path = Path(output)
    previous_probe: dict[str, Any] = {"status": "not_run", "eligible_for_gate0": False}
    if output_path.exists():
        try:
            old = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(old, Mapping) and isinstance(old.get("native_layer_probe"), Mapping):
                previous_probe = dict(old["native_layer_probe"])
        except (OSError, json.JSONDecodeError):
            pass
    payload: dict[str, Any] = {
        "status": "resolved", "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "requirements_sha256": hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
        "packages": {},
        "native_layer_probe": previous_probe,
    }
    for name in ("numpy", "torch", "cvxpy", "cvxpylayers", "diffcp", "ecos"):
        try:
            module = __import__(name)
            payload["packages"][name] = getattr(module, "__version__", "unknown")
        except Exception:
            payload["packages"][name] = None
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "DIFFOPT_METHOD_ID", "DifferentiableIESLayer", "DifferentiableLPForecasterAdapter",
    "DifferentiableLPGateReceipt", "DifferentiableLPProblemSpec", "validate_diffopt_gate_receipt", "write_diffopt_lock",
]
