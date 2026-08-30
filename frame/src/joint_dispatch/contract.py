"""Frozen contract for the end-to-end forecast--dispatch experiment.

The contract is intentionally stricter than a convenience configuration.  It
defines the data order, training variants, selection gates, resource gate, and
the only checkpoint prefixes that may be reused by the optional warm-start
challenger.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np


TASK_ORDER = ("electricity", "cooling", "heating", "gas")
EXOG_ORDER = (
    "temperature", "humidity", "solar_irradiance", "wind_speed", "wind_direction",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
)
DISPATCH_ORDER = (
    "grid", "pv_use", "pv_curt", "wt_use", "wt_curt", "g_chp", "g_gb", "p_chp",
    "q_chp", "q_gb", "p_ec", "q_ec", "q_ac_in", "q_ac", "p_charge", "p_discharge",
    "soc", "slack_e", "slack_c", "slack_h", "q_dump",
)
STATUS_ORDER = (
    "chp_on", "gas_boiler_on", "electric_chiller_on", "absorption_chiller_on",
    "bess_charge_on", "bess_discharge_on",
)
SEEDS = (2026, 2027, 2028, 2029, 2030)
FORECAST_TASK_WEIGHTS = (1.0, 1.0, 1.0, 0.25)
WARM_START_PREFIXES = {
    "forecast": (
        "base.encoders.", "base.step_embeddings.", "base.role_embeddings.", "base.router.",
        "base.message_projector.", "base.fusion.", "base.head.",
    ),
    "scheduler": ("residual.", "output_projection."),
}


@dataclass(frozen=True)
class CurriculumSpec:
    forecast_weight: float
    imitation_start_weight: float
    imitation_final_weight: float
    decision_start_weight: float
    decision_final_weight: float
    ramp_epochs: int
    rollin_start_fraction: float
    model_history_fraction: float


@dataclass(frozen=True)
class JointVariant:
    name: Literal["joint_from_scratch", "warm_started_joint", "frozen_pto"]
    forecasting_trainable: bool
    scheduling_trainable: bool
    initialization: Literal["random", "warm_start", "frozen"]


@dataclass(frozen=True)
class JointTrainingContract:
    schema_version: str
    lookback: int
    horizon: int
    task_order: tuple[str, ...]
    exog_order: tuple[str, ...]
    dispatch_order: tuple[str, ...]
    status_order: tuple[str, ...]
    seeds: tuple[int, ...]
    forecast_task_weights: tuple[float, ...]
    variants: tuple[JointVariant, ...]
    curriculum: CurriculumSpec
    warm_start_key_prefixes: Mapping[str, tuple[str, ...]]
    paths: Mapping[str, Path]
    selection: Mapping[str, Any]
    resource_gate: Mapping[str, Any]
    safety: Mapping[str, Any]
    source_path: Path | None = None

    def variant(self, name: str) -> JointVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise KeyError(f"unknown joint variant: {name}")

    def path(self, name: str) -> Path:
        try:
            return self.paths[name]
        except KeyError as exc:
            raise KeyError(f"unknown joint contract path: {name}") from exc

    def to_dict(self) -> dict[str, Any]:
        def thaw(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): thaw(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            return value

        return {
            "schema_version": self.schema_version,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "task_order": list(self.task_order),
            "exog_order": list(self.exog_order),
            "dispatch_order": list(self.dispatch_order),
            "status_order": list(self.status_order),
            "seeds": list(self.seeds),
            "forecast_task_weights": list(self.forecast_task_weights),
            "variants": [asdict(variant) for variant in self.variants],
            "curriculum": asdict(self.curriculum),
            "warm_start_key_prefixes": thaw(self.warm_start_key_prefixes),
            "paths": thaw(self.paths),
            "selection": thaw(self.selection),
            "resource_gate": thaw(self.resource_gate),
            "safety": thaw(self.safety),
        }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return value


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _exact_float(value: Any, expected: float, field: str) -> None:
    result = _finite_float(value, field)
    if result != expected:
        raise ValueError(f"{field} must equal {expected}")


def _positive_int(value: Any, field: str) -> int:
    result = _finite_float(value, field)
    integer = int(result)
    if result != integer or integer <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return integer


def _validate_exact_sequence(value: Any, expected: Sequence[Any], field: str) -> None:
    actual = tuple(_sequence(value, field))
    if actual != tuple(expected):
        raise ValueError(f"{field} does not match the frozen order")


def validate_joint_contract(data: Mapping[str, Any]) -> None:
    """Validate the complete joint-training contract without touching files."""

    if not isinstance(data, Mapping):
        raise ValueError("joint contract must be an object")
    required = {
        "schema_version", "lookback", "horizon", "task_order", "exog_order", "dispatch_order",
        "status_order", "seeds", "forecast_task_weights", "variants", "curriculum",
        "warm_start_key_prefixes", "paths", "selection", "resource_gate", "safety",
    }
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise ValueError(f"joint contract missing fields: {missing}")
    if unknown:
        raise ValueError(f"joint contract has unknown fields: {unknown}")
    if data["schema_version"] != "joint-forecast-dispatch-v1":
        raise ValueError("schema_version must be joint-forecast-dispatch-v1")
    if _positive_int(data["lookback"], "lookback") != 24:
        raise ValueError("lookback must equal 24")
    if _positive_int(data["horizon"], "horizon") != 4:
        raise ValueError("horizon must equal 4")
    _validate_exact_sequence(data["task_order"], TASK_ORDER, "task_order")
    _validate_exact_sequence(data["exog_order"], EXOG_ORDER, "exog_order")
    _validate_exact_sequence(data["dispatch_order"], DISPATCH_ORDER, "dispatch_order")
    _validate_exact_sequence(data["status_order"], STATUS_ORDER, "status_order")

    seeds = tuple(_positive_int(seed, "seeds item") for seed in _sequence(data["seeds"], "seeds"))
    if seeds != SEEDS:
        raise ValueError("seeds must equal the frozen five-seed protocol")
    weights = tuple(_finite_float(value, "forecast_task_weights item") for value in _sequence(data["forecast_task_weights"], "forecast_task_weights"))
    if weights != FORECAST_TASK_WEIGHTS:
        raise ValueError("forecast_task_weights must equal [1.0, 1.0, 1.0, 0.25]")

    raw_variants = _sequence(data["variants"], "variants")
    variants: dict[str, Mapping[str, Any]] = {}
    for raw in raw_variants:
        item = _mapping(raw, "variants item")
        fields = {"name", "forecasting_trainable", "scheduling_trainable", "initialization"}
        if set(item) != fields:
            raise ValueError("variants item fields are not frozen")
        name = str(item["name"])
        if name in variants:
            raise ValueError("variants must have unique names")
        variants[name] = item
    expected_names = {"joint_from_scratch", "warm_started_joint", "frozen_pto"}
    if set(variants) != expected_names:
        raise ValueError("variants must contain exactly the three frozen names")
    expected_variant_fields = {
        "joint_from_scratch": (True, True, "random"),
        "warm_started_joint": (True, True, "warm_start"),
        "frozen_pto": (False, False, "frozen"),
    }
    for name, (forecasting_trainable, scheduling_trainable, initialization) in expected_variant_fields.items():
        item = variants[name]
        if (item["forecasting_trainable"], item["scheduling_trainable"], item["initialization"]) != (
            forecasting_trainable, scheduling_trainable, initialization
        ):
            raise ValueError(f"{name} has invalid trainability or initialization")

    curriculum = _mapping(data["curriculum"], "curriculum")
    expected_curriculum_fields = {
        "forecast_weight", "imitation_start_weight", "imitation_final_weight",
        "decision_start_weight", "decision_final_weight", "ramp_epochs",
        "rollin_start_fraction", "model_history_fraction",
    }
    if set(curriculum) != expected_curriculum_fields:
        raise ValueError("curriculum fields are not frozen")
    _exact_float(curriculum["forecast_weight"], 1.0, "curriculum.forecast_weight")
    _exact_float(curriculum["imitation_start_weight"], 1.0, "curriculum.imitation_start_weight")
    _exact_float(curriculum["imitation_final_weight"], 0.25, "curriculum.imitation_final_weight")
    _exact_float(curriculum["decision_start_weight"], 0.05, "curriculum.decision_start_weight")
    _exact_float(curriculum["decision_final_weight"], 1.0, "curriculum.decision_final_weight")
    if _positive_int(curriculum["ramp_epochs"], "curriculum.ramp_epochs") != 20:
        raise ValueError("curriculum.ramp_epochs must equal 20")
    _exact_float(curriculum["rollin_start_fraction"], 0.4, "curriculum.rollin_start_fraction")
    _exact_float(curriculum["model_history_fraction"], 0.5, "curriculum.model_history_fraction")

    prefixes = _mapping(data["warm_start_key_prefixes"], "warm_start_key_prefixes")
    if set(prefixes) != set(WARM_START_PREFIXES):
        raise ValueError("warm_start_key_prefixes fields are not frozen")
    for group, expected in WARM_START_PREFIXES.items():
        _validate_exact_sequence(prefixes[group], expected, f"warm_start_key_prefixes.{group}")

    paths = _mapping(data["paths"], "paths")
    expected_paths = {
        "kitakyushu_data_dir", "benchmark_path", "parameter_ledger_path", "renewable_forecast_root",
        "output_root", "warm_start_forecast_checkpoint", "warm_start_scheduler_checkpoint",
    }
    if set(paths) != expected_paths:
        raise ValueError("paths fields are not frozen")
    for name, value in paths.items():
        if not isinstance(value, str) or (not value and not name.startswith("warm_start_")):
            raise ValueError(f"paths.{name} must be a nonempty string")

    selection = _mapping(data["selection"], "selection")
    expected_selection = {
        "selection_split", "rigid_task_indices", "aggregate_wape_relative_max",
        "per_task_wape_relative_max", "bootstrap_block_hours", "bootstrap_replicates", "alpha",
    }
    if set(selection) != expected_selection:
        raise ValueError("selection fields are not frozen")
    if selection["selection_split"] != "validation":
        raise ValueError("selection_split must be validation")
    _validate_exact_sequence(selection["rigid_task_indices"], (0, 1, 2), "selection.rigid_task_indices")
    _exact_float(selection["aggregate_wape_relative_max"], 1.02, "selection.aggregate_wape_relative_max")
    _exact_float(selection["per_task_wape_relative_max"], 1.05, "selection.per_task_wape_relative_max")
    if _positive_int(selection["bootstrap_block_hours"], "selection.bootstrap_block_hours") != 168:
        raise ValueError("bootstrap_block_hours must equal 168")
    if _positive_int(selection["bootstrap_replicates"], "selection.bootstrap_replicates") != 2000:
        raise ValueError("bootstrap_replicates must equal 2000")
    _exact_float(selection["alpha"], 0.05, "selection.alpha")

    resource = _mapping(data["resource_gate"], "resource_gate")
    if set(resource) != {"benchmark_solves", "max_projected_p95_hours"}:
        raise ValueError("resource_gate fields are not frozen")
    if _positive_int(resource["benchmark_solves"], "resource_gate.benchmark_solves") != 500:
        raise ValueError("resource_gate.benchmark_solves must equal 500")
    _exact_float(resource["max_projected_p95_hours"], 24.0, "resource_gate.max_projected_p95_hours")

    safety = _mapping(data["safety"], "safety")
    expected_safety = {
        "online_exact_lp_calls", "test_year", "normalization_fit_split", "allow_output_overwrite",
        "allow_future_binary_decisions", "primary_carbon_loss",
    }
    if set(safety) != expected_safety:
        raise ValueError("safety fields are not frozen")
    if safety["online_exact_lp_calls"] != 0:
        raise ValueError("online_exact_lp_calls must equal 0")
    if safety["test_year"] != 2021:
        raise ValueError("test_year must equal 2021")
    if safety["normalization_fit_split"] != "train":
        raise ValueError("normalization_fit_split must be train")
    if safety["allow_output_overwrite"] is not False:
        raise ValueError("allow_output_overwrite must be false")
    if safety["allow_future_binary_decisions"] is not False:
        raise ValueError("allow_future_binary_decisions must be false")
    if safety["primary_carbon_loss"] is not False:
        raise ValueError("primary_carbon_loss must be false")


def _repo_root(contract_path: Path) -> Path:
    # The config lives at <repo>/frame/configs; two parents reach <repo>.
    return contract_path.resolve().parents[2]


def load_joint_training_contract(path: str | Path) -> JointTrainingContract:
    """Load, validate, and materialize the immutable joint contract."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_joint_contract(payload)
    curriculum_payload = payload["curriculum"]
    curriculum = CurriculumSpec(
        forecast_weight=float(curriculum_payload["forecast_weight"]),
        imitation_start_weight=float(curriculum_payload["imitation_start_weight"]),
        imitation_final_weight=float(curriculum_payload["imitation_final_weight"]),
        decision_start_weight=float(curriculum_payload["decision_start_weight"]),
        decision_final_weight=float(curriculum_payload["decision_final_weight"]),
        ramp_epochs=int(curriculum_payload["ramp_epochs"]),
        rollin_start_fraction=float(curriculum_payload["rollin_start_fraction"]),
        model_history_fraction=float(curriculum_payload["model_history_fraction"]),
    )
    root = _repo_root(source)
    path_values = {
        name: (Path(value) if Path(value).is_absolute() else root / Path(value))
        for name, value in payload["paths"].items()
    }
    variants = tuple(
        JointVariant(
            name=str(item["name"]),
            forecasting_trainable=bool(item["forecasting_trainable"]),
            scheduling_trainable=bool(item["scheduling_trainable"]),
            initialization=str(item["initialization"]),
        )
        for item in payload["variants"]
    )
    prefixes = MappingProxyType({
        group: tuple(str(value) for value in values)
        for group, values in payload["warm_start_key_prefixes"].items()
    })
    return JointTrainingContract(
        schema_version=str(payload["schema_version"]),
        lookback=int(payload["lookback"]),
        horizon=int(payload["horizon"]),
        task_order=tuple(str(value) for value in payload["task_order"]),
        exog_order=tuple(str(value) for value in payload["exog_order"]),
        dispatch_order=tuple(str(value) for value in payload["dispatch_order"]),
        status_order=tuple(str(value) for value in payload["status_order"]),
        seeds=tuple(int(value) for value in payload["seeds"]),
        forecast_task_weights=tuple(float(value) for value in payload["forecast_task_weights"]),
        variants=variants,
        curriculum=curriculum,
        warm_start_key_prefixes=prefixes,
        paths=MappingProxyType(path_values),
        selection=MappingProxyType(dict(payload["selection"])),
        resource_gate=MappingProxyType(dict(payload["resource_gate"])),
        safety=MappingProxyType(dict(payload["safety"])),
        source_path=source,
    )


__all__ = [
    "CURRICULUM_SPEC", "DISPATCH_ORDER", "EXOG_ORDER", "FORECAST_TASK_WEIGHTS", "JointTrainingContract",
    "JointVariant", "CurriculumSpec", "STATUS_ORDER", "TASK_ORDER", "WARM_START_PREFIXES",
    "load_joint_training_contract", "validate_joint_contract",
]
