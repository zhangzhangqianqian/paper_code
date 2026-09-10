"""Canonical frozen contract for the complete RSC-PF formal experiment.

This module is deliberately independent of the legacy v4.x contracts.  It
defines the exact primary roster, row accounting, data boundary, and the
fail-closed transition required before sealed evaluation may be read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


SCHEMA_VERSION = "rsc-pf-complete-formal-v1"
PRIMARY_METHOD_IDS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
    "Seasonal-Naive-PTO",
    "Perfect-Information-MPC",
)
STOCHASTIC_METHOD_IDS = PRIMARY_METHOD_IDS[:7]
DETERMINISTIC_METHOD_IDS = PRIMARY_METHOD_IDS[7:]
FROZEN_SEEDS = (2026, 2027, 2028, 2029, 2030)
FROZEN_STAGES = ("gate1", "gate2")


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    role: str
    stochastic: bool
    deployable: bool
    forecast_metrics_applicable: bool
    inference_lp_calls_per_origin: int
    reproduction_level: str


@dataclass(frozen=True)
class MethodSeedKey:
    method_id: str
    seed: Optional[int]


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _exact_sequence(value: Any, expected: Tuple[Any, ...], field: str) -> None:
    actual = _require_sequence(value, field)
    if actual != expected:
        raise ValueError(f"{field} differs from the frozen protocol")


def _frame_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _resolve_frame_path(frame_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (frame_root / path).resolve()


@dataclass(frozen=True)
class CompleteFormalContract:
    payload: Mapping[str, Any]
    contract_sha256: str
    source_path: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "CompleteFormalContract":
        source = Path(path).resolve()
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("complete formal contract must be a JSON object")
        contract = cls(MappingProxyType(dict(payload)), hashlib.sha256(raw).hexdigest(), source)
        contract.validate()
        return contract

    @property
    def primary_method_ids(self) -> Tuple[str, ...]:
        return tuple(str(row["method_id"]) for row in self.payload["methods"])

    @property
    def selection_origin_count(self) -> int:
        return int(self.payload["data_boundary"]["selection_origin_count"])

    @property
    def evaluation_origin_count(self) -> int:
        return int(self.payload["data_boundary"]["evaluation_origin_count"])

    @property
    def train_years(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data_boundary"]["train_years"])

    @property
    def selection_year(self) -> int:
        return int(self.payload["data_boundary"]["selection_year"])

    @property
    def evaluation_year(self) -> int:
        return int(self.payload["data_boundary"]["evaluation_year"])

    @property
    def excluded_years(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data_boundary"]["excluded_years"])

    @property
    def capacity_binding(self) -> Mapping[str, Any]:
        return _require_mapping(self.payload.get("capacity_binding"), "capacity_binding")

    @property
    def capacity_parameters(self) -> Mapping[str, float]:
        binding = self.capacity_binding
        values = _require_mapping(binding.get("benchmark_values"), "capacity_binding.benchmark_values")
        return MappingProxyType({str(key): float(value) for key, value in values.items()})

    def method(self, method_id: str) -> MethodSpec:
        for row in self.payload["methods"]:
            if row["method_id"] == method_id:
                return MethodSpec(
                    method_id=str(row["method_id"]),
                    display_name=str(row["display_name"]),
                    role=str(row["role"]),
                    stochastic=bool(row["stochastic"]),
                    deployable=bool(row["deployable"]),
                    forecast_metrics_applicable=bool(row["forecast_metrics_applicable"]),
                    inference_lp_calls_per_origin=int(row["inference_lp_calls_per_origin"]),
                    reproduction_level=str(row["reproduction_level"]),
                )
        raise KeyError(f"unknown formal method: {method_id}")

    def expected_rows(self, stage: str) -> Tuple[MethodSeedKey, ...]:
        if stage not in FROZEN_STAGES:
            raise ValueError(f"stage must be one of {FROZEN_STAGES}")
        rows = []
        for method_id in self.primary_method_ids:
            spec = self.method(method_id)
            if spec.stochastic:
                rows.extend(MethodSeedKey(method_id, seed) for seed in FROZEN_SEEDS)
            else:
                rows.append(MethodSeedKey(method_id, None))
        return tuple(rows)

    def authorize_evaluation(self, gate1_transition_path: str | Path) -> Mapping[str, Any]:
        """Return an authorized transition or fail closed before sealed access."""

        transition_path = Path(gate1_transition_path)
        if not transition_path.is_file():
            raise PermissionError("sealed evaluation requires an authorized Gate 1 transition")
        try:
            payload = json.loads(transition_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PermissionError("sealed evaluation requires an authorized Gate 1 transition") from exc
        if not isinstance(payload, Mapping):
            raise PermissionError("sealed evaluation requires an authorized Gate 1 transition")
        if payload.get("contract_sha256") != self.contract_sha256:
            raise PermissionError("Gate 1 transition does not authorize this contract")
        if payload.get("authorized_gate2") is not True:
            raise PermissionError("sealed evaluation requires an authorized Gate 1 transition")
        # Synthetic/tiny matrices are useful for testing the protocol, but can
        # never authorize access to the sealed evaluation split.  Require an
        # explicit paper-result marker so an omitted field cannot be treated as
        # a real experiment by default.
        if payload.get("synthetic") is not False or payload.get("paper_result") is not True:
            raise PermissionError("sealed evaluation requires a non-synthetic paper-result Gate 1 transition")
        if payload.get("audit_status") != "pass":
            raise PermissionError("sealed evaluation requires a passing Gate 1 audit")
        if payload.get("evaluation_year_accessed") is not False:
            raise PermissionError("Gate 1 transition reports evaluation-year access")
        if payload.get("test_set_accessed") is not False:
            raise PermissionError("Gate 1 transition reports test-set access")
        return MappingProxyType(dict(payload))

    def validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("complete formal schema required")
        if self.payload.get("protocol_status") != "frozen":
            raise ValueError("complete formal contract must be frozen")
        frame_root = _frame_root(self.source_path)

        architecture = _require_mapping(self.payload.get("architecture"), "architecture")
        if architecture.get("lookback") != 24 or architecture.get("horizon") != 4:
            raise ValueError("24-hour lookback and four-hour horizon are required")
        if architecture.get("latent_control_dim") != 15 or architecture.get("dispatch_dim") != 21:
            raise ValueError("latent control and dispatch dimensions are invalid")
        if architecture.get("allow_future_binary_decisions") is not False:
            raise ValueError("future binary decisions are outside the formal protocol")
        if architecture.get("gas_semantics") != "station_side_auxiliary_prior":
            raise ValueError("gas must remain a station-side auxiliary prior")
        _exact_sequence(architecture.get("task_order"), ("electricity", "cooling", "heating", "gas"), "architecture.task_order")
        _exact_sequence(architecture.get("rigid_demand_order"), ("electricity", "cooling", "heating"), "architecture.rigid_demand_order")
        _exact_sequence(architecture.get("exog_order"), ("temperature", "humidity", "solar_irradiance", "wind_speed", "wind_direction", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"), "architecture.exog_order")
        _exact_sequence(architecture.get("dispatch_order"), ("grid", "pv_use", "pv_curt", "wt_use", "wt_curt", "g_chp", "g_gb", "p_chp", "q_chp", "q_gb", "p_ec", "q_ec", "q_ac_in", "q_ac", "p_charge", "p_discharge", "soc", "slack_e", "slack_c", "slack_h", "q_dump"), "architecture.dispatch_order")
        _exact_sequence(architecture.get("status_order"), ("chp_on", "gas_boiler_on", "electric_chiller_on", "absorption_chiller_on", "bess_charge_on", "bess_discharge_on"), "architecture.status_order")

        boundary = _require_mapping(self.payload.get("data_boundary"), "data_boundary")
        if self.train_years != (2015, 2016, 2017, 2018) or self.selection_year != 2019:
            raise ValueError("training and selection years are invalid")
        if self.evaluation_year != 2020 or self.excluded_years != (2021,):
            raise ValueError("evaluation and excluded years are invalid")
        if self.selection_origin_count != 8709 or self.evaluation_origin_count != 8757:
            raise ValueError("origin counts are not frozen")
        if boundary.get("allow_evaluation_access_before_gate1") is not False:
            raise ValueError("evaluation access must be fail closed before Gate 1")
        if boundary.get("allow_test_set_access_before_gate2") is not False:
            raise ValueError("test-set access must be fail closed before Gate 2")
        if boundary.get("normalization_fit_split") != "train_only":
            raise ValueError("normalization must be fit on training years only")
        if boundary.get("selection_uses_evaluation_year") is not False or boundary.get("evaluation_uses_excluded_year") is not False:
            raise ValueError("selection/evaluation year boundary is invalid")

        if self.primary_method_ids != PRIMARY_METHOD_IDS:
            raise ValueError("primary method roster or order is not frozen")
        for method_id in PRIMARY_METHOD_IDS:
            spec = self.method(method_id)
            if spec.stochastic != (method_id in STOCHASTIC_METHOD_IDS):
                raise ValueError(f"{method_id} stochastic flag is invalid")
        if self.method("Direct-Policy").forecast_metrics_applicable:
            raise ValueError("Direct-Policy must report forecast metrics as not applicable")
        direct_policy_row = next(row for row in self.payload["methods"] if row["method_id"] == "Direct-Policy")
        if direct_policy_row.get("feasibility_adapter_id") != "state_conditioned_chp_ramp_projection_v1":
            raise ValueError("Direct-Policy feasibility adapter is not frozen")
        if self.method("Differentiable-LP").reproduction_level != "cvxpylayers_methodology_adaptation":
            raise ValueError("Differentiable-LP reproduction level is invalid")
        if self.method("Differentiable-LP").inference_lp_calls_per_origin != 1:
            raise ValueError("Differentiable-LP must declare one inference LP call per origin")
        if self.method("RSC-PF").inference_lp_calls_per_origin != 0:
            raise ValueError("RSC-PF must declare no inference LP call")
        if self.method("Perfect-Information-MPC").deployable:
            raise ValueError("Perfect-Information-MPC is a reference, not a deployable model")

        capacity = self.capacity_binding
        if capacity.get("schema_version") != "rsc-pf-complete-formal-capacity-binding-v1":
            raise ValueError("capacity binding schema is invalid")
        if capacity.get("status") != "pass":
            raise ValueError("capacity binding must be a passing artifact")
        if tuple(int(value) for value in capacity.get("fit_years", [])) != self.train_years:
            raise ValueError("capacity binding must use the frozen training years")
        if capacity.get("selection_influenced_capacity") is not False:
            raise ValueError("selection data must not influence capacity binding")
        if capacity.get("evaluation_year_accessed") is not False:
            raise ValueError("capacity binding reports evaluation-year access")
        if not math.isfinite(float(capacity.get("selected_multiplier", float("nan")))):
            raise ValueError("capacity binding multiplier is invalid")
        chronological = _require_mapping(capacity.get("chronological_audit"), "capacity_binding.chronological_audit")
        if chronological.get("meets_threshold") is not True or int(chronological.get("solves", 0)) <= 0:
            raise ValueError("capacity chronological audit is not passing")
        thresholds = _require_mapping(capacity.get("thresholds"), "capacity_binding.thresholds")
        if float(thresholds.get("cooling_shortage_energy_ratio_max")) != 0.005 or float(thresholds.get("cooling_shortage_hour_rate_max")) != 0.01:
            raise ValueError("capacity thresholds are not frozen")
        provenance = _require_mapping(capacity.get("provenance"), "capacity_binding.provenance")
        upstream = _resolve_frame_path(frame_root, str(provenance.get("upstream_contract_path")))
        if not upstream.is_file() or hashlib.sha256(upstream.read_bytes()).hexdigest() != str(provenance.get("upstream_contract_sha256")).lower():
            raise ValueError("capacity upstream contract provenance is invalid")
        benchmark_path = _resolve_frame_path(frame_root, "configs/standard_ies_benchmark_formal_v4_complete.json")
        if not benchmark_path.is_file():
            raise ValueError("current formal benchmark binding is missing")
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark.get("schema_version") != "standard-ies-benchmark-v4.1" or tuple(int(value) for value in benchmark.get("source_years", [])) != self.train_years:
            raise ValueError("current formal benchmark is not training-only")
        benchmark_values = _require_mapping(benchmark.get("values"), "benchmark.values")
        declared_values = _require_mapping(capacity.get("benchmark_values"), "capacity_binding.benchmark_values")
        required_capacity_keys = ("grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity", "gas_boiler_capacity", "electric_chiller_capacity", "absorption_chiller_capacity", "bess_power_capacity", "bess_energy_capacity")
        for key in required_capacity_keys:
            actual = float(benchmark_values.get(key, float("nan")))
            declared = float(declared_values.get(key, float("nan")))
            if not math.isfinite(actual) or not math.isfinite(declared) or actual != declared or actual <= 0.0:
                raise ValueError(f"capacity benchmark value is invalid: {key}")
        for name, relative in (("rules_sha256", "configs/standard_ies_formal_v4_rules.yaml"), ("ledger_sha256", "configs/scheduling_parameter_ledger_v2.csv")):
            path = _resolve_frame_path(frame_root, relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != str(provenance.get(name)).lower():
                raise ValueError(f"capacity {name} provenance is invalid")

        seeds = tuple(int(value) for value in _require_sequence(self.payload.get("seeds"), "seeds"))
        if seeds != FROZEN_SEEDS:
            raise ValueError("seed order is not frozen")
        if len(self.expected_rows("gate1")) != 37 or self.expected_rows("gate1") != self.expected_rows("gate2"):
            raise ValueError("formal matrix must contain 37 rows at both gates")

        training = _require_mapping(self.payload.get("training"), "training")
        common = _require_mapping(training.get("common"), "training.common")
        if common.get("effective_batch_size") != 64 or common.get("max_epochs_per_stage") != 30:
            raise ValueError("common training budget is invalid")
        if common.get("patience") != 5 or common.get("validation_interval") != 1:
            raise ValueError("validation policy is invalid")
        loss = _require_mapping(training.get("loss"), "training.loss")
        if tuple(float(v) for v in loss.get("forecast_task_weights", [])) != (1.0, 1.0, 1.0, 0.25):
            raise ValueError("forecast task weights are invalid")
        if float(loss.get("gas_adjustment")) != 0.0 or loss.get("primary_carbon_loss") is not False:
            raise ValueError("gas/carbon loss semantics are invalid")
        search = _require_mapping(training.get("search"), "training.search")
        if search.get("max_trials_per_method") != 4:
            raise ValueError("search budget must be four trials")
        if tuple(float(v) for v in search.get("rsc_pf_decision_multiplier_grid", [])) != (1.0, 1.5, 2.0, 3.0):
            raise ValueError("RSC-PF multiplier grid is invalid")
        if tuple(float(v) for v in search.get("difflp_learning_rate_grid", [])) != (0.00001, 0.00003, 0.0001, 0.0003):
            raise ValueError("Differentiable-LP learning-rate grid is invalid")
        if search.get("selection_seed") != 2026 or search.get("selection_year") != 2019:
            raise ValueError("search must use seed 2026 and selection year 2019")
        if search.get("retune_missing_seed") is not False:
            raise ValueError("missing seeds must not be retuned")

        statistics = _require_mapping(self.payload.get("statistics"), "statistics")
        if statistics.get("independent_unit") != "model_seed" or statistics.get("paired_unit") != "168h_temporal_block_within_seed":
            raise ValueError("statistical units are invalid")
        if statistics.get("bootstrap_replicates") != 2000 or statistics.get("bootstrap_order") != ["seed", "paired_168h_block"]:
            raise ValueError("bootstrap protocol is invalid")
        if statistics.get("primary_contrast") != "RSC-PF_minus_Decoupled-RSC-PF":
            raise ValueError("primary contrast is invalid")

        source_contracts = _require_mapping(self.payload.get("source_contracts"), "source_contracts")
        for name, item in source_contracts.items():
            record = _require_mapping(item, f"source_contracts.{name}")
            source = _resolve_frame_path(frame_root, str(record["path"]))
            if not source.is_file():
                raise ValueError(f"source contract is missing: {record['path']}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != str(record["sha256"]).lower():
                raise ValueError(f"source contract hash mismatch: {record['path']}")

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


__all__ = [
    "CompleteFormalContract",
    "FROZEN_SEEDS",
    "FROZEN_STAGES",
    "MethodSeedKey",
    "MethodSpec",
    "PRIMARY_METHOD_IDS",
]
