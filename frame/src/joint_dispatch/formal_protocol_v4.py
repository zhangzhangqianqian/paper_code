"""Strict, year-locked protocol objects for the RSC-PF formal v4/v4.1 run.

This module only validates and describes the protocol.  It deliberately does
not read any experiment results and does not authorize evaluation data until a
caller supplies an explicit Gate 3 authorization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


SCHEMA_VERSION = "joint-forecast-dispatch-formal-v4"
SCHEMA_VERSION_V41 = "joint-forecast-dispatch-formal-v4.1"
SEEDS = (2026, 2027, 2028, 2029, 2030)
TRAIN_YEARS = (2015, 2016, 2017, 2018)
SELECTION_YEAR = 2019
EVALUATION_YEAR = 2020
FORMAL_V4_METHODS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
    "Perfect-Information-MPC",
    "Seasonal-Naive-PTO",
)
TRAINABLE_METHODS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
)
_METHOD_KINDS = {"joint", "decoupled", "direct_policy", "pto", "external_pto", "differentiable_lp", "oracle"}
_METHOD_FIELDS = {"name", "kind", "trainable", "stochastic", "online_exact_lp", "reproduction_level", "comparison_role"}
_ROOT_FIELDS = {
    "schema_version", "protocol_status", "lookback", "horizon", "task_order", "exog_order",
    "dispatch_order", "status_order", "seeds", "train_years", "selection_year", "evaluation_year",
    "allow_test_access", "allow_evaluation_access_before_gate3", "claims", "methods", "trainable_methods",
    "forecast_guardrails", "training", "selection", "statistics", "resource_gate", "capacity",
    "renewable_forecast", "state_feature_config", "paths", "safety",
}
_ROOT_FIELDS_V41 = _ROOT_FIELDS | {
    "invalid_run_registry", "benchmark_rule_config", "source_closure_file", "source_manifest_required",
}


@dataclass(frozen=True)
class MethodV4Spec:
    name: str
    kind: str
    trainable: bool
    stochastic: bool
    online_exact_lp: bool
    reproduction_level: str
    comparison_role: str


@dataclass(frozen=True)
class SearchBudgetV4:
    max_trials: int
    max_epochs: int
    patience: int
    validation_interval: int


@dataclass(frozen=True)
class StateFeatureSpec:
    continuous_features: tuple[str, ...]
    activity_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    activity_threshold: float
    source_dispatch_order: tuple[str, ...]


@dataclass(frozen=True)
class ForecastGuardrails:
    rigid_macro_wape_ratio_max: float
    rigid_per_task_wape_ratio_max: float
    gas_wape_ratio_max: float


@dataclass(frozen=True)
class FormalV4Spec:
    schema_version: str
    protocol_status: str
    lookback: int
    horizon: int
    task_order: tuple[str, ...]
    exog_order: tuple[str, ...]
    dispatch_order: tuple[str, ...]
    status_order: tuple[str, ...]
    seeds: tuple[int, ...]
    train_years: tuple[int, ...]
    selection_year: int
    evaluation_year: int
    allow_test_access: bool
    allow_evaluation_access_before_gate3: bool
    claims: Mapping[str, bool]
    methods: tuple[str, ...]
    trainable_methods: tuple[str, ...]
    method_specs: Mapping[str, MethodV4Spec]
    search_budgets: Mapping[str, SearchBudgetV4]
    guardrails: ForecastGuardrails
    state_features: StateFeatureSpec
    training: Mapping[str, Any]
    selection: Mapping[str, Any]
    statistics: Mapping[str, Any]
    resource_gate: Mapping[str, Any]
    capacity: Mapping[str, Any]
    renewable_forecast: Mapping[str, Any]
    paths: Mapping[str, Path]
    safety: Mapping[str, Any]
    source_path: Path
    invalid_run_registry: Path | None = None
    benchmark_rule_config: Path | None = None
    source_closure_file: Path | None = None
    source_manifest_required: bool = False

    def method(self, name: str) -> MethodV4Spec:
        try:
            return self.method_specs[name]
        except KeyError as exc:
            raise KeyError(name) from exc


def _strict_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"{field} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{field} is missing fields: {missing}")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _int(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if float(value) != result or (positive and result <= 0):
        raise ValueError(f"{field} must be {'a positive ' if positive else 'an '}integer")
    return result


def _float(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not (result == result and abs(result) != float("inf")) or (positive and result <= 0.0):
        raise ValueError(f"{field} must be {'positive ' if positive else ''}finite")
    return result


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()


def _under(path: Path, root: Path, field: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must stay under {root}") from exc


def _validate_state_payload(payload: Mapping[str, Any]) -> StateFeatureSpec:
    expected = {"schema_version", "continuous_features", "activity_features", "excluded_features", "activity_threshold", "source_dispatch_order"}
    _strict_keys(payload, expected, "state feature contract")
    if payload["schema_version"] != "joint-dispatch-state-features-v4":
        raise ValueError("state feature schema_version is not v4")
    continuous = tuple(_nonempty_text(item, "continuous_features[]") for item in _sequence(payload["continuous_features"], "continuous_features"))
    activity = tuple(_nonempty_text(item, "activity_features[]") for item in _sequence(payload["activity_features"], "activity_features"))
    excluded = tuple(_nonempty_text(item, "excluded_features[]") for item in _sequence(payload["excluded_features"], "excluded_features"))
    source_order = tuple(_nonempty_text(item, "source_dispatch_order[]") for item in _sequence(payload["source_dispatch_order"], "source_dispatch_order"))
    if len(continuous) != 17 or len(activity) != 6:
        raise ValueError("formal v4 state ledger must contain 17 continuous and 6 activity features")
    if len(set(continuous + activity + excluded)) != len(continuous) + len(activity) + len(excluded):
        raise ValueError("state feature classes must be disjoint")
    if source_order != tuple(DISPATCH_ORDER):
        raise ValueError("source_dispatch_order does not match frozen dispatch order")
    threshold = _float(payload["activity_threshold"], "activity_threshold")
    if threshold < 0:
        raise ValueError("activity_threshold must be non-negative")
    if set(continuous) | set(excluded) != set(DISPATCH_ORDER) - {"slack_e", "slack_c", "slack_h", "q_dump"} | {"slack_e", "slack_c", "slack_h", "q_dump"}:
        raise ValueError("state ledger does not cover the frozen dispatch fields")
    if set(excluded) != {"slack_e", "slack_c", "slack_h", "q_dump"}:
        raise ValueError("excluded diagnostics are not frozen")
    return StateFeatureSpec(continuous, activity, excluded, threshold, source_order)


def validate_formal_v4_payload(payload: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    """Validate the JSON payload without reading result data."""

    if not isinstance(payload, Mapping):
        raise ValueError("formal v4 contract root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, SCHEMA_VERSION_V41}:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r} or {SCHEMA_VERSION_V41!r}")
    _strict_keys(payload, _ROOT_FIELDS_V41 if schema_version == SCHEMA_VERSION_V41 else _ROOT_FIELDS, "formal v4 contract")
    if schema_version == SCHEMA_VERSION_V41:
        if not isinstance(payload["source_manifest_required"], bool) or payload["source_manifest_required"] is not True:
            raise ValueError("formal v4.1 source_manifest_required must be true")
        for field in ("invalid_run_registry", "benchmark_rule_config", "source_closure_file"):
            _nonempty_text(payload[field], field)
    if payload["protocol_status"] not in {"frozen_candidate", "gate0_authorized", "gate1_frozen", "gate2_authorized"}:
        raise ValueError("protocol_status is invalid")
    if _int(payload["lookback"], "lookback", positive=True) != 24 or _int(payload["horizon"], "horizon", positive=True) != 4:
        raise ValueError("lookback/horizon must equal 24/4")
    if tuple(_nonempty_text(x, "task_order[]") for x in _sequence(payload["task_order"], "task_order")) != tuple(TASK_ORDER):
        raise ValueError("task_order does not match frozen order")
    if tuple(_nonempty_text(x, "exog_order[]") for x in _sequence(payload["exog_order"], "exog_order")) != tuple(EXOG_ORDER):
        raise ValueError("exog_order does not match frozen order")
    if tuple(_nonempty_text(x, "dispatch_order[]") for x in _sequence(payload["dispatch_order"], "dispatch_order")) != tuple(DISPATCH_ORDER):
        raise ValueError("dispatch_order does not match frozen order")
    if tuple(_nonempty_text(x, "status_order[]") for x in _sequence(payload["status_order"], "status_order")) != tuple(STATUS_ORDER):
        raise ValueError("status_order does not match frozen order")
    if tuple(_int(x, "seeds[]", positive=True) for x in _sequence(payload["seeds"], "seeds")) != SEEDS:
        raise ValueError("seeds must equal [2026, 2027, 2028, 2029, 2030]")
    if tuple(_int(x, "train_years[]") for x in _sequence(payload["train_years"], "train_years")) != TRAIN_YEARS:
        raise ValueError("train_years must equal [2015, 2016, 2017, 2018]")
    if _int(payload["selection_year"], "selection_year") != SELECTION_YEAR or _int(payload["evaluation_year"], "evaluation_year") != EVALUATION_YEAR:
        raise ValueError("selection_year/evaluation_year are not frozen")
    if _bool(payload["allow_test_access"], "allow_test_access") is not False:
        raise ValueError("allow_test_access must be false")
    if _bool(payload["allow_evaluation_access_before_gate3"], "allow_evaluation_access_before_gate3") is not False:
        raise ValueError("allow_evaluation_access_before_gate3 must be false")

    claims = payload["claims"]
    if not isinstance(claims, Mapping):
        raise ValueError("claims must be an object")
    _strict_keys(claims, {"price_carbon_response"}, "claims")
    if _bool(claims["price_carbon_response"], "claims.price_carbon_response"):
        raise ValueError("price/carbon response claim is not allowed in formal v4")

    methods = payload["methods"]
    if not isinstance(methods, list):
        raise ValueError("methods must be an array")
    names: list[str] = []
    for index, raw in enumerate(methods):
        if not isinstance(raw, Mapping):
            raise ValueError(f"methods[{index}] must be an object")
        _strict_keys(raw, _METHOD_FIELDS, f"methods[{index}]")
        name = _nonempty_text(raw["name"], f"methods[{index}].name")
        if name in names:
            raise ValueError(f"duplicate method: {name}")
        names.append(name)
        if raw["kind"] not in _METHOD_KINDS:
            raise ValueError(f"methods[{index}].kind is invalid")
        for field in ("trainable", "stochastic", "online_exact_lp"):
            _bool(raw[field], f"methods[{index}].{field}")
        _nonempty_text(raw["reproduction_level"], f"methods[{index}].reproduction_level")
        _nonempty_text(raw["comparison_role"], f"methods[{index}].comparison_role")
    if tuple(names) != FORMAL_V4_METHODS:
        raise ValueError(f"methods must use frozen order {FORMAL_V4_METHODS!r}")
    trainable = tuple(_nonempty_text(x, "trainable_methods[]") for x in _sequence(payload["trainable_methods"], "trainable_methods"))
    if trainable != TRAINABLE_METHODS:
        raise ValueError(f"trainable_methods must use frozen order {TRAINABLE_METHODS!r}")
    by_name = {raw["name"]: raw for raw in methods}
    if by_name["RSC-PF"]["online_exact_lp"] is not False or by_name["Direct-Policy"]["online_exact_lp"] is not False:
        raise ValueError("optimizer-free neural methods must have zero online exact-LP calls")

    guardrails = payload["forecast_guardrails"]
    if not isinstance(guardrails, Mapping):
        raise ValueError("forecast_guardrails must be an object")
    _strict_keys(guardrails, {"rigid_macro_wape_ratio_max", "rigid_per_task_wape_ratio_max", "gas_wape_ratio_max"}, "forecast_guardrails")
    if _float(guardrails["rigid_macro_wape_ratio_max"], "rigid_macro_wape_ratio_max") != 1.02 or _float(guardrails["rigid_per_task_wape_ratio_max"], "rigid_per_task_wape_ratio_max") != 1.05 or _float(guardrails["gas_wape_ratio_max"], "gas_wape_ratio_max") != 1.10:
        raise ValueError("forecast guardrails are not frozen")

    training = payload["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object")
    _strict_keys(training, {"stage_p_max_epochs", "stage_s_max_epochs", "stage_j_max_epochs", "patience", "validation_interval", "rollin_start_fraction", "model_history_fraction", "rollin_refresh_epochs", "imitation_end_epoch", "first_step_weights", "forecast_task_weights"}, "training")
    for field in ("stage_p_max_epochs", "stage_s_max_epochs", "stage_j_max_epochs", "patience", "validation_interval", "rollin_refresh_epochs", "imitation_end_epoch"):
        _int(training[field], f"training.{field}", positive=True)
    if tuple(_int(training["stage_p_max_epochs"], "stage_p_max_epochs") for _ in (0,)) != (30,) or _int(training["stage_s_max_epochs"], "stage_s_max_epochs") != 30 or _int(training["stage_j_max_epochs"], "stage_j_max_epochs") != 30:
        raise ValueError("stage epoch ceilings are not frozen")
    for field in ("rollin_start_fraction", "model_history_fraction"):
        value = _float(training[field], f"training.{field}")
        if not 0.0 < value < 1.0:
            raise ValueError(f"training.{field} must be in (0,1)")
    weights = tuple(_float(x, "training.first_step_weights[]") for x in _sequence(training["first_step_weights"], "training.first_step_weights"))
    if weights != (0.5, 1 / 6, 1 / 6, 1 / 6):
        raise ValueError("first_step_weights are not frozen")
    task_weights = tuple(_float(x, "training.forecast_task_weights[]") for x in _sequence(training["forecast_task_weights"], "training.forecast_task_weights"))
    if task_weights != (1.0, 1.0, 1.0, 0.25):
        raise ValueError("forecast_task_weights are not frozen")

    selection = payload["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be an object")
    _strict_keys(selection, {"selection_split", "rigid_task_indices", "checkpoint_score", "engineering_guardrail", "statistical_noninferiority_claim", "gate1_origins", "gate1_short_rollout_hours", "gate2_seeds", "gate3_seeds"}, "selection")
    if selection["selection_split"] != "selection" or tuple(selection["rigid_task_indices"]) != (0, 1, 2) or selection["checkpoint_score"] != "rolling_penalized_objective":
        raise ValueError("selection semantics are not frozen")
    if selection["engineering_guardrail"] is not True or selection["statistical_noninferiority_claim"] is not False:
        raise ValueError("selection guardrail semantics are not frozen")
    if _int(selection["gate1_origins"], "selection.gate1_origins", positive=True) != 1000 or _int(selection["gate1_short_rollout_hours"], "selection.gate1_short_rollout_hours", positive=True) != 168:
        raise ValueError("Gate 1 selection budget is not frozen")
    if tuple(_int(x, "selection.gate2_seeds", positive=True) for x in selection["gate2_seeds"]) != (2026, 2027, 2028) or tuple(_int(x, "selection.gate3_seeds", positive=True) for x in selection["gate3_seeds"]) != SEEDS:
        raise ValueError("Gate 2/Gate 3 seeds are not frozen")

    statistics = payload["statistics"]
    if not isinstance(statistics, Mapping):
        raise ValueError("statistics must be an object")
    _strict_keys(statistics, {"independent_unit", "paired_unit", "block_hours", "bootstrap_replicates", "alpha", "primary_contrast", "secondary_comparisons_are_descriptive"}, "statistics")
    if statistics["independent_unit"] != "model_seed" or _int(statistics["block_hours"], "statistics.block_hours", positive=True) != 168 or _int(statistics["bootstrap_replicates"], "statistics.bootstrap_replicates", positive=True) != 2000 or _float(statistics["alpha"], "statistics.alpha") != 0.05 or statistics["secondary_comparisons_are_descriptive"] is not True:
        raise ValueError("statistics settings are not frozen")

    resource = payload["resource_gate"]
    if not isinstance(resource, Mapping):
        raise ValueError("resource_gate must be an object")
    _strict_keys(resource, {"benchmark_solves", "max_projected_p95_hours", "minimum_disk_margin_fraction"}, "resource_gate")
    if _int(resource["benchmark_solves"], "resource_gate.benchmark_solves", positive=True) != 500 or _float(resource["max_projected_p95_hours"], "resource_gate.max_projected_p95_hours") != 24.0 or _float(resource["minimum_disk_margin_fraction"], "resource_gate.minimum_disk_margin_fraction") != 0.20:
        raise ValueError("resource gate is not frozen")

    capacity = payload["capacity"]
    if not isinstance(capacity, Mapping):
        raise ValueError("capacity must be an object")
    _strict_keys(capacity, {"candidate_multipliers", "cooling_shortage_energy_ratio_max", "cooling_shortage_hour_rate_max", "main_scenario_name", "stress_scenario_name"}, "capacity")
    multipliers = tuple(_float(x, "capacity.candidate_multipliers[]", positive=True) for x in _sequence(capacity["candidate_multipliers"], "capacity.candidate_multipliers"))
    expected_multipliers = tuple(1.0 + 0.1 * i for i in range(11))
    if len(multipliers) != len(expected_multipliers) or any(abs(a - b) > 1e-12 for a, b in zip(multipliers, expected_multipliers)) or _float(capacity["cooling_shortage_energy_ratio_max"], "capacity.cooling_shortage_energy_ratio_max") != 0.005 or _float(capacity["cooling_shortage_hour_rate_max"], "capacity.cooling_shortage_hour_rate_max") != 0.01:
        raise ValueError("capacity audit settings are not frozen")

    renewable = payload["renewable_forecast"]
    if not isinstance(renewable, Mapping):
        raise ValueError("renewable_forecast must be an object")
    _strict_keys(renewable, {"method", "history_hours", "horizon", "selection_locked"}, "renewable_forecast")
    if renewable["method"] != "last_value_persistence" or _int(renewable["history_hours"], "renewable_forecast.history_hours", positive=True) != 1 or _int(renewable["horizon"], "renewable_forecast.horizon", positive=True) != 4 or renewable["selection_locked"] is not True:
        raise ValueError("renewable forecast is not frozen")

    safety = payload["safety"]
    if not isinstance(safety, Mapping):
        raise ValueError("safety must be an object")
    _strict_keys(safety, {"normalization_fit_years", "allow_output_overwrite", "allow_legacy_overwrite", "allow_future_binary_decisions", "online_exact_lp_calls_rsc_pf", "no_price_carbon_response_claim", "evaluation_requires_gate3"}, "safety")
    if tuple(_int(x, "safety.normalization_fit_years[]") for x in safety["normalization_fit_years"]) != TRAIN_YEARS or safety["allow_output_overwrite"] is not False or safety["allow_legacy_overwrite"] is not False or safety["allow_future_binary_decisions"] is not False or _int(safety["online_exact_lp_calls_rsc_pf"], "safety.online_exact_lp_calls_rsc_pf") != 0 or safety["no_price_carbon_response_claim"] is not True or safety["evaluation_requires_gate3"] is not True:
        raise ValueError("safety settings are not frozen")

    paths = payload["paths"]
    if not isinstance(paths, Mapping):
        raise ValueError("paths must be an object")
    _strict_keys(paths, {"output_root", "data_root", "audit_root", "authorization", "benchmark_path", "parameter_ledger_path", "external_registry_path"}, "paths")
    for name, value in paths.items():
        _nonempty_text(value, f"paths.{name}")
    if repo_root is not None:
        output_root = _resolve_path(paths["output_root"], repo_root=repo_root)
        _under(output_root, repo_root, "paths.output_root")
        for name in ("data_root", "audit_root", "authorization"):
            _under(_resolve_path(paths[name], repo_root=repo_root), output_root, f"paths.{name}")
        if schema_version == SCHEMA_VERSION_V41:
            for name in ("invalid_run_registry", "benchmark_rule_config", "source_closure_file"):
                _under(_resolve_path(payload[name], repo_root=repo_root), repo_root, name)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return payload


def load_formal_v4_spec(path: str | Path) -> FormalV4Spec:
    source = Path(path).resolve()
    repo_root = source.parent.parent
    payload = _load_json(source)
    validate_formal_v4_payload(payload, repo_root=repo_root)
    state_path = _resolve_path(payload["state_feature_config"], repo_root=repo_root)
    state = _validate_state_payload(_load_json(state_path))
    budget_path = repo_root / "configs" / "joint_dispatch_search_budget_v4.json"
    budget_payload = _load_json(budget_path)
    _strict_keys(budget_payload, {"schema_version", "methods"}, "search budget contract")
    if budget_payload["schema_version"] != "joint-dispatch-search-budget-v4" or not isinstance(budget_payload["methods"], Mapping):
        raise ValueError("search budget contract is invalid")
    if tuple(budget_payload["methods"]) != TRAINABLE_METHODS:
        raise ValueError("search budgets must cover exactly the frozen trainable methods")
    budgets: dict[str, SearchBudgetV4] = {}
    for name in TRAINABLE_METHODS:
        raw = budget_payload["methods"][name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"search budget for {name} must be an object")
        _strict_keys(raw, {"max_trials", "max_epochs", "patience", "validation_interval"}, f"search_budgets.{name}")
        budgets[name] = SearchBudgetV4(
            _int(raw["max_trials"], f"search_budgets.{name}.max_trials", positive=True),
            _int(raw["max_epochs"], f"search_budgets.{name}.max_epochs", positive=True),
            _int(raw["patience"], f"search_budgets.{name}.patience", positive=True),
            _int(raw["validation_interval"], f"search_budgets.{name}.validation_interval", positive=True),
        )
    if any(item.max_trials != 4 for item in budgets.values()):
        raise ValueError("every trainable method must have max_trials=4")

    methods = tuple(str(raw["name"]) for raw in payload["methods"])
    method_specs = {
        str(raw["name"]): MethodV4Spec(
            name=str(raw["name"]), kind=str(raw["kind"]), trainable=bool(raw["trainable"]),
            stochastic=bool(raw["stochastic"]), online_exact_lp=bool(raw["online_exact_lp"]),
            reproduction_level=str(raw["reproduction_level"]), comparison_role=str(raw["comparison_role"]),
        )
        for raw in payload["methods"]
    }
    paths = {str(key): _resolve_path(value, repo_root=repo_root) for key, value in payload["paths"].items()}
    invalid_run_registry = _resolve_path(payload["invalid_run_registry"], repo_root=repo_root) if "invalid_run_registry" in payload else None
    benchmark_rule_config = _resolve_path(payload["benchmark_rule_config"], repo_root=repo_root) if "benchmark_rule_config" in payload else None
    source_closure_file = _resolve_path(payload["source_closure_file"], repo_root=repo_root) if "source_closure_file" in payload else None
    return FormalV4Spec(
        schema_version=str(payload["schema_version"]), protocol_status=str(payload["protocol_status"]),
        lookback=int(payload["lookback"]), horizon=int(payload["horizon"]), task_order=tuple(payload["task_order"]),
        exog_order=tuple(payload["exog_order"]), dispatch_order=tuple(payload["dispatch_order"]),
        status_order=tuple(payload["status_order"]), seeds=SEEDS, train_years=TRAIN_YEARS,
        selection_year=SELECTION_YEAR, evaluation_year=EVALUATION_YEAR,
        allow_test_access=False, allow_evaluation_access_before_gate3=False,
        claims=MappingProxyType(dict(payload["claims"])), methods=methods,
        trainable_methods=TRAINABLE_METHODS, method_specs=MappingProxyType(method_specs),
        search_budgets=MappingProxyType(budgets),
        guardrails=ForecastGuardrails(**{key: float(value) for key, value in payload["forecast_guardrails"].items()}),
        state_features=state, training=MappingProxyType(dict(payload["training"])),
        selection=MappingProxyType(dict(payload["selection"])), statistics=MappingProxyType(dict(payload["statistics"])),
        resource_gate=MappingProxyType(dict(payload["resource_gate"])), capacity=MappingProxyType(dict(payload["capacity"])),
        renewable_forecast=MappingProxyType(dict(payload["renewable_forecast"])), paths=MappingProxyType(paths),
        safety=MappingProxyType(dict(payload["safety"])), source_path=source,
        invalid_run_registry=invalid_run_registry, benchmark_rule_config=benchmark_rule_config,
        source_closure_file=source_closure_file,
        source_manifest_required=bool(payload.get("source_manifest_required", False)),
    )


def build_formal_v4_run_matrix(
    spec: FormalV4Spec,
    *,
    phase: str = "selection",
    run_id: str = "formal_v4_20260903",
    gate3_authorized: bool = False,
) -> list[dict[str, Any]]:
    if phase not in {"calibration", "selection", "evaluation"}:
        raise ValueError("phase must be calibration, selection or evaluation")
    if not run_id or any(token in run_id for token in ("/", "\\", "..")):
        raise ValueError("run_id must be a simple non-empty directory name")
    if phase == "evaluation" and (spec.allow_evaluation_access_before_gate3 or not gate3_authorized):
        raise PermissionError("Gate 3 authorization is required for formal v4 evaluation")
    rows: list[dict[str, Any]] = []
    for name in spec.methods:
        method = spec.method(name)
        if phase == "calibration":
            seeds: tuple[int | None, ...] = (SEEDS[0],) if method.stochastic else (None,)
        else:
            seeds = spec.seeds if method.stochastic else (None,)
        for seed in seeds:
            safe = name.replace("-", "_").replace(" ", "_")
            leaf = "deterministic" if seed is None else f"seed_{seed}"
            rows.append({
                "run_id": run_id, "phase": phase, "year": spec.evaluation_year if phase == "evaluation" else spec.selection_year,
                "method": name, "seed": seed, "stochastic": method.stochastic,
                "online_exact_lp": method.online_exact_lp, "comparison_role": method.comparison_role,
                "reproduction_level": method.reproduction_level,
                "output_directory": str(spec.paths["output_root"] / run_id / phase / safe / leaf),
            })
    keys = [(row["phase"], row["method"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("formal v4 run matrix contains duplicates")
    return rows


__all__ = [
    "EVALUATION_YEAR", "FORMAL_V4_METHODS", "ForecastGuardrails", "FormalV4Spec",
    "MethodV4Spec", "SEEDS", "SELECTION_YEAR", "SearchBudgetV4", "StateFeatureSpec",
    "SCHEMA_VERSION", "SCHEMA_VERSION_V41", "TRAINABLE_METHODS", "TRAIN_YEARS",
    "build_formal_v4_run_matrix", "load_formal_v4_spec",
    "validate_formal_v4_payload",
]
