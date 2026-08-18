"""Strict contract and metadata helpers for the pure-simulation scheduling proxy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .dispatch_schema import VARIABLES


FEATURE_ORDER: Tuple[str, ...] = (
    "electricity",
    "cooling",
    "heating",
    "gas_prior",
    "pv_available",
    "wt_available",
    "grid_price",
    "gas_price",
    "carbon_price",
    "initial_soc",
)
LABEL_ORDER: Tuple[str, ...] = tuple(VARIABLES)
HORIZON = 4
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "configs" / "scheduling_proxy_contract_v1.json"


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze the top-level mapping while preserving JSON-friendly values."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed: int
    size: int

    def to_dict(self) -> dict[str, int]:
        return {"seed": int(self.seed), "size": int(self.size)}


@dataclass(frozen=True)
class ProxyContract:
    """Validated immutable scheduling-proxy contract.

    Nested mappings are copied at load time.  Callers should use ``to_dict``
    when serializing rather than mutating contract internals.
    """

    schema_version: str
    generator_version: str
    source_type: str
    horizon: int
    feature_order: Tuple[str, ...]
    label_order: Tuple[str, ...]
    benchmark_sha256: str
    splits: Mapping[str, SplitSpec]
    smoke_splits: Mapping[str, SplitSpec]
    gas_prior: Mapping[str, Any]
    model: Mapping[str, Any]
    loss_weights: Mapping[str, float]
    normalization: Mapping[str, float]
    training: Mapping[str, Any]
    safety: Mapping[str, Any]
    required_artifacts: Tuple[str, ...]
    contract_path: Optional[str] = None

    @property
    def input_dim(self) -> int:
        return len(self.feature_order)

    @property
    def output_dim(self) -> int:
        return len(self.label_order)

    @property
    def is_v2(self) -> bool:
        """Whether this contract selects the feasible decoder v2 path."""

        return self.schema_version == "scheduling-proxy-contract-v2"

    @property
    def decision_dim(self) -> int:
        return int(self.model.get("decision_dim", self.output_dim))

    @property
    def decision_groups(self) -> Mapping[str, Any]:
        groups = self.model.get("decision_groups", {})
        return groups if isinstance(groups, Mapping) else {}

    @property
    def decoder_schema_version(self) -> str | None:
        value = self.model.get("decoder_schema_version")
        return None if value is None else str(value)

    @property
    def decoder_dtype(self) -> str | None:
        value = self.model.get("decoder_dtype")
        return None if value is None else str(value)

    @property
    def control_temperature(self) -> float | None:
        value = self.model.get("control_temperature")
        return None if value is None else float(value)

    @property
    def benchmark_hash(self) -> str:
        return self.benchmark_sha256

    @property
    def hidden_width(self) -> int:
        return int(self.model["hidden_width"])

    @property
    def residual_blocks(self) -> int:
        return int(self.model["residual_blocks"])

    @property
    def dropout(self) -> float:
        return float(self.model["dropout"])

    def split(self, name: str, smoke: bool = False) -> SplitSpec:
        source = self.smoke_splits if smoke else self.splits
        if name not in source:
            raise KeyError(f"unknown proxy split: {name}")
        return source[name]

    def to_dict(self) -> dict[str, Any]:
        def thaw(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): thaw(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [thaw(item) for item in value]
            return value

        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "source_type": self.source_type,
            "horizon": self.horizon,
            "feature_order": list(self.feature_order),
            "label_order": list(self.label_order),
            "benchmark_sha256": self.benchmark_sha256,
            "splits": {name: spec.to_dict() for name, spec in self.splits.items()},
            "smoke_splits": {name: spec.to_dict() for name, spec in self.smoke_splits.items()},
            "gas_prior": thaw(self.gas_prior),
            "model": thaw(self.model),
            "loss_weights": thaw(self.loss_weights),
            "normalization": thaw(self.normalization),
            "training": thaw(self.training),
            "safety": thaw(self.safety),
            "required_artifacts": list(self.required_artifacts),
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: ProxyContract | Mapping[str, Any] | str | Path) -> str:
    """Return a canonical hash for contract provenance."""

    if isinstance(contract, ProxyContract):
        data: Mapping[str, Any] = contract.to_dict()
    elif isinstance(contract, (str, Path)):
        with Path(contract).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = contract
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0 or float(value) != float(result):
        raise ValueError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if result < 0 or numeric != float(result):
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _positive_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative finite number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a non-negative finite number")
    return result


def validate_contract(data: Mapping[str, Any], benchmark_path: str | Path | None = None) -> None:
    """Validate the contract schema and, when supplied, benchmark digest."""

    if not isinstance(data, Mapping):
        raise ValueError("proxy contract must be an object")
    required = {
        "schema_version", "generator_version", "source_type", "horizon", "feature_order",
        "label_order", "benchmark_sha256", "splits", "smoke_splits", "gas_prior", "model",
        "loss_weights", "normalization", "training", "safety", "required_artifacts",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"proxy contract missing fields: {missing}")
    schema_version = str(data["schema_version"])
    if schema_version not in {"scheduling-proxy-contract-v1", "scheduling-proxy-contract-v2"}:
        raise ValueError("schema_version must be scheduling-proxy-contract-v1 or scheduling-proxy-contract-v2")
    if data["source_type"] != "pure_simulation":
        raise ValueError("source_type must be pure_simulation")
    if _positive_int(data["horizon"], "horizon") != HORIZON:
        raise ValueError("horizon must be 4")
    features = tuple(data["feature_order"])
    labels = tuple(data["label_order"])
    if features != FEATURE_ORDER:
        raise ValueError(f"feature_order must be exactly {FEATURE_ORDER}")
    if labels != LABEL_ORDER:
        raise ValueError(f"label_order must be dispatch_schema.VARIABLES: {LABEL_ORDER}")
    if len(set(features)) != len(features) or len(set(labels)) != len(labels):
        raise ValueError("feature_order and label_order must not contain duplicates")
    benchmark_hash = str(data["benchmark_sha256"]).lower()
    if len(benchmark_hash) != 64 or any(char not in "0123456789abcdef" for char in benchmark_hash):
        raise ValueError("benchmark_sha256 must be a SHA-256 hex digest")
    if benchmark_path is not None:
        actual = file_sha256(benchmark_path)
        if actual.lower() != benchmark_hash:
            raise ValueError(f"benchmark SHA-256 mismatch: contract={benchmark_hash}, actual={actual}")

    def check_splits(raw: Any, expected: Mapping[str, tuple[int, int]], field: str) -> None:
        mapping = _mapping(raw, field)
        if set(mapping) != set(expected):
            raise ValueError(f"{field} must contain exactly {tuple(expected)}")
        seeds = []
        for name, (expected_seed, expected_size) in expected.items():
            spec = _mapping(mapping[name], f"{field}.{name}")
            seed = _positive_int(spec.get("seed"), f"{field}.{name}.seed")
            size = _positive_int(spec.get("size"), f"{field}.{name}.size")
            if (seed, size) != (expected_seed, expected_size):
                raise ValueError(f"{field}.{name} must be seed={expected_seed}, size={expected_size}")
            seeds.append(seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{field} seeds must be disjoint")

    check_splits(data["splits"], {"train": (2026, 8192), "validation": (2027, 2048), "test": (2028, 2048)}, "splits")
    check_splits(data["smoke_splits"], {"train": (2026, 64), "validation": (2027, 16), "test": (2028, 16)}, "smoke_splits")

    gas = _mapping(data["gas_prior"], "gas_prior")
    if gas.get("role") != "auxiliary_station_side_gas_consumption":
        raise ValueError("gas_prior.role must describe auxiliary station-side gas consumption")
    if gas.get("teacher_source") != "g_chp_plus_g_gb":
        raise ValueError("gas_prior.teacher_source must be g_chp_plus_g_gb")
    if gas.get("is_balance_equation") is not False:
        raise ValueError("gas_prior must never be a fourth balance equation")
    for name in ("noise_fraction", "additive_noise_scale", "mask_fraction"):
        value = _nonnegative_float(gas.get(name), f"gas_prior.{name}")
        if name == "mask_fraction" and value >= 1.0:
            raise ValueError("gas_prior.mask_fraction must be below one")

    model = _mapping(data["model"], "model")
    if _positive_int(model.get("hidden_width"), "model.hidden_width") != 128:
        raise ValueError("model.hidden_width must be 128")
    if _positive_int(model.get("residual_blocks"), "model.residual_blocks") != 2:
        raise ValueError("model.residual_blocks must be 2")
    dropout = float(model.get("dropout"))
    if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0,1)")
    if model.get("device") != "cpu":
        raise ValueError("model.device must be cpu")

    weights = _mapping(data["loss_weights"], "loss_weights")
    if schema_version == "scheduling-proxy-contract-v2":
        expected_weights = {"coordinate", "renewable_split", "objective", "carbon", "slack"}
    else:
        expected_weights = {"dispatch", "balance", "conversion", "soc", "cost", "carbon", "gas_prior"}
    if set(weights) != expected_weights:
        raise ValueError(f"loss_weights must contain exactly {sorted(expected_weights)}")
    for name, value in weights.items():
        _nonnegative_float(value, f"loss_weights.{name}")
    normalization = _mapping(data["normalization"], "normalization")
    for name in ("epsilon", "scale_floor"):
        _positive_float(normalization.get(name, 0.0), f"normalization.{name}")
    training = _mapping(data["training"], "training")
    for name in ("batch_size", "max_epochs", "patience"):
        _positive_int(training.get(name), f"training.{name}")
    _nonnegative_int(training.get("num_workers", -1), "training.num_workers")
    for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        _positive_float(training.get(name, 0.0), f"training.{name}")
    _positive_int(training.get("seed"), "training.seed")
    safety = _mapping(data["safety"], "safety")
    _positive_float(safety.get("feasibility_tolerance", 0.0), "safety.feasibility_tolerance")
    if not isinstance(safety.get("allow_exact_fallback"), bool):
        raise ValueError("safety.allow_exact_fallback must be boolean")
    if schema_version == "scheduling-proxy-contract-v2":
        if _positive_int(model.get("decision_dim"), "model.decision_dim") != 15:
            raise ValueError("v2 model.decision_dim must be 15")
        if model.get("output_parameterization") != "horizon_reachable_feasible_v2":
            raise ValueError("v2 model.output_parameterization must be horizon_reachable_feasible_v2")
        if model.get("decoder_schema_version") != "horizon-reachable-feasible-v2":
            raise ValueError("v2 model.decoder_schema_version is invalid")
        if model.get("decoder_dtype") != "float64":
            raise ValueError("v2 model.decoder_dtype must be float64")
        temperature = _positive_float(model.get("control_temperature"), "model.control_temperature")
        if temperature != 0.25:
            raise ValueError("v2 model.control_temperature must be 0.25")
        expected_order = ("cooling", "chp", "soc_0", "soc_1", "soc_2", "renewable_pv")
        if tuple(model.get("decision_order", ())) != expected_order:
            raise ValueError("v2 model.decision_order is invalid")
        groups = _mapping(model.get("decision_groups"), "model.decision_groups")
        expected_groups = {"cooling": (0, 4), "chp": (4, 8), "soc": (8, 11), "renewable_pv": (11, 15)}
        if set(groups) != set(expected_groups):
            raise ValueError("v2 model.decision_groups must cover all decision controls")
        ranges: list[tuple[int, int]] = []
        for name, expected in expected_groups.items():
            raw = groups[name]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
                raise ValueError(f"v2 model.decision_groups.{name} must be a [start,end] range")
            start = _nonnegative_int(raw[0], f"model.decision_groups.{name}.start")
            end = _positive_int(raw[1], f"model.decision_groups.{name}.end")
            if (start, end) != expected:
                raise ValueError(f"v2 model.decision_groups.{name} has invalid decision range")
            if end <= start:
                raise ValueError(f"v2 model.decision_groups.{name} must have positive width")
            ranges.append((start, end))
        if sorted(ranges) != [(0, 4), (4, 8), (8, 11), (11, 15)]:
            raise ValueError("v2 model.decision_groups must be contiguous")
        if safety.get("allow_exact_fallback") is not False:
            raise ValueError("v2 safety.allow_exact_fallback must be false")
        if _nonnegative_int(safety.get("inference_exact_lp_calls"), "safety.inference_exact_lp_calls") != 0:
            raise ValueError("v2 safety.inference_exact_lp_calls must be zero")
    artifacts = data["required_artifacts"]
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or not artifacts:
        raise ValueError("required_artifacts must be a non-empty list")


def load_contract(path: str | Path = DEFAULT_CONTRACT_PATH, benchmark_path: str | Path | None = None) -> ProxyContract:
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_contract(data, benchmark_path=benchmark_path)

    def make_splits(raw: Mapping[str, Any]) -> Mapping[str, SplitSpec]:
        return MappingProxyType({
            name: SplitSpec(name=name, seed=int(spec["seed"]), size=int(spec["size"]))
            for name, spec in raw.items()
        })

    return ProxyContract(
        schema_version=str(data["schema_version"]),
        generator_version=str(data["generator_version"]),
        source_type=str(data["source_type"]),
        horizon=int(data["horizon"]),
        feature_order=tuple(str(v) for v in data["feature_order"]),
        label_order=tuple(str(v) for v in data["label_order"]),
        benchmark_sha256=str(data["benchmark_sha256"]).lower(),
        splits=make_splits(data["splits"]),
        smoke_splits=make_splits(data["smoke_splits"]),
        gas_prior=_freeze_mapping(_mapping(data["gas_prior"], "gas_prior")),
        model=_freeze_mapping(_mapping(data["model"], "model")),
        loss_weights=_freeze_mapping({k: float(v) for k, v in _mapping(data["loss_weights"], "loss_weights").items()}),
        normalization=_freeze_mapping({k: float(v) for k, v in _mapping(data["normalization"], "normalization").items()}),
        training=_freeze_mapping(_mapping(data["training"], "training")),
        safety=_freeze_mapping(_mapping(data["safety"], "safety")),
        required_artifacts=tuple(str(v) for v in data["required_artifacts"]),
        contract_path=str(contract_path),
    )


# Explicit aliases make the public contract API discoverable for callers that
# use the plan's ``load_*`` naming convention.
load_proxy_contract = load_contract
validate_proxy_contract = validate_contract


__all__ = [
    "FEATURE_ORDER",
    "LABEL_ORDER",
    "HORIZON",
    "DEFAULT_CONTRACT_PATH",
    "SplitSpec",
    "ProxyContract",
    "file_sha256",
    "contract_sha256",
    "validate_contract",
    "validate_proxy_contract",
    "load_contract",
    "load_proxy_contract",
]
