"""Exact-label synthetic proxy datasets and train-only normalization."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .dispatch_lp import DispatchInputs, VARIABLES, solve_dispatch_lp
from .proxy_contract import FEATURE_ORDER, LABEL_ORDER, HORIZON, ProxyContract, contract_sha256, file_sha256
from .synthetic_scenarios import GENERATOR_VERSION, SyntheticScenarioBatch, load_benchmark


def scenario_id_digest(scenario_ids: np.ndarray) -> str:
    values = np.asarray(scenario_ids, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


@dataclass(frozen=True)
class LabeledProxySplit:
    """A successful exact-LP-labelled synthetic split."""

    inputs: np.ndarray  # [N,H,10], raw proxy features
    dispatch: np.ndarray  # [N,H,21], VARIABLE order
    teacher_objective: np.ndarray  # [N]
    teacher_cost: np.ndarray  # [N]
    teacher_carbon: np.ndarray  # [N]
    teacher_status: np.ndarray  # [N], exact HiGHS status
    teacher_message: np.ndarray  # [N], exact HiGHS message
    teacher_balance_residual: np.ndarray  # [N], max exact balance residual
    teacher_simultaneous_charge_discharge: np.ndarray  # [N], exact LP diagnostic
    demand: np.ndarray  # [N,H,3]
    pv_available: np.ndarray  # [N,H]
    wt_available: np.ndarray  # [N,H]
    prices: np.ndarray  # [N,H,3]
    initial_soc: np.ndarray  # [N]
    gas_prior_mask: np.ndarray  # [N,H]
    scenario_ids: np.ndarray  # [N]
    split: str
    seed: int
    rejected_count: int = 0
    attempted_count: int = 0
    benchmark_sha256: str = ""
    contract_sha256: str = ""
    source_type: str = "pure_simulation"
    generator_version: str = "synthetic-scheduling-v1"

    @property
    def n_samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def features(self) -> np.ndarray:
        return self.inputs

    @property
    def labels(self) -> np.ndarray:
        return self.dispatch

    def validate(self, require_provenance: bool = False) -> None:
        n = self.inputs.shape[0]
        if self.inputs.shape[1:] != (HORIZON, len(FEATURE_ORDER)):
            raise ValueError(f"inputs must have shape [N,4,{len(FEATURE_ORDER)}]")
        if self.dispatch.shape != (n, HORIZON, len(LABEL_ORDER)):
            raise ValueError(f"dispatch must have shape [N,4,{len(LABEL_ORDER)}]")
        if self.demand.shape != (n, HORIZON, 3):
            raise ValueError("demand must have shape [N,4,3]")
        if self.pv_available.shape != (n, HORIZON) or self.wt_available.shape != (n, HORIZON):
            raise ValueError("renewables must have shape [N,4]")
        if self.prices.shape != (n, HORIZON, 3):
            raise ValueError("prices must have shape [N,4,3]")
        if self.teacher_objective.shape != (n,) or self.teacher_cost.shape != (n,) or self.teacher_carbon.shape != (n,):
            raise ValueError("teacher objective/cost/carbon arrays must have shape [N]")
        if self.teacher_status.shape != (n,) or self.teacher_message.shape != (n,):
            raise ValueError("teacher status/message arrays must have shape [N]")
        if self.teacher_balance_residual.shape != (n,) or self.teacher_simultaneous_charge_discharge.shape != (n,):
            raise ValueError("teacher audit arrays must have shape [N]")
        if self.initial_soc.shape != (n,) or self.scenario_ids.shape != (n,):
            raise ValueError("initial_soc/scenario_ids must have shape [N]")
        if not np.issubdtype(np.asarray(self.scenario_ids).dtype, np.integer):
            raise ValueError("scenario_ids must be integer values")
        if self.gas_prior_mask.shape != (n, HORIZON):
            raise ValueError("gas_prior_mask must have shape [N,4]")
        arrays = (self.inputs, self.dispatch, self.teacher_objective, self.teacher_cost, self.teacher_carbon,
                  self.teacher_balance_residual, self.teacher_simultaneous_charge_discharge,
                  self.demand, self.pv_available, self.wt_available, self.prices, self.initial_soc)
        if any(not np.isfinite(np.asarray(array)).all() for array in arrays):
            raise ValueError("labelled dataset contains non-finite values")
        if (self.dispatch < -1e-7).any():
            raise ValueError("LP dispatch labels must be non-negative")
        statuses = np.asarray(self.teacher_status).astype(str)
        messages = np.asarray(self.teacher_message).astype(str)
        if not np.all(statuses == "optimal"):
            raise ValueError("accepted labelled splits must contain only optimal teacher statuses")
        if any(not isinstance(message, str) for message in messages.tolist()):
            raise ValueError("teacher messages must be strings")
        if (self.teacher_balance_residual < 0).any() or (self.teacher_simultaneous_charge_discharge < 0).any():
            raise ValueError("teacher audit diagnostics must be non-negative")
        if not np.isin(self.gas_prior_mask, [0.0, 1.0]).all():
            raise ValueError("gas_prior_mask must be binary")
        if self.seed <= 0 or self.rejected_count < 0 or self.attempted_count != self.n_samples + self.rejected_count:
            raise ValueError("invalid solve accounting")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if self.source_type != "pure_simulation":
            raise ValueError("source_type must be pure_simulation")
        if not self.generator_version:
            raise ValueError("generator_version is required")
        if len(np.unique(self.scenario_ids)) != n:
            raise ValueError("scenario_ids must be unique within a split")
        if require_provenance:
            if not _is_sha256(self.benchmark_sha256) or not _is_sha256(self.contract_sha256):
                raise ValueError("benchmark_sha256 and contract_sha256 are required split provenance")

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": int(self.seed),
            "sample_count": self.n_samples,
            "attempted_count": int(self.attempted_count),
            "rejected_solve_count": int(self.rejected_count),
            "benchmark_sha256": self.benchmark_sha256,
            "contract_sha256": self.contract_sha256,
            "feature_order": list(FEATURE_ORDER),
            "label_order": list(LABEL_ORDER),
            "teacher_status": {str(status): int(count) for status, count in zip(*np.unique(self.teacher_status.astype(str), return_counts=True))},
            "teacher_balance_residual_max": float(np.max(self.teacher_balance_residual)),
            "teacher_simultaneous_charge_discharge_max": float(np.max(self.teacher_simultaneous_charge_discharge)),
            "source_type": self.source_type,
            "generator_version": self.generator_version,
            "scenario_id_digest": scenario_id_digest(self.scenario_ids),
            "scenario_id_min": int(np.min(self.scenario_ids)),
            "scenario_id_max": int(np.max(self.scenario_ids)),
            "gas_prior_is_balance_equation": False,
            "uses_observed_windows": False,
            "normalization_fit_split": "train",
        }


def _params_for_scenario(values: Mapping[str, Any], prices: np.ndarray) -> dict[str, float]:
    params = {str(k): float(v) for k, v in values.items()}
    # The frozen teacher has scalar price parameters.  Synthetic prices are
    # scenario-level constants broadcast over H, so selecting their mean keeps
    # the LP interface exact without modifying the teacher equations.
    params["grid_energy_price"] = float(np.mean(prices[:, 0]))
    params["gas_energy_price"] = float(np.mean(prices[:, 1]))
    params["carbon_price"] = float(np.mean(prices[:, 2]))
    return params


def _dispatch_array(values: Mapping[str, np.ndarray], horizon: int = HORIZON) -> np.ndarray:
    result = np.stack([np.asarray(values[name], dtype=np.float64) for name in LABEL_ORDER], axis=-1)
    if result.shape != (horizon, len(LABEL_ORDER)):
        raise ValueError(f"teacher labels must have shape [{horizon},{len(LABEL_ORDER)}]")
    return result


def _physical_cost(dispatch: np.ndarray, prices: np.ndarray, params: Mapping[str, float]) -> float:
    names = {name: idx for idx, name in enumerate(LABEL_ORDER)}
    grid = dispatch[:, names["grid"]]
    gas = dispatch[:, names["g_chp"]] + dispatch[:, names["g_gb"]]
    throughput = dispatch[:, names["p_charge"]] + dispatch[:, names["p_discharge"]]
    slack = dispatch[:, [names["slack_e"], names["slack_c"], names["slack_h"]]].sum(axis=1)
    return float(np.sum(grid * prices[:, 0] + gas * prices[:, 1] + throughput * float(params["bess_throughput_cost"]) + slack * float(params["unserved_penalty"])))


def _carbon(dispatch: np.ndarray, params: Mapping[str, float]) -> float:
    names = {name: idx for idx, name in enumerate(LABEL_ORDER)}
    grid = dispatch[:, names["grid"]]
    gas = dispatch[:, names["g_chp"]] + dispatch[:, names["g_gb"]]
    return float(np.sum(grid * float(params.get("grid_emission_factor", 0.0)) + gas * float(params.get("gas_emission_factor", 0.0))))


def _make_gas_prior(
    dispatch: np.ndarray,
    contract: ProxyContract | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = dispatch.shape[0]
    rng = np.random.default_rng(int(seed))
    names = {name: idx for idx, name in enumerate(LABEL_ORDER)}
    gas_purchase = dispatch[:, :, names["g_chp"]] + dispatch[:, :, names["g_gb"]]
    noise_fraction = float(contract.gas_prior["noise_fraction"]) if contract is not None else 0.08
    additive_scale = float(contract.gas_prior["additive_noise_scale"]) if contract is not None else 0.02
    mask_fraction = float(contract.gas_prior["mask_fraction"]) if contract is not None else 0.15
    scale = np.maximum(gas_purchase, 1.0)
    prior = gas_purchase * (1.0 + rng.normal(0.0, noise_fraction, size=gas_purchase.shape))
    prior += rng.normal(0.0, additive_scale, size=gas_purchase.shape) * scale
    prior = np.maximum(prior, 0.0)
    mask = (rng.random(gas_purchase.shape) >= mask_fraction).astype(np.float64)
    prior *= mask
    return prior, mask


def build_labeled_proxy_split(
    scenarios: SyntheticScenarioBatch,
    benchmark: Mapping[str, Any] | str | Path,
    contract: ProxyContract | None = None,
    gas_prior_seed: int | None = None,
) -> LabeledProxySplit:
    """Solve each synthetic scenario with the frozen exact LP.

    Failed/non-finite solves are rejected and counted; they are never replaced
    with zero labels.  The gas prior RNG is separate from the scenario RNG and
    runs only after all accepted exact labels have been assembled.
    """

    if isinstance(benchmark, (str, Path)):
        benchmark_data = load_benchmark(benchmark)
        benchmark_hash = file_sha256(benchmark)
    else:
        benchmark_data = benchmark
        benchmark_hash = ""
    values = benchmark_data["values"]
    scenarios.validate()
    accepted_inputs: list[np.ndarray] = []
    accepted_dispatch: list[np.ndarray] = []
    accepted_objective: list[float] = []
    accepted_cost: list[float] = []
    accepted_carbon: list[float] = []
    accepted_status: list[str] = []
    accepted_message: list[str] = []
    accepted_balance_residual: list[float] = []
    accepted_simultaneous: list[float] = []
    accepted_demand: list[np.ndarray] = []
    accepted_pv: list[np.ndarray] = []
    accepted_wt: list[np.ndarray] = []
    accepted_prices: list[np.ndarray] = []
    accepted_soc: list[float] = []
    accepted_ids: list[int] = []
    rejected = 0
    for index in range(scenarios.n_samples):
        prices = np.asarray(scenarios.prices[index], dtype=np.float64)
        params = _params_for_scenario(values, prices)
        inputs = DispatchInputs(
            demand=np.asarray(scenarios.demand[index], dtype=np.float64),
            pv_available=np.asarray(scenarios.pv_available[index], dtype=np.float64),
            wt_available=np.asarray(scenarios.wt_available[index], dtype=np.float64),
            parameters=params,
            initial_soc=float(scenarios.initial_soc[index]),
        )
        try:
            result = solve_dispatch_lp(inputs)
            if not result.success or not np.isfinite(result.objective):
                rejected += 1
                continue
            dispatch = _dispatch_array(result.values)
            if not np.isfinite(dispatch).all():
                rejected += 1
                continue
        except (ValueError, KeyError, FloatingPointError, RuntimeError):
            rejected += 1
            continue
        # Gas prior is deliberately not read from any scenario input and is
        # created only after the exact teacher solution exists.
        gas_placeholder = np.zeros((HORIZON, 1), dtype=np.float64)
        soc_column = np.full((HORIZON, 1), float(scenarios.initial_soc[index]), dtype=np.float64)
        feature = np.concatenate([
            scenarios.demand[index],
            gas_placeholder,
            scenarios.pv_available[index, :, None],
            scenarios.wt_available[index, :, None],
            prices,
            soc_column,
        ], axis=-1)
        accepted_inputs.append(feature)
        accepted_dispatch.append(dispatch)
        accepted_objective.append(float(result.objective))
        accepted_cost.append(_physical_cost(dispatch, prices, params))
        accepted_carbon.append(_carbon(dispatch, params))
        accepted_status.append(str(result.status))
        accepted_message.append(str(result.message))
        accepted_balance_residual.append(float(max(result.balance_residuals.values())) if result.balance_residuals else 0.0)
        accepted_simultaneous.append(float(result.simultaneous_charge_discharge))
        accepted_demand.append(np.asarray(scenarios.demand[index]))
        accepted_pv.append(np.asarray(scenarios.pv_available[index]))
        accepted_wt.append(np.asarray(scenarios.wt_available[index]))
        accepted_prices.append(prices)
        accepted_soc.append(float(scenarios.initial_soc[index]))
        accepted_ids.append(int(scenarios.scenario_ids[index]))

    if not accepted_inputs:
        raise RuntimeError(f"all {scenarios.n_samples} synthetic scenarios were rejected by HiGHS")
    dispatch_array = np.stack(accepted_dispatch, axis=0)
    gas_seed = int(gas_prior_seed if gas_prior_seed is not None else scenarios.seed + 10_000_019)
    prior, prior_mask = _make_gas_prior(dispatch_array, contract, gas_seed)
    inputs_array = np.stack(accepted_inputs, axis=0)
    inputs_array[:, :, 3] = prior
    output = LabeledProxySplit(
        inputs=inputs_array,
        dispatch=dispatch_array,
        teacher_objective=np.asarray(accepted_objective, dtype=np.float64),
        teacher_cost=np.asarray(accepted_cost, dtype=np.float64),
        teacher_carbon=np.asarray(accepted_carbon, dtype=np.float64),
        teacher_status=np.asarray(accepted_status, dtype=str),
        teacher_message=np.asarray(accepted_message, dtype=str),
        teacher_balance_residual=np.asarray(accepted_balance_residual, dtype=np.float64),
        teacher_simultaneous_charge_discharge=np.asarray(accepted_simultaneous, dtype=np.float64),
        demand=np.stack(accepted_demand, axis=0),
        pv_available=np.stack(accepted_pv, axis=0),
        wt_available=np.stack(accepted_wt, axis=0),
        prices=np.stack(accepted_prices, axis=0),
        initial_soc=np.asarray(accepted_soc, dtype=np.float64),
        gas_prior_mask=prior_mask,
        scenario_ids=np.asarray(accepted_ids, dtype=np.int64),
        split=scenarios.split,
        seed=scenarios.seed,
        rejected_count=rejected,
        attempted_count=scenarios.n_samples,
        benchmark_sha256=benchmark_hash,
        contract_sha256=contract_sha256(contract) if contract is not None else "",
        generator_version=scenarios.generator_version,
    )
    output.validate()
    return output


def save_proxy_split(split: LabeledProxySplit, path: str | Path) -> None:
    split.validate(require_provenance=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=split.inputs,
        features=split.inputs,
        gas_prior=split.inputs[:, :, 3],
        dispatch=split.dispatch,
        labels=split.dispatch,
        teacher_objective=split.teacher_objective,
        teacher_cost=split.teacher_cost,
        teacher_carbon=split.teacher_carbon,
        teacher_status=split.teacher_status.astype(str),
        teacher_message=split.teacher_message.astype(str),
        teacher_balance_residual=split.teacher_balance_residual,
        teacher_simultaneous_charge_discharge=split.teacher_simultaneous_charge_discharge,
        demand=split.demand,
        pv_available=split.pv_available,
        wt_available=split.wt_available,
        prices=split.prices,
        initial_soc=split.initial_soc,
        gas_prior_mask=split.gas_prior_mask,
        scenario_ids=split.scenario_ids,
        rejected_count=np.asarray(split.rejected_count, dtype=np.int64),
        attempted_count=np.asarray(split.attempted_count, dtype=np.int64),
        seed=np.asarray(split.seed, dtype=np.int64),
        split=np.asarray(split.split),
        sample_count=np.asarray(split.n_samples, dtype=np.int64),
        feature_order=np.asarray(FEATURE_ORDER),
        label_order=np.asarray(LABEL_ORDER),
        benchmark_sha256=np.asarray(split.benchmark_sha256),
        contract_sha256=np.asarray(split.contract_sha256),
        source_type=np.asarray(split.source_type),
        generator_version=np.asarray(split.generator_version),
        scenario_id_digest=np.asarray(scenario_id_digest(split.scenario_ids)),
    )


def _scalar_text(payload: Any, name: str) -> str:
    if name not in payload:
        raise ValueError(f"split artifact is missing required provenance field: {name}")
    value = np.asarray(payload[name])
    if value.ndim != 0:
        raise ValueError(f"split artifact field {name} must be scalar")
    return str(value.item())


def _scalar_int(payload: Any, name: str) -> int:
    if name not in payload:
        raise ValueError(f"split artifact is missing required field: {name}")
    value = np.asarray(payload[name])
    if value.ndim != 0:
        raise ValueError(f"split artifact field {name} must be scalar")
    try:
        result = int(value.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"split artifact field {name} must be an integer") from exc
    return result


def _contract_split_for_artifact(
    contract: ProxyContract,
    split: str,
    seed: int,
    count: int,
    smoke: bool | None,
) -> tuple[int, int]:
    candidates: list[tuple[int, int, bool]] = []
    for is_smoke in ((bool(smoke),) if smoke is not None else (False, True)):
        spec = contract.split(split, smoke=is_smoke)
        candidates.append((int(spec.seed), int(spec.size), is_smoke))
    matches = [(candidate_seed, candidate_size) for candidate_seed, candidate_size, _ in candidates if (candidate_seed, candidate_size) == (seed, count)]
    if len(matches) != 1:
        if smoke is None and len(matches) > 1:
            raise ValueError("artifact seed/count is ambiguous between formal and smoke contract splits; pass expected_smoke")
        raise ValueError(f"artifact {split} seed/count does not match the expected contract split")
    return matches[0]


def load_proxy_split(
    path: str | Path,
    split: str | None = None,
    *,
    expected_split: str | None = None,
    expected_seed: int | None = None,
    expected_count: int | None = None,
    expected_contract: ProxyContract | None = None,
    expected_benchmark_sha256: str | None = None,
    expected_smoke: bool | None = None,
    expected_source_type: str = "pure_simulation",
    expected_generator_version: str = GENERATOR_VERSION,
) -> LabeledProxySplit:
    """Load a split with fail-closed, explicit provenance validation.

    Older artifacts that relied on inferred fields are intentionally rejected:
    every teacher audit and provenance field is required in the NPZ payload.
    """

    path = Path(path)
    requested_split = expected_split if expected_split is not None else split
    if requested_split not in {"train", "validation", "test"}:
        raise ValueError("load_proxy_split requires an explicit expected split")
    required = {
        "inputs", "dispatch", "teacher_objective", "teacher_cost", "teacher_carbon",
        "teacher_status", "teacher_message", "teacher_balance_residual",
        "teacher_simultaneous_charge_discharge", "demand", "pv_available", "wt_available",
        "prices", "initial_soc", "gas_prior_mask", "scenario_ids", "rejected_count",
        "attempted_count", "seed", "split", "sample_count", "feature_order", "label_order",
        "benchmark_sha256", "contract_sha256", "source_type", "generator_version",
        "scenario_id_digest",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"split artifact is missing required fields: {missing}")
        stored_split = _scalar_text(payload, "split")
        seed = _scalar_int(payload, "seed")
        count = _scalar_int(payload, "sample_count")
        benchmark_hash = _scalar_text(payload, "benchmark_sha256").lower()
        contract_hash = _scalar_text(payload, "contract_sha256").lower()
        source_type = _scalar_text(payload, "source_type")
        generator_version = _scalar_text(payload, "generator_version")
        stored_digest = _scalar_text(payload, "scenario_id_digest").lower()
        feature_order = tuple(str(value) for value in np.asarray(payload["feature_order"]).tolist())
        label_order = tuple(str(value) for value in np.asarray(payload["label_order"]).tolist())
        if stored_split != requested_split:
            raise ValueError(f"stored split identity {stored_split!r} does not match expected {requested_split!r}")
        if expected_split is not None and split is not None and split != expected_split:
            raise ValueError("split and expected_split disagree")
        if feature_order != FEATURE_ORDER or label_order != LABEL_ORDER:
            raise ValueError("split feature/label order does not match the frozen contract")
        if source_type != expected_source_type:
            raise ValueError("split source_type mismatch")
        if generator_version != expected_generator_version:
            raise ValueError("split generator_version mismatch")
        if not _is_sha256(benchmark_hash) or not _is_sha256(contract_hash):
            raise ValueError("split benchmark_sha256 and contract_sha256 must be SHA-256 digests")
        if expected_benchmark_sha256 is not None and benchmark_hash != str(expected_benchmark_sha256).lower():
            raise ValueError("split benchmark SHA-256 mismatch")
        inputs = np.asarray(payload["inputs"], dtype=np.float64)
        dispatch = np.asarray(payload["dispatch"], dtype=np.float64)
        n = int(inputs.shape[0]) if inputs.ndim else -1
        if count != n:
            raise ValueError("split sample_count does not match inputs")
        if expected_seed is not None and seed != int(expected_seed):
            raise ValueError("split seed mismatch")
        if expected_count is not None and count != int(expected_count):
            raise ValueError("split sample_count mismatch")
        if expected_contract is not None:
            expected_hash = contract_sha256(expected_contract)
            if contract_hash != expected_hash.lower():
                raise ValueError("split contract SHA-256 mismatch")
            _contract_split_for_artifact(expected_contract, requested_split, seed, count, expected_smoke)
            if expected_contract.feature_order != feature_order or expected_contract.label_order != label_order:
                raise ValueError("split order does not match expected contract")
            if expected_contract.source_type != source_type or expected_contract.generator_version != generator_version:
                raise ValueError("split source/generator does not match expected contract")
        raw_ids = np.asarray(payload["scenario_ids"])
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("scenario_ids must be stored as integer values")
        ids = np.asarray(raw_ids, dtype=np.int64)
        if scenario_id_digest(ids).lower() != stored_digest:
            raise ValueError("scenario ID digest mismatch")
        output = LabeledProxySplit(
            inputs=inputs,
            dispatch=dispatch,
            teacher_objective=np.asarray(payload["teacher_objective"], dtype=np.float64),
            teacher_cost=np.asarray(payload["teacher_cost"], dtype=np.float64),
            teacher_carbon=np.asarray(payload["teacher_carbon"], dtype=np.float64),
            teacher_status=np.asarray(payload["teacher_status"], dtype=str),
            teacher_message=np.asarray(payload["teacher_message"], dtype=str),
            teacher_balance_residual=np.asarray(payload["teacher_balance_residual"], dtype=np.float64),
            teacher_simultaneous_charge_discharge=np.asarray(payload["teacher_simultaneous_charge_discharge"], dtype=np.float64),
            demand=np.asarray(payload["demand"], dtype=np.float64),
            pv_available=np.asarray(payload["pv_available"], dtype=np.float64),
            wt_available=np.asarray(payload["wt_available"], dtype=np.float64),
            prices=np.asarray(payload["prices"], dtype=np.float64),
            initial_soc=np.asarray(payload["initial_soc"], dtype=np.float64),
            gas_prior_mask=np.asarray(payload["gas_prior_mask"], dtype=np.float64),
            scenario_ids=ids,
            split=stored_split,
            seed=seed,
            rejected_count=_scalar_int(payload, "rejected_count"),
            attempted_count=_scalar_int(payload, "attempted_count"),
            benchmark_sha256=benchmark_hash,
            contract_sha256=contract_hash,
            source_type=source_type,
            generator_version=generator_version,
        )
    output.validate(require_provenance=True)
    return output


@dataclass(frozen=True)
class ProxyNormalizationStats:
    """Train-only input standardization and positive dispatch scales."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    label_scale: np.ndarray
    feature_order: tuple[str, ...] = FEATURE_ORDER
    label_order: tuple[str, ...] = LABEL_ORDER
    epsilon: float = 1.0e-6
    benchmark_sha256: str = ""
    contract_sha256: str = ""
    fit_split: str = "train"
    train_seed: int = 0
    source_type: str = "pure_simulation"
    generator_version: str = GENERATOR_VERSION
    # The fields below are populated when stats are attached to a complete
    # train/validation artifact bundle.  Stand-alone train-only fitting does
    # not know the validation split or optimizer seed and therefore leaves
    # those fields unpopulated; load_trained_proxy requires the complete
    # bundle before accepting a checkpoint.
    selection_split: str = ""
    test_split_used_for_selection: Optional[bool] = None
    seed: int = 0
    train_split: str = "train"
    validation_split: str = ""
    validation_seed: int = 0
    train_scenario_id_digest: str = ""
    validation_scenario_id_digest: str = ""

    @property
    def feature_mean(self) -> np.ndarray:
        return self.input_mean

    @property
    def feature_scale(self) -> np.ndarray:
        return self.input_scale

    @property
    def dispatch_scale(self) -> np.ndarray:
        return self.label_scale

    @property
    def output_scale(self) -> np.ndarray:
        return self.label_scale

    @classmethod
    def fit(
        cls,
        train_inputs: np.ndarray | LabeledProxySplit,
        train_dispatch: np.ndarray | None = None,
        benchmark: Mapping[str, Any] | str | Path | None = None,
        epsilon: float = 1.0e-6,
        scale_floor: float = 1.0e-3,
        benchmark_sha256: str | None = None,
        contract_sha256: str | None = None,
        fit_split: str | None = None,
        train_seed: int | None = None,
        source_type: str | None = None,
        generator_version: str | None = None,
    ) -> "ProxyNormalizationStats":
        source_split: LabeledProxySplit | None = None
        if isinstance(train_inputs, LabeledProxySplit):
            split = train_inputs
            if split.split != "train":
                raise ValueError("normalization statistics may only be fit from the train split")
            split.validate(require_provenance=True)
            source_split = split
            # Permit the convenient ``fit(split, benchmark_path)`` spelling
            # while retaining the explicit ``benchmark=`` form.
            if isinstance(train_dispatch, (str, Path)) and benchmark is None:
                benchmark = train_dispatch
            elif train_dispatch is not None and not np.array_equal(np.asarray(train_dispatch), split.dispatch):
                raise ValueError("train_dispatch does not match the labelled train split")
            train_dispatch = split.dispatch
            train_inputs = split.inputs
        else:
            if fit_split != "train" or train_seed is None or source_type is None or generator_version is None:
                raise ValueError("array normalization fitting requires explicit train provenance")
        x = np.asarray(train_inputs, dtype=np.float64)
        if train_dispatch is None:
            raise ValueError("train_dispatch is required")
        y = np.asarray(train_dispatch, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != (HORIZON, len(FEATURE_ORDER)):
            raise ValueError("train_inputs must have shape [N,4,10]")
        if y.ndim != 3 or y.shape[1:] != (HORIZON, len(LABEL_ORDER)):
            raise ValueError("train_dispatch must have shape [N,4,21]")
        mean = np.mean(x, axis=(0, 1))
        scale = np.std(x, axis=(0, 1))
        scale = np.maximum(scale, float(scale_floor))
        if benchmark is None:
            raise ValueError("normalization fitting requires benchmark values for exact LP label bounds")
        benchmark_data = load_benchmark(benchmark) if isinstance(benchmark, (str, Path)) else benchmark
        values = benchmark_data["values"] if isinstance(benchmark_data, Mapping) and "values" in benchmark_data else benchmark_data
        # These entries mirror the finite upper bounds in dispatch_lp.py in
        # exact LABEL_ORDER.  They are physical bounds, not train statistics.
        capacity_scales = np.asarray([
            values.get("grid_import_capacity", 0.0),
            values.get("pv_capacity", 0.0), values.get("pv_capacity", 0.0),
            values.get("wt_capacity", 0.0), values.get("wt_capacity", 0.0),
            values.get("chp_electric_capacity", 0.0) / max(values.get("chp_electric_efficiency", 1.0), epsilon),
            values.get("gas_boiler_capacity", 0.0) / max(values.get("gas_boiler_efficiency", 1.0), epsilon),
            values.get("chp_electric_capacity", 0.0), values.get("chp_heat_capacity", 0.0), values.get("gas_boiler_capacity", 0.0),
            values.get("electric_chiller_capacity", 0.0) / max(values.get("electric_chiller_cop", 1.0), epsilon), values.get("electric_chiller_capacity", 0.0),
            values.get("absorption_chiller_capacity", 0.0) / max(values.get("absorption_chiller_cop", 1.0), epsilon), values.get("absorption_chiller_capacity", 0.0),
            values.get("bess_power_capacity", 0.0), values.get("bess_power_capacity", 0.0), values.get("bess_energy_capacity", 0.0),
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float64)
        bounded = np.arange(17, dtype=np.int64)
        if not np.isfinite(capacity_scales[bounded]).all() or (capacity_scales[bounded] <= 0.0).any():
            raise ValueError("frozen LP finite label bounds must be positive and finite")
        if not np.isfinite(y).all():
            raise ValueError("train dispatch labels must be finite")
        bound_tolerance = np.maximum(1.0e-8, 1.0e-10 * capacity_scales[bounded])
        excess = np.max(y[..., bounded] - capacity_scales[bounded], axis=(0, 1))
        if np.any(excess > bound_tolerance):
            bad = bounded[np.flatnonzero(excess > bound_tolerance)]
            raise ValueError(f"exact teacher labels exceed frozen LP bounds at label indices {bad.tolist()}")
        empirical = np.max(y, axis=(0, 1))
        robust = np.maximum(empirical * 1.05, np.percentile(y, 99.5, axis=(0, 1)) * 1.1)
        # Only truly unbounded slack/dump variables use robust train-only
        # scales.  Finite-bound variables stay exactly at their LP bounds.
        label_scale = np.empty(len(LABEL_ORDER), dtype=np.float64)
        label_scale[bounded] = capacity_scales[bounded]
        unbounded = np.arange(17, len(LABEL_ORDER), dtype=np.int64)
        label_scale[unbounded] = np.maximum(robust[unbounded], float(scale_floor))
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(label_scale).all():
            raise ValueError("normalization statistics must be finite")
        split_benchmark_hash = source_split.benchmark_sha256 if source_split is not None else ""
        split_contract_hash = source_split.contract_sha256 if source_split is not None else ""
        if isinstance(benchmark, (str, Path)):
            benchmark_file_hash = file_sha256(benchmark)
            if split_benchmark_hash and split_benchmark_hash.lower() != benchmark_file_hash.lower():
                raise ValueError("normalization benchmark SHA-256 does not match train split")
            split_benchmark_hash = benchmark_file_hash
        selected_benchmark_hash = str(benchmark_sha256 if benchmark_sha256 is not None else split_benchmark_hash).lower()
        selected_contract_hash = str(contract_sha256 if contract_sha256 is not None else split_contract_hash).lower()
        if not _is_sha256(selected_benchmark_hash) or not _is_sha256(selected_contract_hash):
            raise ValueError("normalization fitting requires benchmark and contract SHA-256 provenance")
        if source_split is not None:
            if selected_benchmark_hash != source_split.benchmark_sha256.lower() or selected_contract_hash != source_split.contract_sha256.lower():
                raise ValueError("normalization provenance does not match the train split")
            selected_fit_split = source_split.split
            selected_seed = int(source_split.seed)
            selected_source_type = source_split.source_type
            selected_generator = source_split.generator_version
        else:
            selected_fit_split = str(fit_split)
            selected_seed = int(train_seed)
            selected_source_type = str(source_type)
            selected_generator = str(generator_version)
        if selected_fit_split != "train" or selected_seed <= 0 or selected_source_type != "pure_simulation" or not selected_generator:
            raise ValueError("invalid train normalization provenance")
        return cls(
            mean.astype(np.float64), scale.astype(np.float64), label_scale.astype(np.float64),
            epsilon=float(epsilon),
            benchmark_sha256=selected_benchmark_hash,
            contract_sha256=selected_contract_hash,
            fit_split=selected_fit_split,
            train_seed=selected_seed,
            source_type=selected_source_type,
            generator_version=selected_generator,
            train_split=selected_fit_split,
            train_scenario_id_digest=(scenario_id_digest(source_split.scenario_ids) if source_split is not None else ""),
        )

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float64)
        if x.shape[-1] != len(FEATURE_ORDER):
            raise ValueError("inputs last dimension must be 10")
        result = (x - self.input_mean) / self.input_scale
        if not np.isfinite(result).all():
            raise ValueError("normalized inputs are non-finite")
        return result

    def inverse_inputs(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=np.float64) * self.input_scale + self.input_mean

    def normalize_dispatch(self, dispatch: np.ndarray) -> np.ndarray:
        y = np.asarray(dispatch, dtype=np.float64) / self.label_scale
        if not np.isfinite(y).all():
            raise ValueError("normalized dispatch is non-finite")
        return np.clip(y, 0.0, 1.0)

    def inverse_dispatch(self, normalized: np.ndarray) -> np.ndarray:
        y = np.asarray(normalized, dtype=np.float64)
        if y.shape[-1] != len(LABEL_ORDER):
            raise ValueError("normalized dispatch last dimension must be 21")
        return y * self.label_scale

    # Familiar aliases for callers using transform/inverse terminology.
    transform_dispatch = normalize_dispatch

    def save(self, path: str | Path) -> None:
        if tuple(self.feature_order) != FEATURE_ORDER or tuple(self.label_order) != LABEL_ORDER:
            raise ValueError("normalization metadata feature/label order mismatch")
        if not _is_sha256(self.benchmark_sha256) or not _is_sha256(self.contract_sha256) or self.fit_split != "train" or self.train_seed <= 0 or self.source_type != "pure_simulation" or not self.generator_version:
            raise ValueError("normalization provenance is incomplete")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            label_scale=self.label_scale,
            feature_order=np.asarray(self.feature_order),
            label_order=np.asarray(self.label_order),
            epsilon=np.asarray(self.epsilon),
            benchmark_sha256=np.asarray(self.benchmark_sha256),
            contract_sha256=np.asarray(self.contract_sha256),
            fit_split=np.asarray(self.fit_split),
            train_seed=np.asarray(self.train_seed, dtype=np.int64),
            source_type=np.asarray(self.source_type),
            generator_version=np.asarray(self.generator_version),
            selection_split=np.asarray(self.selection_split),
            test_split_used_for_selection=np.asarray(
                -1 if self.test_split_used_for_selection is None else int(bool(self.test_split_used_for_selection)),
                dtype=np.int8,
            ),
            seed=np.asarray(self.seed, dtype=np.int64),
            train_split=np.asarray(self.train_split),
            validation_split=np.asarray(self.validation_split),
            validation_seed=np.asarray(self.validation_seed, dtype=np.int64),
            train_scenario_id_digest=np.asarray(self.train_scenario_id_digest),
            validation_scenario_id_digest=np.asarray(self.validation_scenario_id_digest),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_benchmark_sha256: str | None = None,
        expected_contract_sha256: str | None = None,
        benchmark_sha256: str | None = None,
        contract_sha256: str | None = None,
        expected_fit_split: str | None = None,
        expected_train_seed: int | None = None,
        expected_source_type: str = "pure_simulation",
        expected_generator_version: str = GENERATOR_VERSION,
    ) -> "ProxyNormalizationStats":
        with np.load(path, allow_pickle=False) as payload:
            required = {"input_mean", "input_scale", "label_scale", "feature_order", "label_order", "epsilon", "benchmark_sha256", "contract_sha256", "fit_split", "train_seed", "source_type", "generator_version"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"normalization artifact is missing required fields: {missing}")
            feature_order = tuple(str(v) for v in payload["feature_order"].tolist())
            label_order = tuple(str(v) for v in payload["label_order"].tolist())
            if feature_order != FEATURE_ORDER or label_order != LABEL_ORDER:
                raise ValueError("normalization metadata feature/label order mismatch")
            epsilon = float(np.asarray(payload["epsilon"]).item())
            stored_benchmark_hash = str(np.asarray(payload["benchmark_sha256"]).item()).lower()
            stored_contract_hash = str(np.asarray(payload["contract_sha256"]).item()).lower()
            fit_split = str(np.asarray(payload["fit_split"]).item())
            train_seed = int(np.asarray(payload["train_seed"]).item())
            source_type = str(np.asarray(payload["source_type"]).item())
            generator_version = str(np.asarray(payload["generator_version"]).item())
            selection_split = str(np.asarray(payload["selection_split"]).item()) if "selection_split" in payload else ""
            if "test_split_used_for_selection" in payload:
                raw_test_selection = int(np.asarray(payload["test_split_used_for_selection"]).item())
                if raw_test_selection not in {-1, 0, 1}:
                    raise ValueError("normalization test-selection flag must be -1, 0 or 1")
                test_split_used_for_selection = None if raw_test_selection == -1 else bool(raw_test_selection)
            else:
                test_split_used_for_selection = None
            seed = int(np.asarray(payload["seed"]).item()) if "seed" in payload else 0
            train_split = str(np.asarray(payload["train_split"]).item()) if "train_split" in payload else fit_split
            validation_split = str(np.asarray(payload["validation_split"]).item()) if "validation_split" in payload else ""
            validation_seed = int(np.asarray(payload["validation_seed"]).item()) if "validation_seed" in payload else 0
            train_id_digest = str(np.asarray(payload["train_scenario_id_digest"]).item()) if "train_scenario_id_digest" in payload else ""
            validation_id_digest = str(np.asarray(payload["validation_scenario_id_digest"]).item()) if "validation_scenario_id_digest" in payload else ""
            output = cls(
                np.asarray(payload["input_mean"], dtype=np.float64),
                np.asarray(payload["input_scale"], dtype=np.float64),
                np.asarray(payload["label_scale"], dtype=np.float64),
                feature_order=feature_order, label_order=label_order, epsilon=epsilon,
                benchmark_sha256=stored_benchmark_hash, contract_sha256=stored_contract_hash,
                fit_split=fit_split, train_seed=train_seed, source_type=source_type,
                generator_version=generator_version,
                selection_split=selection_split,
                test_split_used_for_selection=test_split_used_for_selection,
                seed=seed,
                train_split=train_split,
                validation_split=validation_split,
                validation_seed=validation_seed,
                train_scenario_id_digest=train_id_digest,
                validation_scenario_id_digest=validation_id_digest,
            )
        if output.input_mean.shape != (len(FEATURE_ORDER),) or output.input_scale.shape != (len(FEATURE_ORDER),) or output.label_scale.shape != (len(LABEL_ORDER),):
            raise ValueError("invalid normalization statistics shapes")
        if not np.isfinite(output.input_mean).all() or not np.isfinite(output.input_scale).all() or not np.isfinite(output.label_scale).all() or not np.isfinite(output.epsilon) or output.epsilon <= 0.0:
            raise ValueError("normalization statistics must be finite")
        if (output.input_scale <= 0).any() or (output.label_scale <= 0).any():
            raise ValueError("normalization scales must be positive")
        if not _is_sha256(output.benchmark_sha256) or not _is_sha256(output.contract_sha256):
            raise ValueError("normalization artifact provenance hashes are invalid")
        if output.fit_split != "train" or output.train_seed <= 0 or output.source_type != expected_source_type or output.generator_version != expected_generator_version:
            raise ValueError("normalization train provenance is invalid")
        if expected_fit_split is not None and output.fit_split != expected_fit_split:
            raise ValueError("normalization fit split mismatch")
        if expected_train_seed is not None and output.train_seed != int(expected_train_seed):
            raise ValueError("normalization train seed mismatch")
        expected_benchmark = expected_benchmark_sha256 if expected_benchmark_sha256 is not None else benchmark_sha256
        expected_contract = expected_contract_sha256 if expected_contract_sha256 is not None else contract_sha256
        if expected_benchmark is not None and output.benchmark_sha256.lower() != str(expected_benchmark).lower():
            raise ValueError("normalization benchmark SHA-256 mismatch")
        if expected_contract is not None and output.contract_sha256.lower() != str(expected_contract).lower():
            raise ValueError("normalization contract SHA-256 mismatch")
        return output


class ProxyDataset(Dataset):
    """PyTorch dataset exposing normalized and physical target views."""

    def __init__(self, split: LabeledProxySplit, stats: ProxyNormalizationStats):
        split.validate()
        self.split = split
        self.stats = stats
        self.normalized_inputs = stats.transform_inputs(split.inputs).astype(np.float32)
        self.normalized_dispatch = stats.normalize_dispatch(split.dispatch).astype(np.float32)

    def __len__(self) -> int:
        return self.split.n_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self.normalized_inputs[index]),
            "raw_inputs": torch.from_numpy(self.split.inputs[index].astype(np.float32)),
            "target": torch.from_numpy(self.normalized_dispatch[index]),
            "raw_target": torch.from_numpy(self.split.dispatch[index].astype(np.float32)),
            "teacher_cost": torch.tensor(self.split.teacher_cost[index], dtype=torch.float32),
            "teacher_carbon": torch.tensor(self.split.teacher_carbon[index], dtype=torch.float32),
            "gas_prior_mask": torch.from_numpy(self.split.gas_prior_mask[index].astype(np.float32)),
            "initial_soc": torch.tensor(self.split.initial_soc[index], dtype=torch.float32),
        }


def write_dataset_manifest(
    splits: Mapping[str, LabeledProxySplit],
    path: str | Path,
    contract: ProxyContract | None = None,
    benchmark_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if contract is None or benchmark_path is None:
        raise ValueError("dataset manifests require both contract and benchmark provenance")
    expected_names = ("train", "validation", "test")
    if set(splits) != set(expected_names):
        raise ValueError("dataset manifest must contain train, validation and test splits")
    benchmark_hash = file_sha256(benchmark_path)
    contract_hash = contract_sha256(contract)
    split_entries: dict[str, Any] = {}
    path = Path(path)
    for name in expected_names:
        split = splits[name]
        spec = contract.split(name, smoke=smoke)
        split.validate(require_provenance=True)
        if split.split != name or split.seed != int(spec.seed) or split.n_samples != int(spec.size):
            raise ValueError(f"{name} split does not match the contract seed/count")
        if split.benchmark_sha256.lower() != benchmark_hash.lower() or split.contract_sha256.lower() != contract_hash.lower():
            raise ValueError(f"{name} split provenance does not match the manifest contract/benchmark")
        if split.source_type != contract.source_type or split.generator_version != contract.generator_version:
            raise ValueError(f"{name} split source/generator does not match the contract")
        artifact = path.parent / f"{name}.npz"
        if not artifact.exists():
            raise ValueError(f"split artifact is missing before manifest creation: {artifact}")
        entry = split.manifest_entry()
        entry.update({"artifact": artifact.name, "artifact_sha256": file_sha256(artifact)})
        split_entries[name] = entry
    ids = [np.asarray(splits[name].scenario_ids, dtype=np.int64) for name in expected_names]
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            if np.intersect1d(left, right).size:
                raise ValueError("scenario IDs must be disjoint across dataset splits")
    manifest: dict[str, Any] = {
        "schema_version": "scheduling-proxy-dataset-v1",
        "source_type": contract.source_type,
        "generator_version": contract.generator_version,
        "feature_order": list(FEATURE_ORDER),
        "label_order": list(LABEL_ORDER),
        "normalization_fit_split": "train",
        "benchmark_sha256": benchmark_hash,
        "contract_sha256": contract_hash,
        "smoke": bool(smoke),
        "split_seeds": {name: int(contract.split(name, smoke=smoke).seed) for name in expected_names},
        "split_sizes": {name: int(contract.split(name, smoke=smoke).size) for name in expected_names},
        "splits": split_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def validate_dataset_manifest(
    path: str | Path,
    contract: ProxyContract,
    benchmark_path: str | Path,
    smoke: bool | None = None,
) -> dict[str, Any]:
    """Validate manifest, artifact hashes, split contracts and ID disjointness."""

    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest: {manifest_path}") from exc
    required = {"schema_version", "source_type", "generator_version", "feature_order", "label_order", "benchmark_sha256", "contract_sha256", "normalization_fit_split", "smoke", "split_seeds", "split_sizes", "splits"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"dataset manifest is missing required fields: {missing}")
    if manifest["schema_version"] != "scheduling-proxy-dataset-v1":
        raise ValueError("dataset manifest schema mismatch")
    if not isinstance(manifest["smoke"], bool) or not isinstance(manifest["source_type"], str) or not isinstance(manifest["generator_version"], str):
        raise ValueError("dataset manifest provenance types are invalid")
    if manifest["source_type"] != contract.source_type or manifest["generator_version"] != contract.generator_version:
        raise ValueError("dataset manifest source/generator mismatch")
    if tuple(manifest["feature_order"]) != FEATURE_ORDER or tuple(manifest["label_order"]) != LABEL_ORDER:
        raise ValueError("dataset manifest feature/label order mismatch")
    actual_benchmark = file_sha256(benchmark_path)
    actual_contract = contract_sha256(contract)
    if str(manifest["benchmark_sha256"]).lower() != actual_benchmark.lower() or str(manifest["contract_sha256"]).lower() != actual_contract.lower():
        raise ValueError("dataset manifest contract/benchmark hash mismatch")
    manifest_smoke = bool(manifest["smoke"])
    if smoke is not None and manifest_smoke != bool(smoke):
        raise ValueError("dataset manifest smoke/formal identity mismatch")
    expected_names = ("train", "validation", "test")
    if set(manifest["splits"]) != set(expected_names):
        raise ValueError("dataset manifest must contain exactly train, validation and test")
    expected_seeds = {name: int(contract.split(name, smoke=manifest_smoke).seed) for name in expected_names}
    expected_sizes = {name: int(contract.split(name, smoke=manifest_smoke).size) for name in expected_names}
    if {str(name): int(value) for name, value in manifest["split_seeds"].items()} != expected_seeds:
        raise ValueError("dataset manifest split seeds do not match the contract")
    if {str(name): int(value) for name, value in manifest["split_sizes"].items()} != expected_sizes:
        raise ValueError("dataset manifest split sizes do not match the contract")
    loaded: dict[str, LabeledProxySplit] = {}
    for name in expected_names:
        entry = manifest["splits"][name]
        required_entry = {"split", "seed", "sample_count", "benchmark_sha256", "contract_sha256", "source_type", "generator_version", "feature_order", "label_order", "scenario_id_digest", "scenario_id_min", "scenario_id_max", "artifact", "artifact_sha256"}
        missing_entry = sorted(required_entry - set(entry))
        if missing_entry:
            raise ValueError(f"manifest split {name} is missing fields: {missing_entry}")
        spec = contract.split(name, smoke=manifest_smoke)
        if entry["split"] != name or int(entry["seed"]) != int(spec.seed) or int(entry["sample_count"]) != int(spec.size):
            raise ValueError(f"manifest split {name} seed/count mismatch")
        if (
            str(entry["benchmark_sha256"]).lower() != actual_benchmark.lower()
            or str(entry["contract_sha256"]).lower() != actual_contract.lower()
            or entry["source_type"] != contract.source_type
            or entry["generator_version"] != contract.generator_version
            or tuple(entry["feature_order"]) != FEATURE_ORDER
            or tuple(entry["label_order"]) != LABEL_ORDER
        ):
            raise ValueError(f"manifest split {name} provenance mismatch")
        artifact = manifest_path.parent / str(entry["artifact"])
        if not artifact.exists() or file_sha256(artifact).lower() != str(entry["artifact_sha256"]).lower():
            raise ValueError(f"manifest artifact hash mismatch for {name}")
        loaded[name] = load_proxy_split(
            artifact,
            expected_split=name,
            expected_seed=int(spec.seed),
            expected_count=int(spec.size),
            expected_contract=contract,
            expected_benchmark_sha256=actual_benchmark,
            expected_smoke=manifest_smoke,
        )
        if entry["scenario_id_digest"] != scenario_id_digest(loaded[name].scenario_ids):
            raise ValueError(f"manifest scenario ID digest mismatch for {name}")
        if int(entry["scenario_id_min"]) != int(np.min(loaded[name].scenario_ids)) or int(entry["scenario_id_max"]) != int(np.max(loaded[name].scenario_ids)):
            raise ValueError(f"manifest scenario ID range mismatch for {name}")
    names = list(expected_names)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            if np.intersect1d(loaded[left_name].scenario_ids, loaded[right_name].scenario_ids).size:
                raise ValueError("dataset manifest scenario IDs overlap across splits")
    return manifest


# Compatibility aliases.
build_proxy_dataset = build_labeled_proxy_split
create_labeled_dataset = build_labeled_proxy_split
build_labeled_dataset = build_labeled_proxy_split
generate_labeled_dataset = build_labeled_proxy_split
save_dataset_split = save_proxy_split
load_dataset_split = load_proxy_split


__all__ = [
    "LabeledProxySplit",
    "build_labeled_proxy_split",
    "build_proxy_dataset",
    "create_labeled_dataset",
    "build_labeled_dataset",
    "generate_labeled_dataset",
    "save_proxy_split",
    "load_proxy_split",
    "save_dataset_split",
    "load_dataset_split",
    "ProxyNormalizationStats",
    "ProxyDataset",
    "scenario_id_digest",
    "write_dataset_manifest",
    "validate_dataset_manifest",
]
