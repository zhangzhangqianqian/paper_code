"""Deterministic, pure-simulation four-hour IES scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import yaml

from .proxy_contract import FEATURE_ORDER, HORIZON, ProxyContract


GENERATOR_VERSION = "synthetic-scheduling-v1"


def load_benchmark(path: str | Path) -> Mapping[str, Any]:
    """Load and validate the frozen standard-IES benchmark YAML."""

    benchmark_path = Path(path)
    with benchmark_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("benchmark YAML must contain an object")
    if int(data.get("horizon_hours", -1)) != HORIZON:
        raise ValueError("benchmark horizon_hours must be 4")
    if not isinstance(data.get("values"), Mapping) or not isinstance(data.get("training_statistics"), Mapping):
        raise ValueError("benchmark must contain values and training_statistics mappings")
    return data


@dataclass(frozen=True)
class SyntheticScenarioBatch:
    """A synthetic split before exact labels are attached.

    ``gas_prior`` and ``gas_prior_mask`` are zero-filled placeholders.  The
    dataset builder replaces them only after the exact LP label has been
    obtained, which makes the auxiliary nature of this channel auditable.
    """

    demand: np.ndarray  # [N,H,3]
    pv_available: np.ndarray  # [N,H]
    wt_available: np.ndarray  # [N,H]
    prices: np.ndarray  # [N,H,3], grid/gas/carbon
    initial_soc: np.ndarray  # [N]
    scenario_ids: np.ndarray  # [N], split-disjoint integer IDs
    split: str
    seed: int
    generator_version: str = GENERATOR_VERSION
    gas_prior: np.ndarray | None = None  # [N,H], generated after labels
    gas_prior_mask: np.ndarray | None = None  # [N,H]
    metadata: Mapping[str, Any] = MappingProxyType({})

    @property
    def n_samples(self) -> int:
        return int(self.demand.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.demand.shape[1])

    @property
    def features_without_gas(self) -> np.ndarray:
        zeros = np.zeros(self.demand.shape[:2] + (1,), dtype=np.float64)
        return np.concatenate(
            [self.demand, zeros, self.pv_available[..., None], self.wt_available[..., None], self.prices,
             np.broadcast_to(self.initial_soc[:, None, None], (self.n_samples, self.horizon, 1))],
            axis=-1,
        )

    @property
    def features(self) -> np.ndarray:
        """Return the ten-feature view, using zero gas until labels exist."""

        gas = np.zeros((self.n_samples, self.horizon, 1), dtype=np.float64)
        if self.gas_prior is not None:
            gas = np.asarray(self.gas_prior, dtype=np.float64).reshape(self.n_samples, self.horizon, 1)
        return np.concatenate(
            [self.demand, gas, self.pv_available[..., None], self.wt_available[..., None], self.prices,
             np.broadcast_to(self.initial_soc[:, None, None], (self.n_samples, self.horizon, 1))],
            axis=-1,
        )

    def with_gas_prior(self, gas_prior: np.ndarray, gas_prior_mask: np.ndarray) -> "SyntheticScenarioBatch":
        prior = np.asarray(gas_prior, dtype=np.float64)
        mask = np.asarray(gas_prior_mask, dtype=np.float64)
        expected = (self.n_samples, self.horizon)
        if prior.shape not in (expected, expected + (1,)):
            raise ValueError(f"gas_prior must have shape {expected} or {expected + (1,)}")
        if mask.shape not in (expected, expected + (1,)):
            raise ValueError(f"gas_prior_mask must have shape {expected} or {expected + (1,)}")
        prior = prior.reshape(expected)
        mask = mask.reshape(expected)
        if not np.isfinite(prior).all() or not np.isfinite(mask).all() or (prior < 0).any():
            raise ValueError("gas prior and mask must be finite and non-negative")
        return SyntheticScenarioBatch(
            demand=self.demand,
            pv_available=self.pv_available,
            wt_available=self.wt_available,
            prices=self.prices,
            initial_soc=self.initial_soc,
            scenario_ids=self.scenario_ids,
            split=self.split,
            seed=self.seed,
            generator_version=self.generator_version,
            gas_prior=prior,
            gas_prior_mask=mask,
            metadata=dict(self.metadata, gas_prior_generated_from_teacher=True),
        )

    def validate(self) -> None:
        n, h = self.demand.shape[:2] if self.demand.ndim >= 2 else (0, 0)
        if self.demand.ndim != 3 or self.demand.shape[2] != 3 or h != HORIZON:
            raise ValueError("demand must have shape [N,4,3]")
        if self.pv_available.shape != (n, h) or self.wt_available.shape != (n, h):
            raise ValueError("renewables must have shape [N,4]")
        if self.prices.shape != (n, h, 3):
            raise ValueError("prices must have shape [N,4,3]")
        if self.initial_soc.shape != (n,) or self.scenario_ids.shape != (n,):
            raise ValueError("initial_soc/scenario_ids must have shape [N]")
        arrays = (self.demand, self.pv_available, self.wt_available, self.prices, self.initial_soc)
        if any(not np.isfinite(np.asarray(array)).all() for array in arrays):
            raise ValueError("synthetic scenario arrays must be finite")
        if any((np.asarray(array) < 0).any() for array in arrays[:-1]):
            raise ValueError("synthetic demand, renewable and prices must be non-negative")
        if (self.initial_soc < 0.0).any() or (self.initial_soc > 1.0).any():
            raise ValueError("initial_soc must be in [0,1]")
        if np.unique(self.scenario_ids).size != n:
            raise ValueError("scenario_ids must be unique within a split")
        if self.gas_prior is not None:
            prior = np.asarray(self.gas_prior)
            mask = np.asarray(self.gas_prior_mask)
            if prior.shape != (n, h) or mask.shape != (n, h):
                raise ValueError("gas prior arrays must have shape [N,4]")
            if not np.isfinite(prior).all() or not np.isfinite(mask).all() or (prior < 0).any():
                raise ValueError("gas prior arrays must be finite and non-negative")
            if not np.isin(mask, [0.0, 1.0]).all():
                raise ValueError("gas_prior_mask must be binary")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "split": self.split,
            "seed": int(self.seed),
            "sample_count": self.n_samples,
            "horizon": self.horizon,
            "source_type": "pure_simulation",
            "uses_observed_windows": False,
            "uses_observed_timestamps": False,
            "scenario_id_min": int(self.scenario_ids.min()) if self.n_samples else None,
            "scenario_id_max": int(self.scenario_ids.max()) if self.n_samples else None,
            **dict(self.metadata),
        }


def _benchmark_values(benchmark: Mapping[str, Any]) -> tuple[Mapping[str, float], Mapping[str, Mapping[str, float]]]:
    values = {str(k): float(v) for k, v in benchmark["values"].items()}
    stats = {
        str(name): {str(key): float(value) for key, value in mapping.items()}
        for name, mapping in benchmark["training_statistics"].items()
    }
    for key in ("electricity", "cooling", "heating"):
        if key not in stats or "mean" not in stats[key] or "p95" not in stats[key]:
            raise ValueError(f"benchmark training_statistics missing {key}.mean/p95")
    return values, stats


def _smooth_process(rng: np.random.Generator, n: int, h: int, mean: float, spread: float) -> np.ndarray:
    innovations = rng.normal(0.0, spread, size=(n, h))
    values = np.empty((n, h), dtype=np.float64)
    values[:, 0] = mean + innovations[:, 0]
    for t in range(1, h):
        values[:, t] = 0.82 * values[:, t - 1] + 0.18 * mean + innovations[:, t]
    return values


def generate_synthetic_scenarios(
    benchmark: Mapping[str, Any] | str | Path,
    split: str,
    seed: int,
    n_samples: int,
    horizon: int = HORIZON,
    generator_version: str = GENERATOR_VERSION,
) -> SyntheticScenarioBatch:
    """Generate a deterministic synthetic split from benchmark statistics only."""

    if isinstance(benchmark, (str, Path)):
        benchmark = load_benchmark(benchmark)
    if horizon != HORIZON:
        raise ValueError("synthetic scheduling horizon is fixed at 4")
    if not split or split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if int(n_samples) <= 0 or int(seed) <= 0:
        raise ValueError("n_samples and seed must be positive")
    values, stats = _benchmark_values(benchmark)
    n_samples = int(n_samples)
    seed = int(seed)
    rng = np.random.default_rng(seed)

    e_mean, c_mean, h_mean = (stats[k]["mean"] for k in ("electricity", "cooling", "heating"))
    e_p95, c_p95, h_p95 = (stats[k]["p95"] for k in ("electricity", "cooling", "heating"))
    e_cap = float(values["grid_import_capacity"])
    c_cap = float(values["electric_chiller_capacity"]) + float(values["absorption_chiller_capacity"])
    h_cap = float(values["chp_heat_capacity"]) + float(values["gas_boiler_capacity"])

    regimes = rng.integers(0, 4, size=n_samples)
    activity = np.array([0.78, 0.98, 1.16, 0.88], dtype=np.float64)[regimes]
    thermal = np.array([0.65, 0.95, 1.22, 0.78], dtype=np.float64)[regimes]
    phase = rng.uniform(-np.pi, np.pi, size=n_samples)
    time = np.arange(horizon, dtype=np.float64)[None, :]
    daily = 1.0 + 0.08 * np.sin(time * (np.pi / 2.0) + phase[:, None])

    electricity = _smooth_process(rng, n_samples, horizon, e_mean, max(1.0, 0.055 * e_p95))
    electricity *= activity[:, None] * daily
    electricity = np.clip(electricity, 0.08 * e_mean, min(0.96 * e_cap, 1.12 * e_p95))

    cooling = _smooth_process(rng, n_samples, horizon, c_mean, max(1.0, 0.06 * c_p95))
    heating = _smooth_process(rng, n_samples, horizon, h_mean, max(1.0, 0.06 * h_p95))
    # Normal regimes are negatively coupled; shoulder regimes retain overlap.
    cooling *= (0.55 + 0.52 * thermal[:, None]) * (1.0 + 0.05 * np.sin(time + phase[:, None]))
    heating *= (1.35 - 0.43 * thermal[:, None]) * (1.0 + 0.04 * np.cos(time + phase[:, None]))
    cooling = np.clip(cooling, 0.02 * c_mean, min(0.82 * c_cap, 1.10 * c_p95))
    heating = np.clip(heating, 0.02 * h_mean, min(0.72 * h_cap, 1.08 * h_p95))
    demand = np.stack([electricity, cooling, heating], axis=-1)

    daylight = np.clip(np.sin((time - 0.5) * np.pi / 4.0), 0.0, 1.0)
    daylight = np.broadcast_to(daylight, (n_samples, horizon))
    cloud = np.clip(_smooth_process(rng, n_samples, horizon, 0.84, 0.08), 0.35, 1.0)
    pv_capacity = float(values.get("pv_capacity", 0.0))
    pv_available = np.clip(pv_capacity * daylight * cloud * rng.uniform(0.55, 1.0, size=(n_samples, 1)), 0.0, pv_capacity)
    wind_shape = np.clip(_smooth_process(rng, n_samples, horizon, 0.48, 0.16), 0.05, 0.95)
    wt_capacity = float(values.get("wt_capacity", 0.0))
    wt_available = np.clip(wt_capacity * wind_shape * rng.uniform(0.65, 1.0, size=(n_samples, 1)), 0.0, wt_capacity)

    grid_base = float(values.get("grid_energy_price", 1.0))
    gas_base = float(values.get("gas_energy_price", 0.6))
    carbon_base = float(values.get("carbon_price_default", 0.0))
    carbon_sensitivity = float(values.get("carbon_price_sensitivity", 0.2))
    grid_prices = grid_base * rng.uniform(0.88, 1.12, size=n_samples)
    gas_prices = gas_base * rng.uniform(0.90, 1.10, size=n_samples)
    carbon_prices = np.maximum(0.0, carbon_base + carbon_sensitivity * rng.uniform(0.05, 0.30, size=n_samples))
    prices = np.stack([
        np.broadcast_to(grid_prices[:, None], (n_samples, horizon)),
        np.broadcast_to(gas_prices[:, None], (n_samples, horizon)),
        np.broadcast_to(carbon_prices[:, None], (n_samples, horizon)),
    ], axis=-1)
    initial_soc = rng.uniform(0.2, 0.8, size=n_samples)
    scenario_ids = np.asarray(seed * 1_000_000 + np.arange(n_samples, dtype=np.int64), dtype=np.int64)

    batch = SyntheticScenarioBatch(
        demand=np.asarray(demand, dtype=np.float64),
        pv_available=np.asarray(pv_available, dtype=np.float64),
        wt_available=np.asarray(wt_available, dtype=np.float64),
        prices=np.asarray(prices, dtype=np.float64),
        initial_soc=np.asarray(initial_soc, dtype=np.float64),
        scenario_ids=scenario_ids,
        split=str(split),
        seed=seed,
        generator_version=str(generator_version),
        gas_prior=np.zeros((n_samples, horizon), dtype=np.float64),
        gas_prior_mask=np.zeros((n_samples, horizon), dtype=np.float64),
        metadata={
            "regime_count": 4,
            "latent_regime_histogram": np.bincount(regimes, minlength=4).tolist(),
            "uses_observed_windows": False,
            "uses_observed_timestamps": False,
            "source_type": "pure_simulation",
        },
    )
    batch.validate()
    return batch


# Short aliases used by scripts and downstream tests.
generate_scenarios = generate_synthetic_scenarios


__all__ = [
    "GENERATOR_VERSION",
    "SyntheticScenarioBatch",
    "load_benchmark",
    "generate_synthetic_scenarios",
    "generate_scenarios",
]
