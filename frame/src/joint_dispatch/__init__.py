"""End-to-end joint forecasting and physics-feasible dispatch package.

Public symbols are loaded lazily so a versioned submodule can be imported from
an otherwise clean source checkout without pulling in unrelated legacy
experiment modules.
"""

from importlib import import_module

from .contract import (
    CurriculumSpec,
    JointTrainingContract,
    JointVariant,
    load_joint_training_contract,
    validate_joint_contract,
)
from .data import (
    JointNormalization,
    JointWindowSplit,
    LPSolveBenchmark,
    benchmark_lp_generation,
    build_causal_device_trajectory,
    build_joint_windows,
    derive_device_status,
    fit_joint_normalization,
    load_joint_split,
    save_joint_split,
)

_LAZY_EXPORTS = {
    "AblationSpec": ("formal_protocol", "AblationSpec"),
    "FormalExperimentSpec": ("formal_protocol", "FormalExperimentSpec"),
    "MethodSpec": ("formal_protocol", "MethodSpec"),
    "TrainingBudget": ("formal_protocol", "TrainingBudget"),
    "load_formal_experiment_spec": ("formal_protocol", "load_formal_experiment_spec"),
    "validate_formal_experiment_payload": ("formal_protocol", "validate_formal_experiment_payload"),
    "PTOForecasts": ("pto", "PTOForecasts"),
    "PTODispatchCache": ("pto", "PTODispatchCache"),
    "load_pto_cache": ("pto", "load_pto_cache"),
    "save_pto_cache": ("pto", "save_pto_cache"),
    "seasonal_naive_forecasts": ("pto", "seasonal_naive_forecasts"),
    "solve_pto_windows": ("pto", "solve_pto_windows"),
    "DeviceHistoryEncoder": ("model", "DeviceHistoryEncoder"),
    "JointForecastDispatchModel": ("model", "JointForecastDispatchModel"),
    "JointForwardOutput": ("model", "JointForwardOutput"),
    "JointSchedulingProxy": ("model", "JointSchedulingProxy"),
    "StateConditionedScheme2R": ("model", "StateConditionedScheme2R"),
}

__all__ = [
    "CurriculumSpec",
    "JointTrainingContract",
    "JointVariant",
    "load_joint_training_contract",
    "validate_joint_contract",
    "JointNormalization",
    "JointWindowSplit",
    "LPSolveBenchmark",
    "benchmark_lp_generation",
    "build_causal_device_trajectory",
    "build_joint_windows",
    "derive_device_status",
    "fit_joint_normalization",
    "load_joint_split",
    "save_joint_split",
    "AblationSpec",
    "FormalExperimentSpec",
    "MethodSpec",
    "TrainingBudget",
    "load_formal_experiment_spec",
    "validate_formal_experiment_payload",
    "PTOForecasts",
    "PTODispatchCache",
    "load_pto_cache",
    "save_pto_cache",
    "seasonal_naive_forecasts",
    "solve_pto_windows",
    "DeviceHistoryEncoder",
    "JointForecastDispatchModel",
    "JointForwardOutput",
    "JointSchedulingProxy",
    "StateConditionedScheme2R",
]


def __getattr__(name: str):
    """Load PyTorch model symbols only when requested.

    The causal data builder and contract remain usable in lightweight
    environments where the optional PyTorch dependency is not installed.
    """

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
