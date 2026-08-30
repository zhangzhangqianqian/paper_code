"""End-to-end joint forecasting and physics-feasible dispatch package."""

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

    if name in {
        "DeviceHistoryEncoder", "JointForecastDispatchModel", "JointForwardOutput",
        "JointSchedulingProxy", "StateConditionedScheme2R",
    }:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
