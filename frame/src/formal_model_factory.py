"""Single construction interface for the frozen formal forecasting models.

This module centralizes construction only.  It does not select a model,
change a candidate configuration, or read test data; callers must supply the
hyperparameters frozen by the relevant experiment contract.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from torch import nn

from .external_models import MMoELiteBaseline, PLELiteBaseline
from .models import build_forecasting_model


FORMAL_MODEL_NAMES = (
    "hard_share",
    "dynamic_symmetric",
    "mmoe_lite",
    "ple_lite",
    "scheme2r",
)


def _required(hyperparameters: Mapping[str, object], key: str) -> object:
    if key not in hyperparameters:
        raise ValueError(f"missing frozen hyperparameter: {key}")
    return hyperparameters[key]


def _int(hyperparameters: Mapping[str, object], key: str) -> int:
    return int(_required(hyperparameters, key))


def _float(hyperparameters: Mapping[str, object], key: str) -> float:
    return float(_required(hyperparameters, key))


def _head_hidden_dim(hyperparameters: Mapping[str, object]) -> int:
    return _int(hyperparameters, "prediction_head_hidden_dim")


def _dilations(hyperparameters: Mapping[str, object], key: str = "dilations") -> Sequence[int]:
    values = _required(hyperparameters, key)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{key} must be a non-empty sequence")
    return tuple(int(value) for value in values)


def build_formal_forecasting_model(
    model_name: str,
    hyperparameters: Mapping[str, object],
    *,
    exog_dim: int,
    lookback: int = 24,
    horizon: int = 4,
    task_count: int = 4,
) -> nn.Module:
    """Build one frozen formal model with the shared experiment interface.

    Supported names are the five same-track models used by the topology
    protocol: Hard-Share, Dynamic-Symmetric, MMoE-lite, PLE-lite and Scheme2R.
    The two external names accept either hyphens or underscores.
    """

    canonical = str(model_name).strip().lower().replace("-", "_")
    if canonical not in FORMAL_MODEL_NAMES:
        raise ValueError(
            f"unsupported formal model {model_name!r}; expected {FORMAL_MODEL_NAMES}"
        )
    if exog_dim <= 0 or lookback <= 0 or horizon <= 0 or task_count <= 0:
        raise ValueError("exog_dim, lookback, horizon and task_count must be positive")

    if canonical == "mmoe_lite":
        return MMoELiteBaseline(
            lookback=lookback,
            horizon=horizon,
            task_count=task_count,
            exog_dim=exog_dim,
            expert_count=_int(hyperparameters, "expert_count"),
            expert_hidden_dim=_int(hyperparameters, "expert_hidden_dim"),
            representation_dim=_int(hyperparameters, "representation_dim"),
            head_hidden_dim=_head_hidden_dim(hyperparameters),
            dropout=_float(hyperparameters, "dropout"),
        )

    if canonical == "ple_lite":
        return PLELiteBaseline(
            lookback=lookback,
            horizon=horizon,
            task_count=task_count,
            exog_dim=exog_dim,
            shared_expert_count=_int(hyperparameters, "shared_expert_count"),
            task_expert_count=_int(hyperparameters, "task_expert_count"),
            expert_hidden_dim=_int(hyperparameters, "expert_hidden_dim"),
            representation_dim=_int(hyperparameters, "representation_dim"),
            head_hidden_dim=_head_hidden_dim(hyperparameters),
            dropout=_float(hyperparameters, "dropout"),
        )

    common = {
        "exog_dim": exog_dim,
        "task_count": task_count,
        "hidden_dim": _int(hyperparameters, "hidden_dim"),
        "dropout": _float(hyperparameters, "dropout"),
        "horizon": horizon,
        "head_hidden_dim": _head_hidden_dim(hyperparameters),
    }
    if canonical == "scheme2r":
        common.update(
            {
                "lookback": lookback,
                "kernel_size": _int(
                    hyperparameters,
                    "scheme2r_kernel_size"
                    if "scheme2r_kernel_size" in hyperparameters
                    else "kernel_size",
                ),
                "dilations": _dilations(
                    hyperparameters,
                    "scheme2r_dilations"
                    if "scheme2r_dilations" in hyperparameters
                    else "dilations",
                ),
                "rank": _int(hyperparameters, "scheme2r_rank"),
                "gate_hidden_dim": _int(hyperparameters, "scheme2r_gate_hidden_dim"),
                "step_embedding_dim": _int(
                    hyperparameters, "scheme2r_step_embedding_dim"
                ),
            }
        )
    else:
        common.update(
            {
                "lookback": lookback,
                "kernel_size": _int(hyperparameters, "kernel_size"),
                "dilations": _dilations(hyperparameters),
            }
        )
        # Dynamic-Symmetric's defaults are part of its frozen H1 contract.
        # Only pass optional dimensions when a contract explicitly freezes them.
        if canonical == "dynamic_symmetric":
            for key in ("state_dim", "state_hidden_dim", "gate_hidden_dim"):
                if key in hyperparameters:
                    common[key] = int(hyperparameters[key])

    return build_forecasting_model(canonical, **common)


__all__ = ["FORMAL_MODEL_NAMES", "build_formal_forecasting_model"]
