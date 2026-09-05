"""Fail-closed Gate 1 views and candidate eligibility for formal-v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Gate1ViewV43:
    name: str
    indices: np.ndarray
    weights: np.ndarray
    secondary_only: bool

    def __post_init__(self) -> None:
        if self.name not in {"full_chronology", "stress_sample"}:
            raise ValueError("unknown Gate 1 view")
        if self.indices.ndim != 1 or self.weights.shape != self.indices.shape:
            raise ValueError("Gate 1 view indices and weights must be one-dimensional and aligned")
        if self.indices.size == 0 or not np.isfinite(self.weights).all() or np.any(self.weights <= 0.0):
            raise ValueError("Gate 1 view must contain positive finite weights")
        if self.name == "full_chronology" and self.secondary_only:
            raise ValueError("full chronology cannot be secondary")
        if self.name == "stress_sample" and not self.secondary_only:
            raise ValueError("stress sample must be secondary")


def _length(selection: Any) -> int:
    if isinstance(selection, Mapping):
        for key in ("forecast_target", "target", "load_history"):
            if key in selection:
                return int(np.asarray(selection[key]).shape[0])
    for key in ("forecast_target", "target", "load_history"):
        if hasattr(selection, key):
            return int(np.asarray(getattr(selection, key)).shape[0])
    try:
        return int(len(selection))
    except TypeError as exc:
        raise ValueError("selection must expose a window count") from exc


def build_full_chronology_view(selection: Any) -> Gate1ViewV43:
    count = _length(selection)
    return Gate1ViewV43(
        name="full_chronology",
        indices=np.arange(count, dtype=np.int64),
        weights=np.ones(count, dtype=np.float64),
        secondary_only=False,
    )


def build_stress_view(indices: np.ndarray) -> Gate1ViewV43:
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size == 0 or np.any(selected < 0):
        raise ValueError("stress indices must be non-empty and non-negative")
    return Gate1ViewV43(
        name="stress_sample",
        indices=selected,
        weights=np.ones(selected.shape[0], dtype=np.float64),
        secondary_only=True,
    )


def candidate_is_eligible_v43(candidate: Mapping[str, Any]) -> bool:
    """Require both views, physical validity, improvement, and gradient evidence."""

    views = candidate.get("views")
    if not isinstance(views, Mapping):
        return False
    full = views.get("full_chronology")
    stress = views.get("stress_sample")
    if not isinstance(full, Mapping) or not isinstance(stress, Mapping):
        return False
    required_full = (
        "forecast_guardrail_passed", "inactive_leakage_guardrail_passed",
        "regime_guardrail_passed",
    )
    if not all(bool(full.get(key, False)) for key in required_full):
        return False
    if not bool(stress.get("forecast_guardrail_passed", False)):
        return False
    return all(bool(candidate.get(key, False)) for key in (
        "physical_feasibility_passed", "dispatch_improvement_passed", "gradient_boundary_passed",
    ))


def authorize_gate1_v43(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if candidate_is_eligible_v43(row)]
    selected = None if not eligible else min(
        eligible,
        key=lambda row: (float(row.get("full_chronology_penalized_objective", np.inf)), str(row.get("candidate_id", ""))),
    )
    return {
        "authorized_gate2": selected is not None,
        "eligible_candidate_count": len(eligible),
        "selected_candidate_id": None if selected is None else selected.get("candidate_id"),
        "evaluation_year_accessed": False,
        "paper_eligible": False,
    }


__all__ = [
    "Gate1ViewV43", "authorize_gate1_v43", "build_full_chronology_view",
    "build_stress_view", "candidate_is_eligible_v43",
]
