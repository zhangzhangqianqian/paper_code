"""Atomic closed-loop state and scheduled roll-in records for formal-v4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor

from .formal_v4_recourse import FormalV4RealizedOutcome


def _hash_state(load_history: Tensor, exog_history: Tensor, device_history: Tensor, activity_history: Tensor, origin: np.datetime64, trajectory_id: str) -> str:
    digest = hashlib.sha256()
    for value in (load_history, exog_history, device_history, activity_history):
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    digest.update(np.asarray(origin, dtype="datetime64[ns]").tobytes())
    digest.update(str(trajectory_id).encode("utf-8"))
    return digest.hexdigest()


def _check_history(value: Tensor, shape_tail: tuple[int, ...], name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [B,{shape_tail[0]},{shape_tail[1]}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class FormalV4ClosedLoopState:
    load_history: Tensor
    exog_history: Tensor
    device_history: Tensor
    activity_history: Tensor
    initial_soc: Tensor
    previous_chp: Tensor
    origin: np.datetime64
    trajectory_id: str
    state_hash: str = ""

    def __post_init__(self) -> None:
        _check_history(self.load_history, (24, 4), "load_history")
        _check_history(self.exog_history, (24, 12), "exog_history")
        _check_history(self.device_history, (24, 17), "device_history")
        _check_history(self.activity_history, (24, 6), "activity_history")
        if not bool(torch.isfinite(self.activity_history).all()) or not bool(torch.all((self.activity_history == 0.0) | (self.activity_history == 1.0))):
            raise ValueError("activity_history must be binary and finite")
        batch = self.load_history.shape[0]
        for value, name in ((self.initial_soc, "initial_soc"), (self.previous_chp, "previous_chp")):
            if value.shape != (batch, 1) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must have shape [B,1] and be finite")
        if bool((self.initial_soc < 0.0).any()) or bool((self.initial_soc > 1.0).any()) or bool((self.previous_chp < 0.0).any()):
            raise ValueError("invalid carried state values")
        origin = np.asarray(self.origin, dtype="datetime64[ns]")
        if origin.ndim != 0 or not self.trajectory_id:
            raise ValueError("origin must be a scalar and trajectory_id non-empty")
        computed = _hash_state(self.load_history, self.exog_history, self.device_history, self.activity_history, origin, self.trajectory_id)
        if self.state_hash and self.state_hash != computed:
            raise ValueError("state_hash does not match the state payload")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "state_hash", computed)


@dataclass(frozen=True)
class RollinStateRecord:
    state: FormalV4ClosedLoopState
    same_info_teacher: Tensor | None
    imitation_allowed: bool
    source_checkpoint_sha256: str = ""

    def __post_init__(self) -> None:
        if self.same_info_teacher is not None and not isinstance(self.same_info_teacher, Tensor):
            raise ValueError("same_info_teacher must be a tensor or None")
        if self.imitation_allowed and self.same_info_teacher is None:
            raise ValueError("imitation_allowed requires a same-information teacher")


def advance_formal_v4_state(
    previous: FormalV4ClosedLoopState,
    outcome: FormalV4RealizedOutcome,
    *,
    realized_load: Tensor | None = None,
    realized_exog: Tensor | None = None,
) -> FormalV4ClosedLoopState:
    """Shift all histories once and append only observations revealed at t."""

    batch = previous.load_history.shape[0]
    if outcome.observable_dispatch.shape != (batch, 17) or outcome.activity_indicators.shape != (batch, 6):
        raise ValueError("outcome observable state does not match previous state")
    load = previous.load_history[:, -1, :] if realized_load is None else realized_load
    exog = previous.exog_history[:, -1, :] if realized_exog is None else realized_exog
    if load.shape != (batch, 4) or exog.shape != (batch, 12):
        raise ValueError("realized_load and realized_exog must have shapes [B,4] and [B,12]")
    # Roll-in artifacts are deliberately detached.  A later forward pass will
    # create a fresh graph, preventing accidental backpropagation through old
    # hours collected by the deployment loop.
    next_load = torch.cat((previous.load_history[:, 1:, :], load.detach().unsqueeze(1)), dim=1)
    next_exog = torch.cat((previous.exog_history[:, 1:, :], exog.detach().unsqueeze(1)), dim=1)
    next_device = torch.cat((previous.device_history[:, 1:, :], outcome.observable_dispatch.detach().unsqueeze(1)), dim=1)
    next_activity = torch.cat((previous.activity_history[:, 1:, :], outcome.activity_indicators.detach().unsqueeze(1)), dim=1)
    return FormalV4ClosedLoopState(
        next_load, next_exog, next_device, next_activity,
        outcome.next_soc.detach(), outcome.next_previous_chp.detach(),
        previous.origin + np.timedelta64(1, "h"), previous.trajectory_id,
    )


TeacherMode = Literal["recompute_same_information", "disable_imitation"]


def build_rollin_windows(
    state: FormalV4ClosedLoopState,
    outcome: FormalV4RealizedOutcome,
    *,
    teacher_mode: TeacherMode,
    teacher_solver: Callable[[FormalV4ClosedLoopState], Tensor] | None = None,
    source_checkpoint_sha256: str = "",
) -> RollinStateRecord:
    """Create a post-action record with explicit teacher semantics."""

    if teacher_mode not in {"recompute_same_information", "disable_imitation"}:
        raise ValueError("teacher_mode must be recompute_same_information or disable_imitation")
    next_state = advance_formal_v4_state(state, outcome)
    if teacher_mode == "disable_imitation":
        return RollinStateRecord(next_state, None, False, source_checkpoint_sha256)
    if teacher_solver is None:
        raise ValueError("recompute_same_information requires teacher_solver")
    teacher = teacher_solver(next_state)
    if not isinstance(teacher, Tensor) or teacher.ndim != 3 or tuple(teacher.shape[1:]) != (4, 21) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("teacher_solver must return finite [B,4,21]")
    return RollinStateRecord(next_state, teacher.detach(), True, source_checkpoint_sha256)


def validate_state_teacher_alignment(rollin_record: RollinStateRecord, teacher_record: RollinStateRecord) -> None:
    if rollin_record.state.state_hash != teacher_record.state.state_hash:
        raise ValueError("state_hash mismatch between roll-in and teacher")
    if rollin_record.state.trajectory_id != teacher_record.state.trajectory_id:
        raise ValueError("trajectory_id mismatch between roll-in and teacher")
    if rollin_record.state.origin != teacher_record.state.origin:
        raise ValueError("origin mismatch between roll-in and teacher")


__all__ = [
    "FormalV4ClosedLoopState", "RollinStateRecord", "TeacherMode",
    "advance_formal_v4_state", "build_rollin_windows", "validate_state_teacher_alignment",
]

