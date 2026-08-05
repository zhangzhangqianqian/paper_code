"""Structure-matched single-task references for the Stage 7-R revision."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .models import FullWindowDSTCNEncoder, TaskStepForecastHead


class MatchedSingleTaskScheme2RModel(nn.Module):
    """One independently trained Scheme2R-compatible forecasting task.

    The model keeps the full-window DS-TCN encoder and task-step-specific
    scalar heads used by Scheme2R, but removes every cross-task routing,
    message projection and fusion component.  It receives the common four-task
    load tensor for interface compatibility and reads only ``task_index``.
    """

    def __init__(
        self,
        exog_dim: int,
        task_index: int,
        task_count: int = 4,
        hidden_dim: int = 32,
        lookback: int = 24,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if task_count <= 0 or not 0 <= task_index < task_count:
            raise ValueError("task_index must identify one of the configured tasks")
        if exog_dim <= 0 or lookback <= 0 or horizon <= 0:
            raise ValueError("exog_dim, lookback and horizon must be positive")
        self.task_index = int(task_index)
        self.task_count = int(task_count)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.exog_dim = int(exog_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = FullWindowDSTCNEncoder(
            exog_dim=exog_dim,
            hidden_dim=hidden_dim,
            lookback=lookback,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.head = TaskStepForecastHead(
            task_count=1,
            horizon=horizon,
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
        )

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        if (
            loads.ndim != 3
            or loads.shape[1] != self.lookback
            or loads.shape[2] != self.task_count
        ):
            raise ValueError(
                f"loads must have shape [batch, {self.lookback}, {self.task_count}]"
            )
        if (
            exog.ndim != 3
            or tuple(exog.shape[:2]) != tuple(loads.shape[:2])
            or exog.shape[2] != self.exog_dim
        ):
            raise ValueError(
                "exog must have shape [batch, lookback, exog_dim] matching loads"
            )
        representation = self.encoder(
            loads[:, :, self.task_index : self.task_index + 1], exog
        )
        per_step = representation[:, None, None, :].expand(
            representation.shape[0], self.horizon, 1, self.hidden_dim
        )
        return self.head(per_step)


class MatchedIndependentSTLModel(nn.Module):
    """Four independent full-window STL models with a unified tensor API.

    This wrapper is useful for interface checks.  Formal Stage 6/7-R training
    still trains each child independently so that validation early stopping
    and checkpoints remain task-specific.
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 4,
        hidden_dim: int = 32,
        lookback: int = 24,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if task_count <= 0:
            raise ValueError("task_count must be positive")
        self.task_count = int(task_count)
        self.models = nn.ModuleList(
            [
                MatchedSingleTaskScheme2RModel(
                    exog_dim=exog_dim,
                    task_index=task_index,
                    task_count=task_count,
                    hidden_dim=hidden_dim,
                    lookback=lookback,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                    horizon=horizon,
                    head_hidden_dim=head_hidden_dim,
                )
                for task_index in range(task_count)
            ]
        )

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        predictions = [model(loads, exog) for model in self.models]
        return torch.cat(predictions, dim=-1)
