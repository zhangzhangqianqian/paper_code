"""阶段3/4：独立、硬共享和静态有向门控的深度可分离因果TCN模型。

模型输入采用[batch, time, channels]，内部转换为Conv1d需要的
[batch, channels, time]。每个任务模型只接收自己的负荷历史和公共外部变量，
不接收其他任务的负荷信息，为后续负迁移评价提供结构匹配的STL参照。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 保留 HEEW 三任务名称以兼容旧实验；Kitakyushu 通过 task_count=4 显式构造。
TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating")
KITAKYUSHU_TASKS: Tuple[str, ...] = (
    "electricity",
    "cooling",
    "heating",
    "gas",
)
KITAKYUSHU_TASK_COUNT = len(KITAKYUSHU_TASKS)
MODEL_NAMES: Tuple[str, ...] = (
    "stl",
    "hard_share",
    "static_gate",
    "dynamic_symmetric",
    "dynamic_directed",
)
SCHEME2R_MODEL_NAME = "scheme2r"
ALL_MODEL_NAMES: Tuple[str, ...] = MODEL_NAMES + (SCHEME2R_MODEL_NAME,)
# Stage 6 uses a strict structure-matched STL reference.  The legacy ``stl``
# model remains available for backwards-compatible scripts, but is not a
# candidate in the scientific selection contract.
MATCHED_STL_MODEL_NAME = "stl_matched"
STAGE6_MODEL_NAMES: Tuple[str, ...] = (
    MATCHED_STL_MODEL_NAME,
    "hard_share",
    "static_gate",
    "dynamic_symmetric",
    "dynamic_directed",
    SCHEME2R_MODEL_NAME,
)


class CausalDepthwiseSeparableConv1d(nn.Module):
    """不使用未来信息的深度可分离一维因果卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("卷积通道数必须为正整数")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("当前TCN要求kernel_size为正奇数")
        if dilation <= 0:
            raise ValueError("dilation必须为正整数")

        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
            padding=0,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("卷积输入必须是[batch, channels, time]")
        padded = F.pad(x, (self.left_padding, 0))
        return self.pointwise(self.depthwise(padded))


class DSTCNResidualBlock(nn.Module):
    """深度可分离因果卷积残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")
        self.conv = CausalDepthwiseSeparableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        output = self.conv(x)
        output = self.activation(output)
        output = self.dropout(output)
        return self.norm(output + residual)


class DSTCNEncoder(nn.Module):
    """将一个任务的历史负荷和公共外生变量编码为任务表示。"""

    def __init__(
        self,
        exog_dim: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if exog_dim < 0:
            raise ValueError("exog_dim不能为负数")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim必须为正整数")
        dilations = tuple(int(dilation) for dilation in dilations)
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations必须包含正整数")

        input_channels = 1 + exog_dim
        blocks = []
        for block_index, dilation in enumerate(dilations):
            blocks.append(
                DSTCNResidualBlock(
                    in_channels=input_channels if block_index == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*blocks)
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.exog_dim = exog_dim
        self.hidden_dim = hidden_dim

    def forward(self, load_history: Tensor, exog_history: Optional[Tensor] = None) -> Tensor:
        if load_history.ndim != 3 or load_history.shape[-1] != 1:
            raise ValueError("load_history必须是[batch, time, 1]")
        batch_size, time_steps, _ = load_history.shape

        if self.exog_dim:
            if exog_history is None:
                raise ValueError("exog_dim大于0时必须提供exog_history")
            if exog_history.ndim != 3:
                raise ValueError("exog_history必须是[batch, time, exog_dim]")
            if exog_history.shape[:2] != (batch_size, time_steps):
                raise ValueError("load_history和exog_history的batch/time维度必须一致")
            if exog_history.shape[-1] != self.exog_dim:
                raise ValueError("exog_history最后一维与exog_dim不一致")
            inputs = torch.cat((load_history, exog_history), dim=-1)
        else:
            inputs = load_history

        encoded = self.encoder(inputs.transpose(1, 2)).transpose(1, 2)
        last_feature = encoded[:, -1, :]
        mean_feature = encoded.mean(dim=1)
        return self.pool_projection(torch.cat((last_feature, mean_feature), dim=-1))


class FullWindowDSTCNEncoder(DSTCNEncoder):
    """方案 2-R 使用的完整历史窗口 DS-TCN 编码器。

    该类独立于旧版 ``DSTCNEncoder``，默认使用 kernel_size=5、
    dilation=(1, 2, 4)，理论感受野为 29 个时间步，可以覆盖当前
    24 小时历史窗口。旧版编码器和模型不会被此类替换。
    """

    def __init__(
        self,
        exog_dim: int,
        hidden_dim: int = 32,
        lookback: int = 24,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        dilations = tuple(int(dilation) for dilation in dilations)
        if lookback <= 0:
            raise ValueError("lookback必须为正整数")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size必须为正奇数")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations必须包含正整数")

        receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        if receptive_field < lookback:
            raise ValueError("DS-TCN理论感受野必须覆盖完整历史窗口")

        super().__init__(
            exog_dim=exog_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.lookback = lookback
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.receptive_field = receptive_field

    def forward(
        self,
        load_history: Tensor,
        exog_history: Optional[Tensor] = None,
    ) -> Tensor:
        if load_history.ndim != 3 or load_history.shape[1] != self.lookback:
            raise ValueError(
                f"load_history必须是[batch, {self.lookback}, 1]"
            )
        if exog_history is not None and exog_history.ndim == 3:
            if exog_history.shape[1] != self.lookback:
                raise ValueError(
                    f"exog_history时间维必须为{self.lookback}"
                )
        return super().forward(load_history, exog_history)


class StateEncoder(nn.Module):
    """将历史外生变量编码为样本级状态向量。

    ``state_history`` 只应包含预测起点及以前可以获得的天气、日历等变量，
    输入形状为 ``[batch, time, input_dim]``，输出形状为
    ``[batch, state_dim]``。可选的 ``task_representations`` 用于后续动态门控
    接入任务表示；当前阶段默认不启用它，以便先独立验证状态编码器。
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        task_repr_dim: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim必须为正整数")
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim和hidden_dim必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")
        if task_repr_dim < 0:
            raise ValueError("task_repr_dim不能为负数")

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.task_repr_dim = task_repr_dim
        self.window_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, state_dim),
            nn.GELU(),
        )
        self.task_projection = (
            nn.Linear(task_repr_dim, state_dim) if task_repr_dim > 0 else None
        )

    def forward(
        self,
        state_history: Tensor,
        task_representations: Optional[Tensor] = None,
    ) -> Tensor:
        if state_history.ndim != 3 or state_history.shape[-1] != self.input_dim:
            raise ValueError(
                "state_history必须是[batch, time, input_dim]"
            )
        if state_history.shape[1] <= 0:
            raise ValueError("state_history至少需要一个时间步")

        encoded = self.window_encoder(state_history)
        last_feature = encoded[:, -1, :]
        mean_feature = encoded.mean(dim=1)
        state = self.pool_projection(
            torch.cat((last_feature, mean_feature), dim=-1)
        )

        if task_representations is not None:
            if self.task_projection is None:
                raise ValueError(
                    "当前StateEncoder未配置task_repr_dim，不能传入任务表示"
                )
            if task_representations.ndim != 3:
                raise ValueError(
                    "task_representations必须是[batch, task_count, task_repr_dim]"
                )
            if task_representations.shape[0] != state_history.shape[0]:
                raise ValueError("state_history和task_representations的batch维度必须一致")
            if task_representations.shape[-1] != self.task_repr_dim:
                raise ValueError(
                    "task_representations最后一维必须等于task_repr_dim"
                )
            task_summary = task_representations.mean(dim=1)
            state = state + self.task_projection(task_summary)

        return state


class EnhancedStateEncoder(nn.Module):
    """方案 2-R 的增强状态编码器。

    对历史外部状态逐时间步编码，并同时汇总最后时刻、均值、标准差
    和首尾变化趋势。该模块只接收预测起点及以前可获得的状态窗口，
    输出固定维度的样本状态向量。
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim必须为正整数")
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim和hidden_dim必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.window_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, state_dim),
            nn.GELU(),
        )

    def forward(self, state_history: Tensor) -> Tensor:
        if state_history.ndim != 3 or state_history.shape[-1] != self.input_dim:
            raise ValueError(
                "state_history必须是[batch, time, input_dim]"
            )
        if state_history.shape[1] <= 0:
            raise ValueError("state_history至少需要一个时间步")

        encoded = self.window_encoder(state_history)
        last_feature = encoded[:, -1, :]
        mean_feature = encoded.mean(dim=1)
        std_feature = encoded.std(dim=1, unbiased=False)
        trend_feature = encoded[:, -1, :] - encoded[:, 0, :]
        summary = torch.cat(
            (last_feature, mean_feature, std_feature, trend_feature),
            dim=-1,
        )
        return self.summary_projection(summary)


class LoadsOnlyStateEncoder(nn.Module):
    """State encoder used by the Scheme2R loads-only control.

    The control removes all meteorological/calendar inputs.  Its routing state
    is therefore derived only from the task representations produced by the
    load-history encoders, rather than from an empty exogenous tensor.
    """

    def __init__(
        self,
        task_count: int,
        task_repr_dim: int,
        state_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if task_count <= 1 or task_repr_dim <= 0:
            raise ValueError("task_count必须大于1且task_repr_dim必须为正整数")
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim和hidden_dim必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")
        self.task_count = int(task_count)
        self.task_repr_dim = int(task_repr_dim)
        self.state_dim = int(state_dim)
        self.summary_projection = nn.Sequential(
            nn.Linear(task_repr_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.GELU(),
        )

    def forward(self, task_representations: Tensor) -> Tensor:
        if task_representations.ndim != 3:
            raise ValueError(
                "task_representations必须是[batch, task_count, task_repr_dim]"
            )
        if tuple(task_representations.shape[1:]) != (
            self.task_count,
            self.task_repr_dim,
        ):
            raise ValueError("task_representations的任务数或表示维度不匹配")
        mean_feature = task_representations.mean(dim=1)
        std_feature = task_representations.std(dim=1, unbiased=False)
        return self.summary_projection(torch.cat((mean_feature, std_feature), dim=-1))


class ForecastStepEmbedding(nn.Module):
    """未来预测步的可学习位置嵌入。"""

    def __init__(self, horizon: int = 4, embedding_dim: int = 4) -> None:
        super().__init__()
        if horizon <= 0 or embedding_dim <= 0:
            raise ValueError("horizon和embedding_dim必须为正整数")
        self.horizon = horizon
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(horizon, embedding_dim)
        self.register_buffer(
            "_default_indices",
            torch.arange(horizon, dtype=torch.long),
            persistent=False,
        )

    def forward(self, step_indices: Optional[Tensor] = None) -> Tensor:
        indices = self._default_indices if step_indices is None else step_indices
        if indices.ndim != 1:
            raise ValueError("step_indices必须是一维整数张量")
        if indices.numel() and (
            indices.min().item() < 0 or indices.max().item() >= self.horizon
        ):
            raise ValueError("step_indices超出预测步范围")
        return self.embedding(indices.to(dtype=torch.long))


class TaskRoleEmbeddings(nn.Module):
    """目标任务和来源任务的独立可学习角色嵌入。"""

    def __init__(self, task_count: int = 4, embedding_dim: int = 8) -> None:
        super().__init__()
        if task_count <= 1 or embedding_dim <= 0:
            raise ValueError("task_count必须大于1且embedding_dim必须为正整数")
        self.task_count = task_count
        self.embedding_dim = embedding_dim
        self.target = nn.Embedding(task_count, embedding_dim)
        self.source = nn.Embedding(task_count, embedding_dim)
        self.register_buffer(
            "_default_indices",
            torch.arange(task_count, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        task_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        indices = self._default_indices if task_indices is None else task_indices
        if indices.ndim != 1:
            raise ValueError("task_indices必须是一维整数张量")
        if indices.numel() and (
            indices.min().item() < 0 or indices.max().item() >= self.task_count
        ):
            raise ValueError("task_indices超出任务范围")
        indices = indices.to(dtype=torch.long)
        return self.target(indices), self.source(indices)


class TwoLevelDirectedRouter(nn.Module):
    """方案 2-R 的总体共享强度与来源分配路由器。

    对每个样本、目标任务和预测步先计算总体共享强度 rho，再在其他
    来源任务之间计算归一化分配 pi，最终得到 g = rho * pi。路由网络
    在所有有序任务对之间共享参数，但目标和来源角色嵌入保持独立，
    因此不会强制正反两个方向的门控相等。
    """

    def __init__(
        self,
        state_dim: int = 16,
        task_repr_dim: int = 32,
        task_count: int = 4,
        task_embedding_dim: int = 8,
        step_embedding_dim: int = 4,
        gate_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or task_repr_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("状态、任务表示和门控隐藏维度必须为正整数")
        if task_count <= 1:
            raise ValueError("task_count必须大于1")
        if task_embedding_dim <= 0 or step_embedding_dim <= 0:
            raise ValueError("任务和预测步嵌入维度必须为正整数")

        self.state_dim = state_dim
        self.task_repr_dim = task_repr_dim
        self.task_count = task_count
        self.task_embedding_dim = task_embedding_dim
        self.step_embedding_dim = step_embedding_dim
        self.gate_hidden_dim = gate_hidden_dim

        rho_input_dim = (
            state_dim + task_repr_dim + task_embedding_dim + step_embedding_dim
        )
        pair_input_dim = (
            state_dim
            + task_repr_dim * 4
            + task_embedding_dim * 2
            + step_embedding_dim
        )
        self.rho_network = nn.Sequential(
            nn.Linear(rho_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.source_network = nn.Sequential(
            nn.Linear(pair_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self,
        state: Tensor,
        task_representations: Tensor,
        step_embeddings: Tensor,
        target_task_embeddings: Tensor,
        source_task_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("state必须是[batch, state_dim]")
        if (
            task_representations.ndim != 3
            or task_representations.shape[1] != self.task_count
            or task_representations.shape[-1] != self.task_repr_dim
        ):
            raise ValueError(
                "task_representations必须是[batch, task_count, task_repr_dim]"
            )
        if (
            step_embeddings.ndim != 2
            or step_embeddings.shape[-1] != self.step_embedding_dim
        ):
            raise ValueError("step_embeddings必须是[horizon, step_embedding_dim]")
        if (
            target_task_embeddings.shape != (
                self.task_count,
                self.task_embedding_dim,
            )
            or source_task_embeddings.shape != (
                self.task_count,
                self.task_embedding_dim,
            )
        ):
            raise ValueError(
                "目标和来源任务嵌入必须是[task_count, task_embedding_dim]"
            )
        if task_representations.shape[0] != state.shape[0]:
            raise ValueError("state和task_representations的batch维度必须一致")

        batch_size = state.shape[0]
        horizon = step_embeddings.shape[0]
        task_count = self.task_count

        # rho: [batch, horizon, target_task]
        state_for_rho = state[:, None, None, :].expand(
            batch_size, horizon, task_count, self.state_dim
        )
        target_repr = task_representations[:, None, :, :].expand(
            batch_size, horizon, task_count, self.task_repr_dim
        )
        target_role = target_task_embeddings[None, None, :, :].expand(
            batch_size, horizon, task_count, self.task_embedding_dim
        )
        step_for_rho = step_embeddings[None, :, None, :].expand(
            batch_size, horizon, task_count, self.step_embedding_dim
        )
        rho_input = torch.cat(
            (state_for_rho, target_repr, target_role, step_for_rho),
            dim=-1,
        )
        rho = torch.sigmoid(self.rho_network(rho_input).squeeze(-1))

        # pi: [batch, horizon, target_task, source_task]
        target_pair = task_representations[:, None, :, None, :].expand(
            batch_size, horizon, task_count, task_count, self.task_repr_dim
        )
        source_pair = task_representations[:, None, None, :, :].expand(
            batch_size, horizon, task_count, task_count, self.task_repr_dim
        )
        state_for_pair = state[:, None, None, None, :].expand(
            batch_size,
            horizon,
            task_count,
            task_count,
            self.state_dim,
        )
        target_role_pair = target_task_embeddings[None, None, :, None, :].expand(
            batch_size,
            horizon,
            task_count,
            task_count,
            self.task_embedding_dim,
        )
        source_role_pair = source_task_embeddings[None, None, None, :, :].expand(
            batch_size,
            horizon,
            task_count,
            task_count,
            self.task_embedding_dim,
        )
        step_for_pair = step_embeddings[None, :, None, None, :].expand(
            batch_size,
            horizon,
            task_count,
            task_count,
            self.step_embedding_dim,
        )
        pair_input = torch.cat(
            (
                state_for_pair,
                target_pair,
                source_pair,
                source_pair - target_pair,
                target_pair * source_pair,
                target_role_pair,
                source_role_pair,
                step_for_pair,
            ),
            dim=-1,
        )
        logits = self.source_network(pair_input).squeeze(-1)
        diagonal = torch.eye(
            task_count,
            dtype=torch.bool,
            device=logits.device,
        )[None, None, :, :]
        masked_logits = logits.masked_fill(diagonal, float("-inf"))
        pi = F.softmax(masked_logits, dim=-1).masked_fill(diagonal, 0.0)
        gates = pi * rho.unsqueeze(-1)
        return rho, pi, gates


class LowRankDirectedMessageProjector(nn.Module):
    """将来源任务表示映射为有序目标—来源任务消息。"""

    def __init__(
        self,
        task_count: int = 4,
        task_repr_dim: int = 32,
        rank: int = 8,
    ) -> None:
        super().__init__()
        if task_count <= 1:
            raise ValueError("task_count必须大于1")
        if task_repr_dim <= 0 or rank <= 0:
            raise ValueError("task_repr_dim和rank必须为正整数")

        self.task_count = task_count
        self.task_repr_dim = task_repr_dim
        self.rank = rank
        self.source_down = nn.ModuleList(
            [
                nn.Linear(task_repr_dim, rank, bias=False)
                for _ in range(task_count)
            ]
        )
        self.target_up = nn.ModuleList(
            [
                nn.Linear(rank, task_repr_dim, bias=False)
                for _ in range(task_count)
            ]
        )
        self.pair_scale = nn.Parameter(
            torch.ones(task_count, task_count, rank)
        )

    def forward(self, task_representations: Tensor) -> Tensor:
        if (
            task_representations.ndim != 3
            or task_representations.shape[1] != self.task_count
            or task_representations.shape[-1] != self.task_repr_dim
        ):
            raise ValueError(
                "task_representations必须是[batch, task_count, task_repr_dim]"
            )

        source_low = torch.stack(
            [
                projection(task_representations[:, source_index, :])
                for source_index, projection in enumerate(self.source_down)
            ],
            dim=1,
        )
        pair_messages = []
        for target_index, projection in enumerate(self.target_up):
            target_messages = []
            for source_index in range(self.task_count):
                scaled = source_low[:, source_index, :] * self.pair_scale[
                    target_index, source_index, :
                ]
                target_messages.append(projection(scaled))
            pair_messages.append(torch.stack(target_messages, dim=1))
        messages = torch.stack(pair_messages, dim=1)

        diagonal = torch.eye(
            self.task_count,
            dtype=torch.bool,
            device=messages.device,
        )[None, :, :, None]
        return messages.masked_fill(diagonal, 0.0)


class StableResidualTaskFusion(nn.Module):
    """按 rho 和 pi 聚合消息并执行稳定的任务残差融合。"""

    def __init__(
        self,
        task_count: int = 4,
        task_repr_dim: int = 32,
        initial_alpha: float = 1e-2,
    ) -> None:
        super().__init__()
        if task_count <= 1 or task_repr_dim <= 0:
            raise ValueError("task_count必须大于1且task_repr_dim必须为正整数")
        if initial_alpha < 0.0:
            raise ValueError("initial_alpha不能为负数")

        self.task_count = task_count
        self.task_repr_dim = task_repr_dim
        self.message_norm = nn.LayerNorm(task_repr_dim)
        self.alpha = nn.Parameter(
            torch.full((task_count,), float(initial_alpha))
        )

    def forward(
        self,
        task_representations: Tensor,
        rho: Tensor,
        pi: Tensor,
        pair_messages: Tensor,
    ) -> Tensor:
        if (
            task_representations.ndim != 3
            or task_representations.shape[1] != self.task_count
            or task_representations.shape[-1] != self.task_repr_dim
        ):
            raise ValueError(
                "task_representations必须是[batch, task_count, task_repr_dim]"
            )
        if rho.ndim != 3 or rho.shape[0] != task_representations.shape[0]:
            raise ValueError("rho必须是[batch, horizon, task_count]")
        if rho.shape[-1] != self.task_count:
            raise ValueError("rho最后一维必须等于task_count")
        if (
            pi.ndim != 4
            or pi.shape[:3] != (
                task_representations.shape[0],
                rho.shape[1],
                self.task_count,
            )
            or pi.shape[-1] != self.task_count
        ):
            raise ValueError("pi必须是[batch, horizon, task_count, task_count]")
        if (
            pair_messages.ndim != 4
            or pair_messages.shape
            != (
                task_representations.shape[0],
                self.task_count,
                self.task_count,
                self.task_repr_dim,
            )
        ):
            raise ValueError(
                "pair_messages必须是[batch, task_count, task_count, task_repr_dim]"
            )

        messages = torch.einsum("bhts,btsd->bhtd", pi, pair_messages)
        normalized_messages = self.message_norm(messages)
        base = task_representations[:, None, :, :]
        alpha = self.alpha[None, None, :, None]
        return base + alpha * rho[:, :, :, None] * normalized_messages


class TaskStepForecastHead(nn.Module):
    """每个任务、每个预测步独立的标量预测头。"""

    def __init__(
        self,
        task_count: int = 4,
        horizon: int = 4,
        hidden_dim: int = 32,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if task_count <= 0 or horizon <= 0:
            raise ValueError("task_count和horizon必须为正整数")
        if hidden_dim <= 0 or head_hidden_dim <= 0:
            raise ValueError("hidden_dim和head_hidden_dim必须为正整数")

        self.task_count = task_count
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(hidden_dim, head_hidden_dim),
                            nn.GELU(),
                            nn.Linear(head_hidden_dim, 1),
                        )
                        for _ in range(horizon)
                    ]
                )
                for _ in range(task_count)
            ]
        )

    def forward(self, fused_representations: Tensor) -> Tensor:
        if (
            fused_representations.ndim != 4
            or fused_representations.shape[2] != self.task_count
            or fused_representations.shape[3] != self.hidden_dim
        ):
            raise ValueError(
                "fused_representations必须是[batch, horizon, task_count, hidden_dim]"
            )
        if fused_representations.shape[1] != self.horizon:
            raise ValueError("fused_representations的预测步维度不一致")

        predictions = []
        for step_index in range(self.horizon):
            task_predictions = []
            for task_index in range(self.task_count):
                prediction = self.heads[task_index][step_index](
                    fused_representations[:, step_index, task_index, :]
                )
                task_predictions.append(prediction.squeeze(-1))
            predictions.append(torch.stack(task_predictions, dim=-1))
        return torch.stack(predictions, dim=1)


class TaskForecastHead(nn.Module):
    """单个任务的多步预测头。"""

    def __init__(
        self, hidden_dim: int, horizon: int, head_hidden_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        if hidden_dim <= 1 or horizon <= 0:
            raise ValueError("hidden_dim必须大于1且horizon必须为正整数")
        head_hidden_dim = hidden_dim // 2 if head_hidden_dim is None else int(head_hidden_dim)
        if head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim必须为正整数")
        self.head_hidden_dim = head_hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, horizon),
        )

    def forward(self, representation: Tensor) -> Tensor:
        return self.network(representation)


class SingleTaskDSTCN(nn.Module):
    """单个能源任务的DS-TCN点预测模型。"""

    def __init__(
        self,
        exog_dim: int,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon必须为正整数")
        self.encoder = DSTCNEncoder(
            exog_dim=exog_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.head = TaskForecastHead(hidden_dim=hidden_dim, horizon=horizon)
        self.exog_dim = exog_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

    def encode(self, load_history: Tensor, exog_history: Optional[Tensor] = None) -> Tensor:
        return self.encoder(load_history, exog_history)

    def forward(self, load_history: Tensor, exog_history: Optional[Tensor] = None) -> Tensor:
        """输出[batch, horizon]，只预测一个任务。"""

        return self.head(self.encode(load_history, exog_history))


class IndependentSTLModel(nn.Module):
    """多个结构相同但参数独立的单任务模型组合。

    输入：
        loads: [batch, time, task_count]
        exog: [batch, time, exog_dim]
    输出：
        prediction: [batch, horizon, task_count]
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 3,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
    ) -> None:
        super().__init__()
        if task_count <= 0:
            raise ValueError("task_count必须为正整数")
        self.task_count = task_count
        self.horizon = horizon
        self.models = nn.ModuleList(
            [
                SingleTaskDSTCN(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                    horizon=horizon,
                )
                for _ in range(task_count)
            ]
        )

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        if loads.ndim != 3 or loads.shape[-1] != self.task_count:
            raise ValueError("loads必须是[batch, time, task_count]")
        if exog is not None and exog.shape[:2] != loads.shape[:2]:
            raise ValueError("loads和exog的batch/time维度必须一致")
        predictions = [
            model(loads[:, :, task_index : task_index + 1], exog)
            for task_index, model in enumerate(self.models)
        ]
        return torch.stack(predictions, dim=-1)


class HardShareMTLModel(nn.Module):
    """硬参数共享多任务模型。

    每个任务仍只接收自己的负荷历史和公共外生变量，但所有任务共享同一个
    DS-TCN 编码器，并使用独立预测头。这样可以作为动态任务门控模型之前的
    结构匹配 MTL 参照，避免把显式跨任务输入混合误认为共享收益。
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 3,
        hidden_dim: int = 32,
        lookback: Optional[int] = None,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if task_count <= 0:
            raise ValueError("task_count必须为正整数")
        self.task_count = task_count
        self.horizon = horizon
        encoder_type = FullWindowDSTCNEncoder if lookback is not None else DSTCNEncoder
        encoder_kwargs = {
            "exog_dim": exog_dim,
            "hidden_dim": hidden_dim,
            "kernel_size": kernel_size,
            "dilations": dilations,
            "dropout": dropout,
        }
        if lookback is not None:
            encoder_kwargs["lookback"] = int(lookback)
        self.encoder = encoder_type(
            **encoder_kwargs,
        )
        self.heads = nn.ModuleList(
            [
                TaskForecastHead(
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    head_hidden_dim=head_hidden_dim,
                )
                for _ in range(task_count)
            ]
        )

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        if loads.ndim != 3 or loads.shape[-1] != self.task_count:
            raise ValueError("loads必须是[batch, time, task_count]")
        if exog is not None and exog.shape[:2] != loads.shape[:2]:
            raise ValueError("loads和exog的batch/time维度必须一致")
        predictions = []
        for task_index, head in enumerate(self.heads):
            representation = self.encoder(
                loads[:, :, task_index : task_index + 1], exog
            )
            predictions.append(head(representation))
        return torch.stack(predictions, dim=-1)


class StaticDirectedMTLModel(nn.Module):
    """带全局静态有向门控的多任务模型。

    每个任务先使用参数独立的 DS-TCN 编码器得到表示 ``h_i``，随后通过
    全局可学习门控矩阵和有序任务对消息投影进行跨任务残差融合：

    ``h_tilde_i = h_i + sum_j G[i, j] P[i <- j](h_j)``。

    门控矩阵的行是目标任务、列是来源任务；矩阵不依赖样本状态，因而
    属于静态门控。主对角线在输出矩阵中固定为0，避免把自身信息误认为
    跨任务消息。所有门控为0时，融合结果严格退化为任务独立表示。
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 3,
        hidden_dim: int = 32,
        lookback: Optional[int] = None,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: Optional[int] = None,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if task_count <= 0:
            raise ValueError("task_count必须为正整数")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate必须位于(0,1)")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim必须为正整数")

        self.task_count = task_count
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        encoder_type = FullWindowDSTCNEncoder if lookback is not None else DSTCNEncoder
        self.encoders = nn.ModuleList(
            [
                encoder_type(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    **({"lookback": int(lookback)} if lookback is not None else {}),
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
                for _ in range(task_count)
            ]
        )
        self.heads = nn.ModuleList(
            [
                TaskForecastHead(
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    head_hidden_dim=head_hidden_dim,
                )
                for _ in range(task_count)
            ]
        )
        self.message_projections = nn.ModuleDict(
            {
                f"{target}_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target in range(task_count)
                for source in range(task_count)
                if target != source
            }
        )
        initial_logit = float(torch.logit(torch.tensor(initial_gate)))
        self.gate_logits = nn.Parameter(
            torch.full((task_count, task_count), initial_logit)
        )

    def _validate_inputs(self, loads: Tensor, exog: Optional[Tensor]) -> None:
        if loads.ndim != 3 or loads.shape[-1] != self.task_count:
            raise ValueError("loads必须是[batch, time, task_count]")
        if exog is not None and exog.shape[:2] != loads.shape[:2]:
            raise ValueError("loads和exog的batch/time维度必须一致")

    def get_gate_matrix(self) -> Tensor:
        """返回[目标任务,来源任务]的静态门控矩阵，范围为[0,1]。"""

        off_diagonal = ~torch.eye(
            self.task_count, dtype=torch.bool, device=self.gate_logits.device
        )
        return torch.sigmoid(self.gate_logits).masked_fill(~off_diagonal, 0.0)

    def encode_tasks(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        """编码所有任务，返回[batch, task_count, hidden_dim]。"""

        self._validate_inputs(loads, exog)
        representations = [
            encoder(loads[:, :, task_index : task_index + 1], exog)
            for task_index, encoder in enumerate(self.encoders)
        ]
        return torch.stack(representations, dim=1)

    def fuse_representations(self, representations: Tensor) -> Tensor:
        """按静态有向门控执行消息传递和残差融合。"""

        expected_shape = (self.task_count, self.hidden_dim)
        if representations.ndim != 3 or tuple(representations.shape[1:]) != expected_shape:
            raise ValueError(
                "representations必须是[batch, task_count, hidden_dim]"
            )
        gates = self.get_gate_matrix()
        fused = representations.clone()
        for target in range(self.task_count):
            message = torch.zeros_like(representations[:, target, :])
            for source in range(self.task_count):
                if source == target:
                    continue
                projected = self.message_projections[f"{target}_{source}"](
                    representations[:, source, :]
                )
                message = message + gates[target, source] * projected
            fused[:, target, :] = fused[:, target, :] + message
        return fused

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        representations = self.encode_tasks(loads, exog)
        fused = self.fuse_representations(representations)
        predictions = [
            head(fused[:, task_index, :])
            for task_index, head in enumerate(self.heads)
        ]
        return torch.stack(predictions, dim=-1)


class DynamicSymmetricMTLModel(nn.Module):
    """样本级动态对称任务门控模型。

    每个样本先由历史外生变量和任务表示得到状态向量，再为每个无序任务
    对生成一个门控值。无序任务对的门控值同时写入两个方向，因此严格满足
    ``G[i, j] == G[j, i]``；任务自身的对角线固定为0。跨任务消息仍使用有序
    的目标/来源投影，随后通过残差方式加入目标任务表示。
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 3,
        hidden_dim: int = 32,
        lookback: Optional[int] = None,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: Optional[int] = None,
        state_dim: int = 16,
        state_hidden_dim: int = 32,
        gate_hidden_dim: int = 16,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if exog_dim <= 0:
            raise ValueError("动态门控需要至少一个外生变量")
        if task_count <= 1:
            raise ValueError("动态对称门控至少需要两个任务")
        if state_dim <= 0 or state_hidden_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("状态和门控隐藏维度必须为正整数")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate必须位于(0,1)")

        self.task_count = task_count
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.state_dim = state_dim
        self.pair_indices = tuple(
            (target, source)
            for target in range(task_count)
            for source in range(target + 1, task_count)
        )
        encoder_type = FullWindowDSTCNEncoder if lookback is not None else DSTCNEncoder
        self.encoders = nn.ModuleList(
            [
                encoder_type(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    **({"lookback": int(lookback)} if lookback is not None else {}),
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
                for _ in range(task_count)
            ]
        )
        self.state_encoder = StateEncoder(
            input_dim=exog_dim,
            state_dim=state_dim,
            hidden_dim=state_hidden_dim,
            dropout=dropout,
            task_repr_dim=hidden_dim,
        )
        self.gate_network = nn.Sequential(
            nn.Linear(state_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, len(self.pair_indices)),
        )
        initial_logit = float(torch.logit(torch.tensor(initial_gate)))
        final_layer = self.gate_network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.constant_(final_layer.bias, initial_logit)
        self.heads = nn.ModuleList(
            [
                TaskForecastHead(
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    head_hidden_dim=head_hidden_dim,
                )
                for _ in range(task_count)
            ]
        )
        self.message_projections = nn.ModuleDict(
            {
                f"{target}_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target in range(task_count)
                for source in range(task_count)
                if target != source
            }
        )

    def _validate_inputs(self, loads: Tensor, exog: Tensor) -> None:
        if loads.ndim != 3 or loads.shape[-1] != self.task_count:
            raise ValueError("loads必须是[batch, time, task_count]")
        if exog.ndim != 3 or exog.shape[:2] != loads.shape[:2]:
            raise ValueError("exog必须是[batch, time, exog_dim]且与loads的batch/time一致")
        if exog.shape[-1] != self.state_encoder.input_dim:
            raise ValueError("exog最后一维与exog_dim不一致")

    def encode_tasks(self, loads: Tensor, exog: Tensor) -> Tensor:
        """编码所有任务，返回 ``[batch, task_count, hidden_dim]``。"""

        self._validate_inputs(loads, exog)
        representations = [
            encoder(loads[:, :, task_index : task_index + 1], exog)
            for task_index, encoder in enumerate(self.encoders)
        ]
        return torch.stack(representations, dim=1)

    def encode_state(self, exog: Tensor, representations: Tensor) -> Tensor:
        """根据历史状态和任务表示生成 ``[batch, state_dim]``。"""

        if exog.ndim != 3 or exog.shape[-1] != self.state_encoder.input_dim:
            raise ValueError("exog必须是[batch, time, exog_dim]")
        if representations.ndim != 3:
            raise ValueError("representations必须是[batch, task_count, hidden_dim]")
        if tuple(representations.shape[1:]) != (self.task_count, self.hidden_dim):
            raise ValueError("representations的task_count/hidden_dim与模型不一致")
        if representations.shape[0] != exog.shape[0]:
            raise ValueError("exog和representations的batch维度必须一致")
        return self.state_encoder(exog, representations)

    def get_gate_matrix(self, state: Tensor) -> Tensor:
        """将状态映射为 ``[batch, task_count, task_count]`` 对称门控矩阵。"""

        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("state必须是[batch, state_dim]")
        pair_gates = torch.sigmoid(self.gate_network(state))
        gates = state.new_zeros((state.shape[0], self.task_count, self.task_count))
        for pair_index, (target, source) in enumerate(self.pair_indices):
            value = pair_gates[:, pair_index]
            gates[:, target, source] = value
            gates[:, source, target] = value
        return gates

    def fuse_representations(self, representations: Tensor, gates: Tensor) -> Tensor:
        """根据样本级门控执行有序消息传递和残差融合。"""

        expected_shape = (self.task_count, self.hidden_dim)
        if representations.ndim != 3 or tuple(representations.shape[1:]) != expected_shape:
            raise ValueError("representations必须是[batch, task_count, hidden_dim]")
        expected_gate_shape = (representations.shape[0], self.task_count, self.task_count)
        if tuple(gates.shape) != expected_gate_shape:
            raise ValueError("gates必须是[batch, task_count, task_count]")
        fused = representations.clone()
        for target in range(self.task_count):
            message = torch.zeros_like(representations[:, target, :])
            for source in range(self.task_count):
                if source == target:
                    continue
                projected = self.message_projections[f"{target}_{source}"](
                    representations[:, source, :]
                )
                message = message + gates[:, target, source].unsqueeze(-1) * projected
            fused[:, target, :] = fused[:, target, :] + message
        return fused

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        representations = self.encode_tasks(loads, exog)
        state = self.encode_state(exog, representations)
        gates = self.get_gate_matrix(state)
        fused = self.fuse_representations(representations, gates)
        predictions = [
            head(fused[:, task_index, :])
            for task_index, head in enumerate(self.heads)
        ]
        return torch.stack(predictions, dim=-1)


class DynamicDirectedMTLModel(nn.Module):
    """样本级动态有向任务门控模型。

    与动态对称模型不同，本模型对每个有序目标/来源任务对分别计算门控值。
    门控网络同时接收样本状态、目标和来源任务表示以及两者的任务嵌入，因此
    ``G[i, j]`` 与 ``G[j, i]`` 可以不同。消息投影和残差融合保持不变。
    """

    def __init__(
        self,
        exog_dim: int,
        task_count: int = 3,
        hidden_dim: int = 32,
        lookback: Optional[int] = None,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
        head_hidden_dim: Optional[int] = None,
        state_dim: int = 16,
        state_hidden_dim: int = 32,
        gate_hidden_dim: int = 16,
        task_embedding_dim: int = 8,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if exog_dim <= 0:
            raise ValueError("动态门控需要至少一个外生变量")
        if task_count <= 1:
            raise ValueError("动态有向门控至少需要两个任务")
        if state_dim <= 0 or state_hidden_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("状态和门控隐藏维度必须为正整数")
        if task_embedding_dim <= 0:
            raise ValueError("task_embedding_dim必须为正整数")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate必须位于(0,1)")

        self.task_count = task_count
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.state_dim = state_dim
        self.task_embedding_dim = task_embedding_dim
        self.pair_indices = tuple(
            (target, source)
            for target in range(task_count)
            for source in range(task_count)
            if target != source
        )
        encoder_type = FullWindowDSTCNEncoder if lookback is not None else DSTCNEncoder
        self.encoders = nn.ModuleList(
            [
                encoder_type(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    **({"lookback": int(lookback)} if lookback is not None else {}),
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
                for _ in range(task_count)
            ]
        )
        self.state_encoder = StateEncoder(
            input_dim=exog_dim,
            state_dim=state_dim,
            hidden_dim=state_hidden_dim,
            dropout=dropout,
            task_repr_dim=hidden_dim,
        )
        self.task_embeddings = nn.Parameter(
            torch.randn(task_count, task_embedding_dim) * 0.02
        )
        gate_input_dim = state_dim + 2 * hidden_dim + 2 * task_embedding_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        initial_logit = float(torch.logit(torch.tensor(initial_gate)))
        final_layer = self.gate_network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.constant_(final_layer.bias, initial_logit)
        self.heads = nn.ModuleList(
            [
                TaskForecastHead(
                    hidden_dim=hidden_dim,
                    horizon=horizon,
                    head_hidden_dim=head_hidden_dim,
                )
                for _ in range(task_count)
            ]
        )
        self.message_projections = nn.ModuleDict(
            {
                f"{target}_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target in range(task_count)
                for source in range(task_count)
                if target != source
            }
        )

    def _validate_inputs(self, loads: Tensor, exog: Tensor) -> None:
        if loads.ndim != 3 or loads.shape[-1] != self.task_count:
            raise ValueError("loads必须是[batch, time, task_count]")
        if exog.ndim != 3 or exog.shape[:2] != loads.shape[:2]:
            raise ValueError("exog必须是[batch, time, exog_dim]且与loads的batch/time一致")
        if exog.shape[-1] != self.state_encoder.input_dim:
            raise ValueError("exog最后一维与exog_dim不一致")

    def encode_tasks(self, loads: Tensor, exog: Tensor) -> Tensor:
        """编码所有任务，返回 ``[batch, task_count, hidden_dim]``。"""

        self._validate_inputs(loads, exog)
        representations = [
            encoder(loads[:, :, task_index : task_index + 1], exog)
            for task_index, encoder in enumerate(self.encoders)
        ]
        return torch.stack(representations, dim=1)

    def encode_state(self, exog: Tensor, representations: Tensor) -> Tensor:
        """根据历史状态和任务表示生成 ``[batch, state_dim]``。"""

        if exog.ndim != 3 or exog.shape[-1] != self.state_encoder.input_dim:
            raise ValueError("exog必须是[batch, time, exog_dim]")
        if representations.ndim != 3:
            raise ValueError("representations必须是[batch, task_count, hidden_dim]")
        if tuple(representations.shape[1:]) != (self.task_count, self.hidden_dim):
            raise ValueError("representations的task_count/hidden_dim与模型不一致")
        if representations.shape[0] != exog.shape[0]:
            raise ValueError("exog和representations的batch维度必须一致")
        return self.state_encoder(exog, representations)

    def get_gate_matrix(self, state: Tensor, representations: Tensor) -> Tensor:
        """将状态和有序任务信息映射为 ``[batch, task_count, task_count]``。"""

        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("state必须是[batch, state_dim]")
        if representations.ndim != 3:
            raise ValueError("representations必须是[batch, task_count, hidden_dim]")
        if tuple(representations.shape[1:]) != (self.task_count, self.hidden_dim):
            raise ValueError("representations的task_count/hidden_dim与模型不一致")
        if representations.shape[0] != state.shape[0]:
            raise ValueError("state和representations的batch维度必须一致")

        batch_size = state.shape[0]
        gates = state.new_zeros((batch_size, self.task_count, self.task_count))
        for target, source in self.pair_indices:
            target_embedding = self.task_embeddings[target].expand(batch_size, -1)
            source_embedding = self.task_embeddings[source].expand(batch_size, -1)
            pair_features = torch.cat(
                (
                    state,
                    representations[:, target, :],
                    representations[:, source, :],
                    target_embedding,
                    source_embedding,
                ),
                dim=-1,
            )
            value = torch.sigmoid(self.gate_network(pair_features).squeeze(-1))
            gates[:, target, source] = value
        return gates

    def fuse_representations(self, representations: Tensor, gates: Tensor) -> Tensor:
        """根据样本级有向门控执行消息传递和残差融合。"""

        expected_shape = (self.task_count, self.hidden_dim)
        if representations.ndim != 3 or tuple(representations.shape[1:]) != expected_shape:
            raise ValueError("representations必须是[batch, task_count, hidden_dim]")
        expected_gate_shape = (representations.shape[0], self.task_count, self.task_count)
        if tuple(gates.shape) != expected_gate_shape:
            raise ValueError("gates必须是[batch, task_count, task_count]")
        fused = representations.clone()
        for target in range(self.task_count):
            message = torch.zeros_like(representations[:, target, :])
            for source in range(self.task_count):
                if source == target:
                    continue
                projected = self.message_projections[f"{target}_{source}"](
                    representations[:, source, :]
                )
                message = message + gates[:, target, source].unsqueeze(-1) * projected
            fused[:, target, :] = fused[:, target, :] + message
        return fused

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        representations = self.encode_tasks(loads, exog)
        state = self.encode_state(exog, representations)
        gates = self.get_gate_matrix(state, representations)
        fused = self.fuse_representations(representations, gates)
        predictions = [
            head(fused[:, task_index, :])
            for task_index, head in enumerate(self.heads)
        ]
        return torch.stack(predictions, dim=-1)


class Scheme2RModel(nn.Module):
    """方案 2-R：状态和预测步相关的两级有向任务路由模型。

    该模型新增于 Legacy Dynamic Directed Gate 之后，使用完整历史窗口
    DS-TCN、增强状态上下文、rho × pi 两级路由、低秩消息投影和任务—
    预测步专属预测头。旧版模型类保持不变。
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
        state_dim: int = 16,
        state_hidden_dim: int = 32,
        gate_hidden_dim: int = 16,
        task_embedding_dim: int = 8,
        step_embedding_dim: int = 4,
        rank: int = 8,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if exog_dim < 0:
            raise ValueError("Scheme2R的exog_dim不能为负数")
        if task_count <= 1:
            raise ValueError("Scheme2R至少需要两个任务")
        if hidden_dim <= 0 or lookback <= 0 or horizon <= 0:
            raise ValueError("hidden_dim、lookback和horizon必须为正整数")

        self.task_count = task_count
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.horizon = horizon
        self.exog_dim = exog_dim
        self.encoders = nn.ModuleList(
            [
                FullWindowDSTCNEncoder(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    lookback=lookback,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
                for _ in range(task_count)
            ]
        )
        if exog_dim > 0:
            self.state_encoder = EnhancedStateEncoder(
                input_dim=exog_dim,
                state_dim=state_dim,
                hidden_dim=state_hidden_dim,
                dropout=dropout,
            )
            self.state_source = "historical_exogenous_summary"
        else:
            self.state_encoder = LoadsOnlyStateEncoder(
                task_count=task_count,
                task_repr_dim=hidden_dim,
                state_dim=state_dim,
                hidden_dim=state_hidden_dim,
                dropout=dropout,
            )
            self.state_source = "load_task_representation_summary"
        self.step_embeddings = ForecastStepEmbedding(
            horizon=horizon,
            embedding_dim=step_embedding_dim,
        )
        self.role_embeddings = TaskRoleEmbeddings(
            task_count=task_count,
            embedding_dim=task_embedding_dim,
        )
        self.router = TwoLevelDirectedRouter(
            state_dim=state_dim,
            task_repr_dim=hidden_dim,
            task_count=task_count,
            task_embedding_dim=task_embedding_dim,
            step_embedding_dim=step_embedding_dim,
            gate_hidden_dim=gate_hidden_dim,
        )
        self.message_projector = LowRankDirectedMessageProjector(
            task_count=task_count,
            task_repr_dim=hidden_dim,
            rank=rank,
        )
        self.fusion = StableResidualTaskFusion(
            task_count=task_count,
            task_repr_dim=hidden_dim,
        )
        self.head = TaskStepForecastHead(
            task_count=task_count,
            horizon=horizon,
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
        )

    def _validate_inputs(self, loads: Tensor, exog: Optional[Tensor]) -> None:
        if (
            loads.ndim != 3
            or loads.shape[1] != self.lookback
            or loads.shape[-1] != self.task_count
        ):
            raise ValueError(
                f"loads必须是[batch, {self.lookback}, {self.task_count}]"
            )
        if self.exog_dim == 0:
            if exog is not None and (
                exog.ndim != 3
                or exog.shape[:2] != loads.shape[:2]
                or exog.shape[-1] != 0
            ):
                raise ValueError(
                    "loads-only Scheme2R 的 exog 必须为 None 或 [batch, lookback, 0]"
                )
            return
        if (
            exog is None
            or exog.ndim != 3
            or exog.shape[:2] != loads.shape[:2]
            or exog.shape[-1] != self.exog_dim
        ):
            raise ValueError(
                "exog必须是[batch, lookback, exog_dim]且与loads的batch/time一致"
            )

    def encode_tasks(self, loads: Tensor, exog: Optional[Tensor]) -> Tensor:
        """编码所有任务，返回 [batch, task_count, hidden_dim]。"""

        self._validate_inputs(loads, exog)
        representations = [
            encoder(loads[:, :, task_index : task_index + 1], exog)
            for task_index, encoder in enumerate(self.encoders)
        ]
        return torch.stack(representations, dim=1)

    def encode_state(
        self,
        exog: Optional[Tensor],
        representations: Tensor,
    ) -> Tensor:
        if self.exog_dim == 0:
            return self.state_encoder(representations)
        if exog is None or exog.ndim != 3 or exog.shape[-1] != self.exog_dim:
            raise ValueError("exog必须是[batch, time, exog_dim]")
        if exog.shape[1] != self.lookback:
            raise ValueError(f"exog时间维必须为{self.lookback}")
        return self.state_encoder(exog)

    def forward_with_details(
        self,
        loads: Tensor,
        exog: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        representations = self.encode_tasks(loads, exog)
        state = self.encode_state(exog, representations)
        step_embeddings = self.step_embeddings()
        target_embeddings, source_embeddings = self.role_embeddings()
        rho, pi, gates = self.router(
            state,
            representations,
            step_embeddings,
            target_embeddings,
            source_embeddings,
        )
        pair_messages = self.message_projector(representations)
        fused = self.fusion(representations, rho, pi, pair_messages)
        predictions = self.head(fused)
        details = {
            "representations": representations,
            "state": state,
            "step_embeddings": step_embeddings,
            "target_task_embeddings": target_embeddings,
            "source_task_embeddings": source_embeddings,
            "rho": rho,
            "pi": pi,
            "gates": gates,
            "pair_messages": pair_messages,
            "fused_representations": fused,
        }
        return predictions, details

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        predictions, _ = self.forward_with_details(loads, exog)
        return predictions


def build_forecasting_model(model_name: str, **kwargs) -> nn.Module:
    """使用统一名称构造旧版内部模型或新增 Scheme2R。"""

    builders = {
        "stl": IndependentSTLModel,
        "hard_share": HardShareMTLModel,
        "static_gate": StaticDirectedMTLModel,
        "dynamic_symmetric": DynamicSymmetricMTLModel,
        "dynamic_directed": DynamicDirectedMTLModel,
        SCHEME2R_MODEL_NAME: Scheme2RModel,
    }
    if model_name == MATCHED_STL_MODEL_NAME:
        # Import lazily because stage7_reference intentionally depends on the
        # encoder/head definitions in this module.
        from .stage7_reference import MatchedIndependentSTLModel

        return MatchedIndependentSTLModel(**kwargs)
    try:
        builder = builders[model_name]
    except KeyError as error:
        raise ValueError(
            f"未知模型 {model_name!r}；可选模型为 {ALL_MODEL_NAMES}"
        ) from error
    return builder(**kwargs)


def build_kitakyushu_forecasting_model(model_name: str, **kwargs) -> nn.Module:
    """按 Kitakyushu 四任务协议构造内部模型。

    旧的 ``build_forecasting_model`` 默认保留三任务兼容性；新数据集调用
    本函数时会自动固定 ``task_count=4``，除非调用方显式传入该参数。
    """

    kwargs.setdefault("task_count", KITAKYUSHU_TASK_COUNT)
    return build_forecasting_model(model_name, **kwargs)


def count_trainable_parameters(module: nn.Module) -> int:
    """统计可训练参数量，供后续轻量化比较使用。"""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
