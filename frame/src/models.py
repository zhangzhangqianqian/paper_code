"""阶段3/4：独立、硬共享和静态有向门控的深度可分离因果TCN模型。

模型输入采用[batch, time, channels]，内部转换为Conv1d需要的
[batch, channels, time]。每个任务模型只接收自己的负荷历史和公共外部变量，
不接收其他任务的负荷信息，为后续负迁移评价提供结构匹配的STL参照。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating")


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


class TaskForecastHead(nn.Module):
    """单个任务的多步预测头。"""

    def __init__(self, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        if hidden_dim <= 1 or horizon <= 0:
            raise ValueError("hidden_dim必须大于1且horizon必须为正整数")
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, horizon),
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
    """三个结构相同但参数独立的单任务模型组合。

    输入：
        loads: [batch, time, 3]
        exog: [batch, time, exog_dim]
    输出：
        prediction: [batch, horizon, 3]
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

    每个任务仍只接收自己的负荷历史和公共外生变量，但三个任务共享同一个
    DS-TCN 编码器，并使用独立预测头。这样可以作为动态任务门控模型之前的
    结构匹配 MTL 参照，避免把显式跨任务输入混合误认为共享收益。
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
        self.encoder = DSTCNEncoder(
            exog_dim=exog_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.heads = nn.ModuleList(
            [TaskForecastHead(hidden_dim=hidden_dim, horizon=horizon) for _ in range(task_count)]
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
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.1,
        horizon: int = 4,
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
        self.encoders = nn.ModuleList(
            [
                DSTCNEncoder(
                    exog_dim=exog_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout,
                )
                for _ in range(task_count)
            ]
        )
        self.heads = nn.ModuleList(
            [TaskForecastHead(hidden_dim=hidden_dim, horizon=horizon) for _ in range(task_count)]
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
        """编码三个任务，返回[batch, task_count, hidden_dim]。"""

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


def count_trainable_parameters(module: nn.Module) -> int:
    """统计可训练参数量，供后续轻量化比较使用。"""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
