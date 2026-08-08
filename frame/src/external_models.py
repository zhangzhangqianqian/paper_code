"""阶段5外部轻量基线模型。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


KITAKYUSHU_TASK_COUNT = 4


class MovingAverageDecomposition(nn.Module):
    """使用边界复制的滑动平均分解趋势和季节项。"""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving average kernel_size必须为正奇数")
        self.kernel_size = kernel_size

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 3:
            raise ValueError("values必须是[batch, time, channels]")
        padding = (self.kernel_size - 1) // 2
        if padding:
            front = values[:, :1, :].expand(-1, padding, -1)
            back = values[:, -1:, :].expand(-1, padding, -1)
            padded = torch.cat((front, values, back), dim=1)
        else:
            padded = values
        trend = F.avg_pool1d(
            padded.transpose(1, 2),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        seasonal = values - trend
        return seasonal, trend


class DLinearBaseline(nn.Module):
    """面向多能源负荷的轻量 DLinear 基线。

    该基线只使用历史负荷，不接收天气或日历外生变量。输入为
    ``[batch, lookback, task_count]``，输出为 ``[batch, horizon, task_count]``。
    标准化由训练流程负责，模型内部只执行趋势/季节分解和两个线性映射。
    当前采用跨任务共享的时间线性层（等价于标准 DLinear 的
    ``individual=False`` 轻量配置）。
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 4,
        task_count: int = 3,
        moving_avg: int = 5,
    ) -> None:
        super().__init__()
        if lookback <= 0 or horizon <= 0 or task_count <= 0:
            raise ValueError("lookback、horizon和task_count必须为正整数")
        self.lookback = lookback
        self.horizon = horizon
        self.task_count = task_count
        self.channel_shared = True
        self.decomposition = MovingAverageDecomposition(moving_avg)
        self.seasonal_linear = nn.Linear(lookback, horizon)
        self.trend_linear = nn.Linear(lookback, horizon)

    def forward(
        self,
        loads: Tensor,
        exog: Optional[Tensor] = None,
    ) -> Tensor:
        if loads.ndim != 3 or tuple(loads.shape[1:]) != (
            self.lookback,
            self.task_count,
        ):
            raise ValueError(
                "loads必须是[batch, lookback, task_count]且维度与模型一致"
            )
        if exog is not None:
            raise ValueError("DLinearBaseline的公平性契约固定为loads_only")
        seasonal, trend = self.decomposition(loads)
        seasonal = self.seasonal_linear(seasonal.transpose(1, 2))
        trend = self.trend_linear(trend.transpose(1, 2))
        return (seasonal + trend).transpose(1, 2)


class MMoELiteBaseline(nn.Module):
    """轻量化 MMoE-like 多能源负荷预测基线。

    该实现保留 MMoE 的核心结构：多个共享专家、每个任务独立的门控网络，
    以及任务专属预测头。它只使用历史负荷和历史外生变量，不使用未来外生
    变量，也不包含 Shao 等人的 Frequency 或 STIM 模块，因此不是完整模型复现。

    输入：
        ``loads``: ``[batch, lookback, task_count]``
        ``exog``: ``[batch, lookback, exog_dim]``（必须提供）
    输出：
        ``[batch, horizon, task_count]``
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 4,
        task_count: int = 3,
        exog_dim: int = 14,
        expert_count: int = 4,
        expert_hidden_dim: int = 32,
        representation_dim: int = 32,
        head_hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback <= 0 or horizon <= 0 or task_count <= 0:
            raise ValueError("lookback、horizon和task_count必须为正整数")
        if exog_dim <= 0:
            raise ValueError("MMoE-lite要求exog_dim为正整数")
        if expert_count <= 0:
            raise ValueError("expert_count必须为正整数")
        if expert_hidden_dim <= 0 or representation_dim <= 0 or head_hidden_dim <= 0:
            raise ValueError("MLP隐藏维度必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")

        self.lookback = lookback
        self.horizon = horizon
        self.task_count = task_count
        self.exog_dim = exog_dim
        self.expert_count = expert_count
        self.expert_hidden_dim = expert_hidden_dim
        self.representation_dim = representation_dim
        self.head_hidden_dim = head_hidden_dim
        self.input_dim = lookback * (task_count + exog_dim)

        def make_expert() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.input_dim, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, representation_dim),
                nn.GELU(),
            )

        self.experts = nn.ModuleList(
            [make_expert() for _ in range(expert_count)]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(self.input_dim, expert_count) for _ in range(task_count)]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(representation_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, horizon),
                )
                for _ in range(task_count)
            ]
        )

    def _flatten_inputs(self, loads: Tensor, exog: Optional[Tensor]) -> Tensor:
        if loads.ndim != 3 or tuple(loads.shape[1:]) != (
            self.lookback,
            self.task_count,
        ):
            raise ValueError(
                "loads必须是[batch, lookback, task_count]且维度与模型一致"
            )
        if exog is None:
            raise ValueError("MMoELiteBaseline必须提供历史exog")
        if exog.ndim != 3 or tuple(exog.shape[1:]) != (
            self.lookback,
            self.exog_dim,
        ):
            raise ValueError(
                "exog必须是[batch, lookback, exog_dim]且维度与模型一致"
            )
        if exog.shape[0] != loads.shape[0]:
            raise ValueError("loads和exog的batch维度必须一致")
        return torch.cat((loads, exog), dim=-1).reshape(loads.shape[0], -1)

    def gate_weights(self, loads: Tensor, exog: Optional[Tensor]) -> Tensor:
        """返回每个样本、每个任务对专家的选择权重 ``[batch, task_count, expert_count]``。"""

        flattened = self._flatten_inputs(loads, exog)
        logits = torch.stack(
            [gate(flattened) for gate in self.gates], dim=1
        )
        return torch.softmax(logits, dim=-1)

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        flattened = self._flatten_inputs(loads, exog)
        expert_outputs = torch.stack(
            [expert(flattened) for expert in self.experts], dim=1
        )
        gates = torch.softmax(
            torch.stack([gate(flattened) for gate in self.gates], dim=1),
            dim=-1,
        )
        task_representations = torch.einsum(
            "bte,ber->btr", gates, expert_outputs
        )
        predictions = [
            head(task_representations[:, task_index, :])
            for task_index, head in enumerate(self.heads)
        ]
        return torch.stack(predictions, dim=-1)


class PLEExpert(nn.Module):
    """Two-layer lightweight expert used by a PLE extraction layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("PLE expert dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError("PLE expert input must be [batch, features]")
        return self.network(values)


class PLECGCLayer(nn.Module):
    """One Customized Gate Control layer for the two-level PLE-lite model.

    A task gate can see only that task's private experts and the shared
    experts.  The optional shared gate can see every private expert and every
    shared expert, which creates the shared stream consumed by the next level.
    """

    def __init__(
        self,
        input_dim: int,
        task_count: int,
        shared_expert_count: int,
        task_expert_count: int,
        expert_hidden_dim: int,
        representation_dim: int,
        dropout: float,
        *,
        produce_shared: bool,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or task_count <= 0:
            raise ValueError("input_dim and task_count must be positive")
        if shared_expert_count <= 0 or task_expert_count <= 0:
            raise ValueError("PLE requires positive shared and task expert counts")
        self.input_dim = input_dim
        self.task_count = task_count
        self.shared_expert_count = shared_expert_count
        self.task_expert_count = task_expert_count
        self.representation_dim = representation_dim
        self.produce_shared = produce_shared

        def make_expert() -> PLEExpert:
            return PLEExpert(
                input_dim,
                expert_hidden_dim,
                representation_dim,
                dropout,
            )

        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList([make_expert() for _ in range(task_expert_count)])
                for _ in range(task_count)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [make_expert() for _ in range(shared_expert_count)]
        )
        task_gate_width = task_expert_count + shared_expert_count
        self.task_gates = nn.ModuleList(
            [nn.Linear(input_dim, task_gate_width) for _ in range(task_count)]
        )
        self.shared_gate = (
            nn.Linear(
                input_dim,
                task_count * task_expert_count + shared_expert_count,
            )
            if produce_shared
            else None
        )

    def _validate_inputs(
        self,
        task_inputs: Sequence[Tensor],
        shared_input: Tensor,
    ) -> None:
        if len(task_inputs) != self.task_count:
            raise ValueError(
                f"expected {self.task_count} task streams, got {len(task_inputs)}"
            )
        if shared_input.ndim != 2 or shared_input.shape[1] != self.input_dim:
            raise ValueError("shared stream has an unexpected shape")
        batch_size = shared_input.shape[0]
        for values in task_inputs:
            if values.ndim != 2 or tuple(values.shape) != (
                batch_size,
                self.input_dim,
            ):
                raise ValueError("task stream has an unexpected shape")

    def forward(
        self,
        task_inputs: Sequence[Tensor],
        shared_input: Tensor,
    ) -> tuple[list[Tensor], Optional[Tensor], Dict[str, Tensor]]:
        self._validate_inputs(task_inputs, shared_input)
        private_outputs = [
            [expert(task_inputs[index]) for expert in self.task_experts[index]]
            for index in range(self.task_count)
        ]
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]

        task_gate_values: list[Tensor] = []
        task_outputs: list[Tensor] = []
        for task_index in range(self.task_count):
            gate = torch.softmax(
                self.task_gates[task_index](task_inputs[task_index]), dim=-1
            )
            candidates = torch.stack(
                [*private_outputs[task_index], *shared_outputs], dim=1
            )
            task_outputs.append(torch.einsum("be,ber->br", gate, candidates))
            task_gate_values.append(gate)

        shared_output: Optional[Tensor] = None
        gates: Dict[str, Tensor] = {
            "task": torch.stack(task_gate_values, dim=1)
        }
        if self.shared_gate is not None:
            shared_weights = torch.softmax(self.shared_gate(shared_input), dim=-1)
            shared_candidates = torch.stack(
                [
                    *[
                        output
                        for task_outputs_for_one_task in private_outputs
                        for output in task_outputs_for_one_task
                    ],
                    *shared_outputs,
                ],
                dim=1,
            )
            shared_output = torch.einsum(
                "be,ber->br", shared_weights, shared_candidates
            )
            gates["shared"] = shared_weights
        return task_outputs, shared_output, gates


class PLELiteBaseline(nn.Module):
    """Two-level PLE adaptation for four-task multi-energy forecasting.

    The model consumes historical loads and historical exogenous variables,
    but never future exogenous variables.  It preserves PLE's progressive
    shared/private extraction while keeping expert widths CPU-friendly.
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 4,
        task_count: int = 4,
        exog_dim: int = 12,
        shared_expert_count: int = 2,
        task_expert_count: int = 1,
        expert_hidden_dim: int = 32,
        representation_dim: int = 32,
        head_hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback <= 0 or horizon <= 0 or task_count <= 0 or exog_dim <= 0:
            raise ValueError("lookback, horizon, task_count and exog_dim must be positive")
        if min(expert_hidden_dim, representation_dim, head_hidden_dim) <= 0:
            raise ValueError("PLE hidden dimensions must be positive")
        self.lookback = lookback
        self.horizon = horizon
        self.task_count = task_count
        self.exog_dim = exog_dim
        self.input_dim = lookback * (task_count + exog_dim)
        self.layer1 = PLECGCLayer(
            self.input_dim,
            task_count,
            shared_expert_count,
            task_expert_count,
            expert_hidden_dim,
            representation_dim,
            dropout,
            produce_shared=True,
        )
        self.layer2 = PLECGCLayer(
            representation_dim,
            task_count,
            shared_expert_count,
            task_expert_count,
            expert_hidden_dim,
            representation_dim,
            dropout,
            produce_shared=False,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(representation_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, horizon),
                )
                for _ in range(task_count)
            ]
        )

    def _flatten_inputs(self, loads: Tensor, exog: Optional[Tensor]) -> Tensor:
        if loads.ndim != 3 or tuple(loads.shape[1:]) != (
            self.lookback,
            self.task_count,
        ):
            raise ValueError("loads must be [batch, lookback, task_count]")
        if exog is None:
            raise ValueError("PLELiteBaseline requires historical exogenous variables")
        if exog.ndim != 3 or tuple(exog.shape[1:]) != (
            self.lookback,
            self.exog_dim,
        ):
            raise ValueError("exog must be [batch, lookback, exog_dim]")
        if exog.shape[0] != loads.shape[0]:
            raise ValueError("loads and exog batch dimensions must match")
        return torch.cat((loads, exog), dim=-1).reshape(loads.shape[0], -1)

    def _route(
        self,
        loads: Tensor,
        exog: Optional[Tensor],
    ) -> tuple[list[Tensor], Dict[str, Tensor]]:
        flattened = self._flatten_inputs(loads, exog)
        first_tasks, first_shared, first_gates = self.layer1(
            [flattened for _ in range(self.task_count)], flattened
        )
        if first_shared is None:
            raise RuntimeError("the first PLE layer must produce a shared stream")
        second_tasks, second_shared, second_gates = self.layer2(
            first_tasks, first_shared
        )
        if second_shared is not None:
            raise RuntimeError("the final PLE layer must not produce a shared stream")
        return second_tasks, {
            "layer1_task": first_gates["task"],
            "layer1_shared": first_gates["shared"],
            "layer2_task": second_gates["task"],
        }

    def gate_weights(
        self,
        loads: Tensor,
        exog: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        return self._route(loads, exog)[1]

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        task_representations, _ = self._route(loads, exog)
        predictions = [
            self.heads[index](task_representations[index])
            for index in range(self.task_count)
        ]
        return torch.stack(predictions, dim=-1)


class STARCore(nn.Module):
    """SOFTS 的 Star Aggregate-Redistribute 核心模块。

    输入的第二维是负荷通道，模块先将所有通道聚合为全局核心，再把核心
    表示分发回每个通道并完成通道级融合。
    """

    def __init__(
        self,
        d_series: int,
        d_core: int,
        stochastic_pooling: bool = True,
    ) -> None:
        super().__init__()
        if d_series <= 0 or d_core <= 0:
            raise ValueError("d_series和d_core必须为正整数")
        self.d_series = d_series
        self.d_core = d_core
        self.stochastic_pooling = stochastic_pooling
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.d_series:
            raise ValueError("STAR输入必须是[batch, channels, d_series]")
        batch_size, channels, _ = values.shape
        combined = self.gen2(F.gelu(self.gen1(values)))

        if self.training and self.stochastic_pooling:
            ratio = F.softmax(combined, dim=1).permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, self.d_core, 1).permute(0, 2, 1)
            core = torch.gather(combined, 1, indices).repeat(1, channels, 1)
        else:
            weights = F.softmax(combined, dim=1)
            core = torch.sum(combined * weights, dim=1, keepdim=True)
            core = core.repeat(1, channels, 1)

        fused = torch.cat((values, core), dim=-1)
        fused = self.gen4(F.gelu(self.gen3(fused)))
        return fused


class SOFTSResidualBlock(nn.Module):
    """带 STAR 和前馈残差的轻量 SOFTS 编码块。"""

    def __init__(
        self,
        d_model: int,
        d_core: int,
        d_ff: int,
        dropout: float,
        stochastic_pooling: bool,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model和d_ff必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")
        self.star = STARCore(d_model, d_core, stochastic_pooling)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, values: Tensor) -> Tensor:
        values = self.norm1(values + self.dropout(self.star(values)))
        feedforward = self.ffn2(self.dropout(F.gelu(self.ffn1(values))))
        return self.norm2(values + self.dropout(feedforward))


class SOFTSBaseline(nn.Module):
    """SOFTS 的最小本地兼容适配版本。

    该类保留官方实现的反转时间嵌入、STAR 通道聚合—分发和多步投影，
    但不复用官方仓库的数据加载器与旧版依赖。它是 ``loads_only`` 外部
    基线，不是 SOFTS 原论文的完整复现实验。
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 4,
        task_count: int = 3,
        d_model: int = 32,
        d_core: int = 16,
        d_ff: int = 64,
        e_layers: int = 1,
        dropout: float = 0.1,
        use_instance_norm: bool = True,
        stochastic_pooling: bool = True,
    ) -> None:
        super().__init__()
        if lookback <= 0 or horizon <= 0 or task_count <= 0:
            raise ValueError("lookback、horizon和task_count必须为正整数")
        if d_model <= 0 or d_core <= 0 or d_ff <= 0 or e_layers <= 0:
            raise ValueError("SOFTS隐藏维度和层数必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout必须位于[0,1)")

        self.lookback = lookback
        self.horizon = horizon
        self.task_count = task_count
        self.d_model = d_model
        self.d_core = d_core
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.use_instance_norm = use_instance_norm
        self.stochastic_pooling = stochastic_pooling
        self.temporal_embedding = nn.Sequential(
            nn.Linear(lookback, d_model),
            nn.Dropout(dropout),
        )
        self.encoder = nn.ModuleList(
            [
                SOFTSResidualBlock(
                    d_model=d_model,
                    d_core=d_core,
                    d_ff=d_ff,
                    dropout=dropout,
                    stochastic_pooling=stochastic_pooling,
                )
                for _ in range(e_layers)
            ]
        )
        self.projection = nn.Linear(d_model, horizon)

    def _validate_loads(self, loads: Tensor, exog: Optional[Tensor]) -> None:
        if loads.ndim != 3 or tuple(loads.shape[1:]) != (
            self.lookback,
            self.task_count,
        ):
            raise ValueError(
                "loads必须是[batch, lookback, task_count]且维度与模型一致"
            )
        if exog is not None:
            raise ValueError("SOFTSBaseline的最小适配版本固定为loads_only")

    def forward(self, loads: Tensor, exog: Optional[Tensor] = None) -> Tensor:
        self._validate_loads(loads, exog)
        if self.use_instance_norm:
            means = loads.mean(dim=1, keepdim=True).detach()
            centered = loads - means
            scales = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            encoded_input = centered / scales
        else:
            means = None
            scales = None
            encoded_input = loads

        # 反转时间嵌入：每个通道的历史窗口成为一个通道 token。
        encoded = self.temporal_embedding(encoded_input.transpose(1, 2))
        for block in self.encoder:
            encoded = block(encoded)
        prediction = self.projection(encoded).transpose(1, 2)

        if self.use_instance_norm:
            prediction = prediction * scales[:, :1, :]
            prediction = prediction + means[:, :1, :]
        return prediction
