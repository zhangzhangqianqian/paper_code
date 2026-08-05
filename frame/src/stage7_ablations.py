"""Stage 7.1: the ordered A0--A4 ablation model interfaces.

The ablations are intentionally implemented as separate, named model
configurations.  They are not trained in this module; Stage 7.2 owns the
smoke-training protocol and later stages own formal runs.

All models use the Kitakyushu tensor convention::

    loads: [batch, lookback, task_count]
    exog:  [batch, lookback, exog_dim]
    output: [batch, horizon, task_count]

The definitions follow the frozen recursive sequence in the methodology:

* A0 -- legacy dynamic directed gate;
* A1 -- A0 with the full-window DS-TCN encoder;
* A2 -- A1 with prediction-step-dependent directed gates;
* A3 -- A2 with rho-times-pi two-level routing;
* A4 -- A3 with low-rank messages, normalized stable residual fusion and
  task-step-specific scalar heads (Scheme2R).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from .models import (
    DynamicDirectedMTLModel,
    FullWindowDSTCNEncoder,
    Scheme2RModel,
    StateEncoder,
    TaskForecastHead,
    TaskRoleEmbeddings,
    TwoLevelDirectedRouter,
    ForecastStepEmbedding,
)


ABLATION_NAMES: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")
ABLATION_DESCRIPTIONS: Mapping[str, str] = {
    "A0": "Legacy Dynamic Directed Gate",
    "A1": "A0 + full-window DS-TCN",
    "A2": "A1 + prediction-step-dependent directed gates",
    "A3": "A2 + rho-times-pi two-level directed routing",
    "A4": "A3 + low-rank messages, stable residual fusion and task-step heads (Scheme2R)",
}


@dataclass(frozen=True)
class AblationSpec:
    """Machine-readable record for one recursive ablation."""

    ablation_id: str
    description: str
    encoder: str
    routing: str
    message_projection: str
    fusion: str
    head: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "ablation_id": self.ablation_id,
            "description": self.description,
            "encoder": self.encoder,
            "routing": self.routing,
            "message_projection": self.message_projection,
            "fusion": self.fusion,
            "head": self.head,
        }


ABLATION_SPECS: Tuple[AblationSpec, ...] = (
    AblationSpec(
        "A0",
        ABLATION_DESCRIPTIONS["A0"],
        "legacy_dstcn",
        "per-edge_sigmoid_shared_across_horizon",
        "full_rank",
        "plain_residual",
        "task_head_four_step",
    ),
    AblationSpec(
        "A1",
        ABLATION_DESCRIPTIONS["A1"],
        "full_window_dstcn",
        "per-edge_sigmoid_shared_across_horizon",
        "full_rank",
        "plain_residual",
        "task_head_four_step",
    ),
    AblationSpec(
        "A2",
        ABLATION_DESCRIPTIONS["A2"],
        "full_window_dstcn",
        "per-edge_sigmoid_per_horizon_step",
        "full_rank",
        "plain_residual",
        "shared_task_scalar_head_per_step",
    ),
    AblationSpec(
        "A3",
        ABLATION_DESCRIPTIONS["A3"],
        "full_window_dstcn",
        "rho_times_pi_per_horizon_step",
        "full_rank",
        "plain_residual_fixed_alpha_1",
        "shared_task_scalar_head_per_step",
    ),
    AblationSpec(
        "A4",
        ABLATION_DESCRIPTIONS["A4"],
        "full_window_dstcn",
        "rho_times_pi_per_horizon_step",
        "low_rank",
        "layer_norm_learned_residual_scale",
        "task_step_scalar_heads",
    ),
)


def _spec(ablation_id: str) -> AblationSpec:
    for item in ABLATION_SPECS:
        if item.ablation_id == ablation_id:
            return item
    raise KeyError(ablation_id)


class A0LegacyDynamicDirectedGate(DynamicDirectedMTLModel):
    """A0: the already implemented legacy dynamic directed model."""

    ablation_id = "A0"
    ablation_description = ABLATION_DESCRIPTIONS[ablation_id]

    def __init__(self, exog_dim: int, task_count: int = 4, **kwargs) -> None:
        super().__init__(exog_dim=exog_dim, task_count=task_count, **kwargs)

    def forward_with_details(
        self, loads: Tensor, exog: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        representations = self.encode_tasks(loads, exog)
        state = self.encode_state(exog, representations)
        gates = self.get_gate_matrix(state, representations)
        fused = self.fuse_representations(representations, gates)
        predictions = torch.stack(
            [head(fused[:, index, :]) for index, head in enumerate(self.heads)],
            dim=-1,
        )
        return predictions, {
            "representations": representations,
            "state": state,
            "gates": gates,
            "fused_representations": fused,
        }

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        return self.forward_with_details(loads, exog)[0]


class _FullWindowLegacyBase(nn.Module):
    """Shared implementation for A1--A3 before Scheme2R-specific modules."""

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
        head_hidden_dim: int = 16,
        step_dependent: bool = False,
        two_level_router: bool = False,
    ) -> None:
        super().__init__()
        if task_count <= 1:
            raise ValueError("task_count must be greater than one")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive")
        if exog_dim <= 0:
            raise ValueError("exog_dim must be positive")
        if step_dependent and step_embedding_dim <= 0:
            raise ValueError("step_embedding_dim must be positive")

        self.task_count = task_count
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.horizon = horizon
        self.state_dim = state_dim
        self.task_embedding_dim = task_embedding_dim
        self.step_embedding_dim = step_embedding_dim
        self.head_hidden_dim = head_hidden_dim
        self.step_dependent = step_dependent
        self.two_level_router = two_level_router
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
        self.state_encoder = StateEncoder(
            input_dim=exog_dim,
            state_dim=state_dim,
            hidden_dim=state_hidden_dim,
            dropout=dropout,
        )
        self.task_embeddings = nn.Parameter(
            torch.randn(task_count, task_embedding_dim) * 0.02
        )
        self.pair_indices = tuple(
            (target, source)
            for target in range(task_count)
            for source in range(task_count)
            if target != source
        )

        if two_level_router:
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
        else:
            self.step_embeddings = (
                ForecastStepEmbedding(
                    horizon=horizon,
                    embedding_dim=step_embedding_dim,
                )
                if step_dependent
                else None
            )
            gate_input_dim = state_dim + 2 * hidden_dim + 2 * task_embedding_dim
            if step_dependent:
                gate_input_dim += step_embedding_dim
            gate_output_dim = 1
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, gate_output_dim),
            )
            initial_logit = float(torch.logit(torch.tensor(0.1)))
            final_layer = self.gate_network[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.constant_(final_layer.bias, initial_logit)

        self.message_projections = nn.ModuleDict(
            {
                f"{target}_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target, source in self.pair_indices
            }
        )

        if step_dependent or two_level_router:
            self.step_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, head_hidden_dim),
                        nn.GELU(),
                        nn.Linear(head_hidden_dim, 1),
                    )
                    for _ in range(task_count)
                ]
            )
            self.heads = None
        else:
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
            self.step_heads = None

    def _validate_inputs(self, loads: Tensor, exog: Tensor) -> None:
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
            or exog.shape[-1] != self.exog_dim
        ):
            raise ValueError(
                "exog must have shape [batch, lookback, exog_dim] matching loads"
            )

    def encode_tasks(self, loads: Tensor, exog: Tensor) -> Tensor:
        self._validate_inputs(loads, exog)
        outputs = [
            encoder(loads[:, :, i : i + 1], exog)
            for i, encoder in enumerate(self.encoders)
        ]
        return torch.stack(outputs, dim=1)

    def encode_state(self, exog: Tensor) -> Tensor:
        if exog.ndim != 3 or exog.shape[-1] != self.exog_dim:
            raise ValueError("exog must have shape [batch, time, exog_dim]")
        if exog.shape[1] != self.lookback:
            raise ValueError("exog time dimension does not match lookback")
        return self.state_encoder(exog)

    def _legacy_gates(self, state: Tensor, representations: Tensor) -> Tensor:
        """Return [B,T,T] or [B,H,T,T] per-edge sigmoid gates."""

        batch_size = state.shape[0]
        step_embeddings = (
            self.step_embeddings()
            if self.step_dependent and self.step_embeddings is not None
            else None
        )
        if step_embeddings is None:
            gates = state.new_zeros((batch_size, self.task_count, self.task_count))
            for target, source in self.pair_indices:
                pair_input = torch.cat(
                    (
                        state,
                        representations[:, target, :],
                        representations[:, source, :],
                        self.task_embeddings[target].expand(batch_size, -1),
                        self.task_embeddings[source].expand(batch_size, -1),
                    ),
                    dim=-1,
                )
                gates[:, target, source] = torch.sigmoid(
                    self.gate_network(pair_input).squeeze(-1)
                )
            return gates

        horizon = step_embeddings.shape[0]
        gates = state.new_zeros(
            (batch_size, horizon, self.task_count, self.task_count)
        )
        for step_index, step_embedding in enumerate(step_embeddings):
            for target, source in self.pair_indices:
                pair_input = torch.cat(
                    (
                        state,
                        representations[:, target, :],
                        representations[:, source, :],
                        self.task_embeddings[target].expand(batch_size, -1),
                        self.task_embeddings[source].expand(batch_size, -1),
                        step_embedding.expand(batch_size, -1),
                    ),
                    dim=-1,
                )
                gates[:, step_index, target, source] = torch.sigmoid(
                    self.gate_network(pair_input).squeeze(-1)
                )
        return gates

    def _fuse_legacy(self, representations: Tensor, gates: Tensor) -> Tensor:
        if gates.ndim == 3:
            fused = representations.clone()
            for target in range(self.task_count):
                for source in range(self.task_count):
                    if source == target:
                        continue
                    message = self.message_projections[f"{target}_{source}"](
                        representations[:, source, :]
                    )
                    fused[:, target, :] += gates[:, target, source, None] * message
            return fused

        fused = representations[:, None, :, :].expand(
            representations.shape[0], self.horizon, self.task_count, self.hidden_dim
        ).clone()
        for target in range(self.task_count):
            for source in range(self.task_count):
                if source == target:
                    continue
                message = self.message_projections[f"{target}_{source}"](
                    representations[:, source, :]
                )
                fused[:, :, target, :] += (
                    gates[:, :, target, source, None] * message[:, None, :]
                )
        return fused

    def _predict(self, fused: Tensor) -> Tensor:
        if fused.ndim == 3:
            assert self.heads is not None
            predictions = [
                head(fused[:, i, :]) for i, head in enumerate(self.heads)
            ]
            return torch.stack(predictions, dim=-1)

        assert self.step_heads is not None
        predictions = []
        for step_index in range(self.horizon):
            predictions.append(
                torch.stack(
                    [
                        head(fused[:, step_index, task_index, :]).squeeze(-1)
                        for task_index, head in enumerate(self.step_heads)
                    ],
                    dim=-1,
                )
            )
        return torch.stack(predictions, dim=1)

    def forward_with_details(
        self, loads: Tensor, exog: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        representations = self.encode_tasks(loads, exog)
        state = self.encode_state(exog)
        if self.two_level_router:
            assert self.step_embeddings is not None
            assert hasattr(self, "role_embeddings")
            step_embeddings = self.step_embeddings()
            target_embeddings, source_embeddings = self.role_embeddings()
            rho, pi, gates = self.router(
                state,
                representations,
                step_embeddings,
                target_embeddings,
                source_embeddings,
            )
            pair_messages = torch.stack(
                [
                    torch.stack(
                        [
                            self.message_projections[f"{target}_{source}"](
                                representations[:, source, :]
                            )
                            if target != source
                            else torch.zeros_like(representations[:, source, :])
                            for source in range(self.task_count)
                        ],
                        dim=1,
                    )
                    for target in range(self.task_count)
                ],
                dim=1,
            )
            fused = representations[:, None, :, :] + torch.einsum(
                "bhts,btsd->bhtd", gates, pair_messages
            )
            predictions = self._predict(fused)
            details = {
                "representations": representations,
                "state": state,
                "rho": rho,
                "pi": pi,
                "gates": gates,
                "pair_messages": pair_messages,
                "fused_representations": fused,
            }
            return predictions, details

        gates = self._legacy_gates(state, representations)
        fused = self._fuse_legacy(representations, gates)
        predictions = self._predict(fused)
        return predictions, {
            "representations": representations,
            "state": state,
            "gates": gates,
            "fused_representations": fused,
        }

    def forward(self, loads: Tensor, exog: Tensor) -> Tensor:
        return self.forward_with_details(loads, exog)[0]


class A1FullWindowDynamicDirected(_FullWindowLegacyBase):
    """A1: replace only the legacy encoder with full-window DS-TCN."""

    ablation_id = "A1"
    ablation_description = ABLATION_DESCRIPTIONS[ablation_id]

    def __init__(self, exog_dim: int, task_count: int = 4, **kwargs) -> None:
        super().__init__(
            exog_dim=exog_dim,
            task_count=task_count,
            step_dependent=False,
            two_level_router=False,
            **kwargs,
        )


class A2StepDirected(_FullWindowLegacyBase):
    """A2: A1 with one directed sigmoid gate per horizon step."""

    ablation_id = "A2"
    ablation_description = ABLATION_DESCRIPTIONS[ablation_id]

    def __init__(self, exog_dim: int, task_count: int = 4, **kwargs) -> None:
        super().__init__(
            exog_dim=exog_dim,
            task_count=task_count,
            step_dependent=True,
            two_level_router=False,
            **kwargs,
        )


class A3TwoLevelDirected(_FullWindowLegacyBase):
    """A3: A2 with rho-times-pi routing and full-rank unnormalized messages."""

    ablation_id = "A3"
    ablation_description = ABLATION_DESCRIPTIONS[ablation_id]

    def __init__(self, exog_dim: int, task_count: int = 4, **kwargs) -> None:
        super().__init__(
            exog_dim=exog_dim,
            task_count=task_count,
            step_dependent=False,
            two_level_router=True,
            **kwargs,
        )


class A4Scheme2R(Scheme2RModel):
    """A4: the frozen Scheme2R primary model."""

    ablation_id = "A4"
    ablation_description = ABLATION_DESCRIPTIONS[ablation_id]

    def __init__(self, exog_dim: int, task_count: int = 4, **kwargs) -> None:
        super().__init__(exog_dim=exog_dim, task_count=task_count, **kwargs)


_BUILDERS = {
    "A0": A0LegacyDynamicDirectedGate,
    "A1": A1FullWindowDynamicDirected,
    "A2": A2StepDirected,
    "A3": A3TwoLevelDirected,
    "A4": A4Scheme2R,
}


def build_stage7_ablation_model(ablation_id: str, **kwargs) -> nn.Module:
    """Build one frozen A0--A4 interface with common defaults."""

    if ablation_id not in _BUILDERS:
        raise ValueError(
            f"unknown Stage 7 ablation {ablation_id!r}; choose from {ABLATION_NAMES}"
        )
    options = dict(kwargs)
    options.setdefault("task_count", 4)
    options.setdefault("horizon", 4)
    options.setdefault("lookback", 24)
    if ablation_id == "A0":
        # The legacy implementation validates the time dimension only when
        # its encoder receives a batch; it has no ``lookback`` constructor
        # argument.  Keep the common Stage 7 interface while not leaking the
        # new encoder-only option into the legacy class.
        options.pop("lookback", None)
    model = _BUILDERS[ablation_id](**options)
    model.ablation_id = ablation_id
    model.ablation_description = _spec(ablation_id).description
    return model


def ablation_specs_as_dicts() -> Tuple[Dict[str, str], ...]:
    return tuple(spec.as_dict() for spec in ABLATION_SPECS)
