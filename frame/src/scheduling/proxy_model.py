"""CPU-friendly horizon-aware residual MLP scheduling proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from .proxy_contract import FEATURE_ORDER, HORIZON, LABEL_ORDER, ProxyContract, contract_sha256
from .proxy_decoder import CONTROL_DIM, CONTROL_TEMPERATURE, DECISION_GROUPS, DECODER_SCHEMA_VERSION, decode_feasible_dispatch


@dataclass(frozen=True)
class ProxyModelConfig:
    input_dim: int = len(FEATURE_ORDER)
    horizon: int = HORIZON
    output_dim: int = len(LABEL_ORDER)
    hidden_width: int = 128
    residual_blocks: int = 2
    dropout: float = 0.1

    @classmethod
    def from_contract(cls, contract: ProxyContract) -> "ProxyModelConfig":
        return cls(
            input_dim=contract.input_dim,
            horizon=contract.horizon,
            output_dim=contract.output_dim,
            hidden_width=int(contract.model["hidden_width"]),
            residual_blocks=int(contract.model["residual_blocks"]),
            dropout=float(contract.model["dropout"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class FeasibleSchedulingProxyConfig:
    """Configuration for the v2 15-control residual MLP."""

    input_dim: int = len(FEATURE_ORDER)
    horizon: int = HORIZON
    decision_dim: int = CONTROL_DIM
    hidden_width: int = 128
    residual_blocks: int = 2
    dropout: float = 0.1
    decoder_schema_version: str = DECODER_SCHEMA_VERSION
    decoder_dtype: str = "float64"
    control_temperature: float = CONTROL_TEMPERATURE

    @classmethod
    def from_contract(cls, contract: ProxyContract) -> "FeasibleSchedulingProxyConfig":
        if not contract.is_v2:
            raise ValueError("FeasibleSchedulingProxyConfig requires a v2 contract")
        return cls(
            input_dim=contract.input_dim,
            horizon=contract.horizon,
            decision_dim=contract.decision_dim,
            hidden_width=int(contract.model["hidden_width"]),
            residual_blocks=int(contract.model["residual_blocks"]),
            dropout=float(contract.model["dropout"]),
            decoder_schema_version=str(contract.model["decoder_schema_version"]),
            decoder_dtype=str(contract.model["decoder_dtype"]),
            control_temperature=float(contract.model["control_temperature"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_dim(self) -> int:
        """Compatibility spelling for callers that inspect model dimensions."""

        return self.decision_dim


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.norm(x + residual)


class SchedulingProxy(nn.Module):
    """Map normalized ``[B,4,10]`` features to normalized ``[B,4,21]`` labels."""

    def __init__(self, config: ProxyModelConfig | None = None):
        super().__init__()
        self.config = config or ProxyModelConfig()
        if self.config.horizon != HORIZON:
            raise ValueError("scheduling proxy horizon is fixed at 4")
        if self.config.input_dim != len(FEATURE_ORDER) or self.config.output_dim != len(LABEL_ORDER):
            raise ValueError("proxy dimensions must be input=10 and output=21")
        if self.config.hidden_width <= 0 or self.config.residual_blocks <= 0:
            raise ValueError("hidden_width and residual_blocks must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        flat_in = self.config.horizon * self.config.input_dim
        flat_out = self.config.horizon * self.config.output_dim
        self.input_block = nn.Sequential(
            nn.Linear(flat_in, self.config.hidden_width),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_width),
            nn.Dropout(self.config.dropout),
        )
        self.residual = nn.Sequential(*[
            ResidualMLPBlock(self.config.hidden_width, self.config.dropout)
            for _ in range(self.config.residual_blocks)
        ])
        self.output_projection = nn.Linear(self.config.hidden_width, flat_out)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (self.config.horizon, self.config.input_dim):
            raise ValueError(f"proxy inputs must have shape [B,{self.config.horizon},{self.config.input_dim}]")
        if not torch.isfinite(inputs).all():
            raise ValueError("proxy inputs must be finite")
        x = inputs.reshape(inputs.shape[0], -1)
        x = self.input_block(x)
        x = self.residual(x)
        x = torch.sigmoid(self.output_projection(x))
        return x.reshape(inputs.shape[0], self.config.horizon, self.config.output_dim)

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def to_config(self) -> dict[str, Any]:
        return self.config.to_dict()


class FeasibleSchedulingProxy(nn.Module):
    """Residual MLP emitting v2 control logits for the physical decoder."""

    def __init__(self, config: FeasibleSchedulingProxyConfig | None = None):
        super().__init__()
        self.config = config or FeasibleSchedulingProxyConfig()
        if self.config.horizon != HORIZON or self.config.input_dim <= 0:
            raise ValueError("v2 proxy dimensions must have a positive input_dim and horizon=4")
        if self.config.decision_dim != CONTROL_DIM:
            raise ValueError("v2 proxy decision dimension must be 15")
        if self.config.hidden_width <= 0 or self.config.residual_blocks <= 0:
            raise ValueError("hidden_width and residual_blocks must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.config.decoder_dtype != "float64" or self.config.decoder_schema_version != DECODER_SCHEMA_VERSION:
            raise ValueError("v2 proxy decoder metadata is invalid")
        if float(self.config.control_temperature) != CONTROL_TEMPERATURE:
            raise ValueError("v2 proxy control temperature must be 0.25")
        flat_in = self.config.horizon * self.config.input_dim
        self.input_block = nn.Sequential(
            nn.Linear(flat_in, self.config.hidden_width),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_width),
            nn.Dropout(self.config.dropout),
        )
        self.residual = nn.Sequential(*[
            ResidualMLPBlock(self.config.hidden_width, self.config.dropout)
            for _ in range(self.config.residual_blocks)
        ])
        self.output_projection = nn.Linear(self.config.hidden_width, self.config.decision_dim)

    def _check_inputs(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (self.config.horizon, self.config.input_dim):
            raise ValueError(f"proxy inputs must have shape [B,{self.config.horizon},{self.config.input_dim}]")
        if not torch.isfinite(inputs).all():
            raise ValueError("proxy inputs must be finite")
        return inputs

    def forward_logits(self, normalized_inputs: Tensor) -> Tensor:
        inputs = self._check_inputs(normalized_inputs)
        x = inputs.reshape(inputs.shape[0], -1)
        x = self.input_block(x)
        x = self.residual(x)
        return self.output_projection(x)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_logits(inputs)

    def predict_dispatch(
        self,
        normalized_inputs: Tensor,
        physical_inputs: Tensor,
        parameters: Mapping[str, Any],
    ) -> Tensor:
        logits = self.forward_logits(normalized_inputs)
        return decode_feasible_dispatch(
            logits,
            physical_inputs,
            parameters,
            temperature=float(self.config.control_temperature),
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def to_config(self) -> dict[str, Any]:
        return self.config.to_dict()


def build_proxy_model(
    contract: ProxyContract | None = None,
    config: ProxyModelConfig | FeasibleSchedulingProxyConfig | None = None,
) -> SchedulingProxy | FeasibleSchedulingProxy:
    if config is None:
        if contract is not None and contract.is_v2:
            config = FeasibleSchedulingProxyConfig.from_contract(contract)
        else:
            config = ProxyModelConfig.from_contract(contract) if contract is not None else ProxyModelConfig()
    if isinstance(config, FeasibleSchedulingProxyConfig):
        return FeasibleSchedulingProxy(config)
    return SchedulingProxy(config)


def save_model_checkpoint(
    path: str | Path,
    model: SchedulingProxy | FeasibleSchedulingProxy,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    validation_loss: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata or {})
    if isinstance(model, FeasibleSchedulingProxy):
        checkpoint_metadata.setdefault("model_family", "feasible_scheduling_proxy_v2")
        checkpoint_metadata.setdefault("decoder_schema_version", DECODER_SCHEMA_VERSION)
        checkpoint_metadata.setdefault("decoder_dtype", "float64")
        checkpoint_metadata.setdefault("control_temperature", CONTROL_TEMPERATURE)
        checkpoint_metadata.setdefault("decision_dim", CONTROL_DIM)
        checkpoint_metadata.setdefault("decision_groups", {name: list(bounds) for name, bounds in DECISION_GROUPS.items()})
        checkpoint_metadata.setdefault("inference_exact_lp_calls", 0)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.to_config(),
        "epoch": int(epoch),
        "validation_loss": None if validation_loss is None else float(validation_loss),
        "metadata": checkpoint_metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    # Write beside the destination and replace atomically so a killed smoke or
    # resumed training run cannot leave a partially serialized checkpoint.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, str(temporary))
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def load_model_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    expected_contract: ProxyContract | None = None,
) -> tuple[SchedulingProxy | FeasibleSchedulingProxy, Mapping[str, Any]]:
    if str(device) != "cpu" and torch.device(device).type != "cpu":
        raise ValueError("scheduling proxy is CPU-only")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload or "model_config" not in payload:
        raise ValueError("invalid scheduling proxy checkpoint")
    raw_config = payload["model_config"]
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint model_config must be an object")
    checkpoint_metadata = payload.get("metadata", {})
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("checkpoint metadata is invalid")
    is_v2 = int(raw_config.get("decision_dim", 0)) == CONTROL_DIM or checkpoint_metadata.get("model_family") == "feasible_scheduling_proxy_v2"
    if is_v2:
        try:
            config = FeasibleSchedulingProxyConfig(**dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid v2 scheduling proxy model configuration") from exc
        if checkpoint_metadata.get("model_family") != "feasible_scheduling_proxy_v2":
            raise ValueError("v2 checkpoint model family is missing or mismatched")
        if str(checkpoint_metadata.get("decoder_schema_version", "")) != DECODER_SCHEMA_VERSION:
            raise ValueError("v2 checkpoint decoder schema is missing or mismatched")
        if str(checkpoint_metadata.get("decoder_dtype", "")) != "float64":
            raise ValueError("v2 checkpoint decoder dtype is missing or mismatched")
        try:
            control_temperature = float(checkpoint_metadata.get("control_temperature"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("v2 checkpoint control temperature is missing or mismatched") from exc
        if control_temperature != CONTROL_TEMPERATURE:
            raise ValueError("v2 checkpoint control temperature is missing or mismatched")
        try:
            decision_dim = int(checkpoint_metadata.get("decision_dim"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("v2 checkpoint decision dimension is missing or mismatched") from exc
        if decision_dim != CONTROL_DIM:
            raise ValueError("v2 checkpoint decision dimension is missing or mismatched")
        try:
            lp_calls = int(checkpoint_metadata.get("inference_exact_lp_calls"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("v2 checkpoint inference_exact_lp_calls must be zero") from exc
        if lp_calls != 0:
            raise ValueError("v2 checkpoint inference_exact_lp_calls must be zero")
        expected_groups = {"cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15]}
        if checkpoint_metadata.get("decision_groups") != expected_groups:
            raise ValueError("v2 checkpoint decision groups are missing or mismatched")
        model: SchedulingProxy | FeasibleSchedulingProxy = FeasibleSchedulingProxy(config)
    else:
        try:
            config = ProxyModelConfig(**dict(raw_config))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid scheduling proxy model configuration") from exc
        model = SchedulingProxy(config)
    if expected_contract is not None:
        expected_config: ProxyModelConfig | FeasibleSchedulingProxyConfig
        expected_config = FeasibleSchedulingProxyConfig.from_contract(expected_contract) if expected_contract.is_v2 else ProxyModelConfig.from_contract(expected_contract)
        if config != expected_config:
            raise ValueError("checkpoint model configuration does not match contract")
        if expected_contract.is_v2 != is_v2:
            raise ValueError("checkpoint model family does not match contract")
        metadata = payload.get("metadata", {})
        stored_contract_hash = metadata.get("contract_sha256") if isinstance(metadata, Mapping) else None
        expected_hash = contract_sha256(expected_contract)
        if stored_contract_hash is None or str(stored_contract_hash).lower() != expected_hash.lower():
            raise ValueError("checkpoint contract SHA-256 mismatch")
    model.load_state_dict(payload["model_state_dict"])
    model.to("cpu")
    model.eval()
    return model, payload


# API aliases.
ProxyModel = SchedulingProxy
SchedulingProxyModel = SchedulingProxy
load_checkpoint = load_model_checkpoint
save_checkpoint = save_model_checkpoint


__all__ = [
    "ProxyModelConfig",
    "FeasibleSchedulingProxyConfig",
    "ResidualMLPBlock",
    "SchedulingProxy",
    "FeasibleSchedulingProxy",
    "ProxyModel",
    "SchedulingProxyModel",
    "build_proxy_model",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
