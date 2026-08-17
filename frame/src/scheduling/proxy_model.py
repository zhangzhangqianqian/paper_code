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


def build_proxy_model(contract: ProxyContract | None = None, config: ProxyModelConfig | None = None) -> SchedulingProxy:
    if config is None:
        config = ProxyModelConfig.from_contract(contract) if contract is not None else ProxyModelConfig()
    return SchedulingProxy(config)


def save_model_checkpoint(
    path: str | Path,
    model: SchedulingProxy,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    validation_loss: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.to_config(),
        "epoch": int(epoch),
        "validation_loss": None if validation_loss is None else float(validation_loss),
        "metadata": dict(metadata or {}),
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
) -> tuple[SchedulingProxy, Mapping[str, Any]]:
    if str(device) != "cpu" and torch.device(device).type != "cpu":
        raise ValueError("scheduling proxy is CPU-only")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload or "model_config" not in payload:
        raise ValueError("invalid scheduling proxy checkpoint")
    config = ProxyModelConfig(**dict(payload["model_config"]))
    if expected_contract is not None:
        if config != ProxyModelConfig.from_contract(expected_contract):
            raise ValueError("checkpoint model configuration does not match contract")
        metadata = payload.get("metadata", {})
        stored_contract_hash = metadata.get("contract_sha256") if isinstance(metadata, Mapping) else None
        expected_hash = contract_sha256(expected_contract)
        if stored_contract_hash is None or str(stored_contract_hash).lower() != expected_hash.lower():
            raise ValueError("checkpoint contract SHA-256 mismatch")
    model = SchedulingProxy(config)
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
    "ResidualMLPBlock",
    "SchedulingProxy",
    "ProxyModel",
    "SchedulingProxyModel",
    "build_proxy_model",
    "save_model_checkpoint",
    "load_model_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
