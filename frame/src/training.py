"""阶段 2/3：训练集专属标准化、窗口数据集和 CPU 训练接口。"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .data_pipeline import TASKS
from .fairness_contract import INPUT_MODES


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0).astype(np.float32)
    return np.where(scale < 1e-8, 1.0, scale).astype(np.float32)


@dataclass
class StandardizationStats:
    """只使用训练集拟合的负荷和外生变量标准化参数。"""

    load_mean: np.ndarray
    load_scale: np.ndarray
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    exog_columns: Tuple[str, ...]
    task_columns: Tuple[str, ...] = TASKS

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        exog_columns: Sequence[str] = (),
        task_columns: Sequence[str] = TASKS,
    ) -> "StandardizationStats":
        task_columns = tuple(str(column) for column in task_columns)
        if not task_columns:
            raise ValueError("task_columns不能为空")
        required = [*task_columns, *exog_columns]
        missing = [column for column in required if column not in train_frame.columns]
        if missing:
            raise ValueError(f"标准化拟合缺少字段：{missing}")
        load_values = train_frame[list(task_columns)].to_numpy(dtype=np.float32)
        exog_values = train_frame[list(exog_columns)].to_numpy(dtype=np.float32)
        if not np.isfinite(load_values).all() or not np.isfinite(exog_values).all():
            raise ValueError("训练集标准化拟合不能包含 NaN 或 Inf")
        return cls(
            load_mean=load_values.mean(axis=0).astype(np.float32),
            load_scale=_safe_scale(load_values),
            exog_mean=exog_values.mean(axis=0).astype(np.float32),
            exog_scale=_safe_scale(exog_values),
            exog_columns=tuple(exog_columns),
            task_columns=task_columns,
        )

    def transform_windows(self, windows: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """标准化窗口中的 loads、exog 和 target，不修改原始数组。"""

        loads = np.asarray(windows["loads"], dtype=np.float32)
        exog = np.asarray(windows["exog"], dtype=np.float32)
        target = np.asarray(windows["target"], dtype=np.float32)
        if loads.ndim != 3 or loads.shape[-1] != len(self.task_columns):
            raise ValueError(
                f"loads必须为[样本,时间,{len(self.task_columns)}]"
            )
        if exog.ndim != 3 or exog.shape[-1] != len(self.exog_columns):
            raise ValueError("exog维度与标准化参数不一致")
        if target.ndim != 3 or target.shape[-1] != len(self.task_columns):
            raise ValueError(
                f"target必须为[样本,预测步,{len(self.task_columns)}]"
            )
        return {
            "loads": ((loads - self.load_mean) / self.load_scale).astype(np.float32),
            "exog": ((exog - self.exog_mean) / self.exog_scale).astype(np.float32),
            "target": ((target - self.load_mean) / self.load_scale).astype(np.float32),
            "target_times": np.asarray(windows["target_times"]),
        }

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        targets = np.asarray(values, dtype=np.float32)
        if targets.ndim != 3 or targets.shape[-1] != len(self.task_columns):
            raise ValueError(
                f"预测目标必须为[样本,预测步,{len(self.task_columns)}]"
            )
        return (targets * self.load_scale + self.load_mean).astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """将规范 DataFrame 转为标准化 DataFrame，便于检查训练统计量。"""

        working = frame.copy()
        working.loc[:, list(self.task_columns)] = (
            working[list(self.task_columns)] - self.load_mean
        ) / self.load_scale
        if self.exog_columns:
            working.loc[:, list(self.exog_columns)] = (
                working[list(self.exog_columns)] - self.exog_mean
            ) / self.exog_scale
        return working

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            load_mean=self.load_mean,
            load_scale=self.load_scale,
            exog_mean=self.exog_mean,
            exog_scale=self.exog_scale,
            exog_columns=np.asarray(self.exog_columns, dtype="U"),
            task_columns=np.asarray(self.task_columns, dtype="U"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StandardizationStats":
        with np.load(path, allow_pickle=False) as values:
            task_columns = (
                tuple(str(value) for value in values["task_columns"])
                if "task_columns" in values.files
                else TASKS
            )
            return cls(
                load_mean=values["load_mean"].astype(np.float32),
                load_scale=values["load_scale"].astype(np.float32),
                exog_mean=values["exog_mean"].astype(np.float32),
                exog_scale=values["exog_scale"].astype(np.float32),
                exog_columns=tuple(str(value) for value in values["exog_columns"]),
                task_columns=task_columns,
            )

    def summary(self) -> Dict[str, object]:
        return {
            "tasks": list(self.task_columns),
            "exog_columns": list(self.exog_columns),
            "load_mean": self.load_mean.tolist(),
            "load_scale": self.load_scale.tolist(),
            "exog_mean": self.exog_mean.tolist(),
            "exog_scale": self.exog_scale.tolist(),
        }


class WindowDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """将标准化窗口转换为 PyTorch 张量。"""

    def __init__(self, windows: Mapping[str, np.ndarray]) -> None:
        self.loads = torch.from_numpy(np.asarray(windows["loads"], dtype=np.float32))
        self.exog = torch.from_numpy(np.asarray(windows["exog"], dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(windows["target"], dtype=np.float32))
        if not (len(self.loads) == len(self.exog) == len(self.target)):
            raise ValueError("loads、exog和target样本数必须一致")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.loads[index], self.exog[index], self.target[index]


def make_dataloader(
    windows: Mapping[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size必须为正整数")
    if seed is not None and seed < 0:
        raise ValueError("DataLoader seed must be non-negative")
    dataset = WindowDataset(windows)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


@dataclass(frozen=True)
class TrainerConfig:
    device: str = "cpu"
    torch_threads: int = 8
    seed: int = 2026
    deterministic_algorithms: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    max_epochs: int = 100
    early_stopping_patience: int = 12


def set_reproducible(config: TrainerConfig) -> torch.device:
    if config.torch_threads <= 0:
        raise ValueError("torch_threads必须为正整数")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config.torch_threads)
    return device


def _forward_model(
    model: nn.Module,
    loads: Tensor,
    exog: Tensor,
    input_mode: str,
) -> Tensor:
    if input_mode == "loads_only":
        return model(loads)
    if input_mode == "loads_and_exog":
        return model(loads, exog)
    raise ValueError(f"未知输入模式{input_mode!r}；可选模式为{INPUT_MODES}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 1.0,
    input_mode: str = "loads_and_exog",
) -> float:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"未知输入模式{input_mode!r}；可选模式为{INPUT_MODES}")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for loads, exog, target in loader:
            loads = loads.to(device)
            target = target.to(device)
            if input_mode == "loads_and_exog":
                exog = exog.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = _forward_model(model, loads, exog, input_mode)
            loss = loss_function(prediction, target)
            if training:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            batch_size = loads.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_count += batch_size
    if total_count == 0:
        raise ValueError("训练或验证 DataLoader 为空")
    return total_loss / total_count


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    input_mode: str = "loads_and_exog",
) -> Tuple[float, np.ndarray, np.ndarray]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"未知输入模式{input_mode!r}；可选模式为{INPUT_MODES}")
    device = torch.device(device)
    loss_function = nn.SmoothL1Loss()
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for loads, exog, target in loader:
            loads = loads.to(device)
            if input_mode == "loads_and_exog":
                exog = exog.to(device)
            prediction = _forward_model(
                model, loads, exog, input_mode
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(target.numpy())
    if not predictions:
        raise ValueError("评估 DataLoader 为空")
    prediction_array = np.concatenate(predictions, axis=0).astype(np.float32)
    target_array = np.concatenate(targets, axis=0).astype(np.float32)
    loss = float(loss_function(torch.from_numpy(prediction_array), torch.from_numpy(target_array)))
    return loss, prediction_array, target_array


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainerConfig,
    checkpoint_path: str | Path,
    input_mode: str = "loads_and_exog",
) -> List[Dict[str, float]]:
    if config.max_epochs <= 0 or config.early_stopping_patience < 0:
        raise ValueError("训练轮数必须为正数，early stopping patience不能为负数")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"未知输入模式{input_mode!r}；可选模式为{INPUT_MODES}")
    device = set_reproducible(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            input_mode=input_mode,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
            input_mode=input_mode,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        if validation_loss < best_validation:
            best_validation = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_loss": best_validation,
                    "trainer_config": asdict(config),
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    load_checkpoint(model, checkpoint, device)
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint
