from pathlib import Path

import numpy as np

from src.joint_dispatch.external_baseline_data import ExternalBaselineBatch
from src.joint_dispatch.external_v46_data import ExternalV46Normalization, load_external_v46_split


ROOT = Path(__file__).parents[1]
DATA = ROOT / "reports" / "joint_forecast_dispatch_formal_v4_4" / "formal_v4_4_20260905_f" / "pilot" / "data"


def test_pilot_can_be_normalized_from_train_and_batched_without_labels() -> None:
    train = load_external_v46_split(DATA / "train.npz", "train").take(np.arange(2))
    pilot = load_external_v46_split(DATA / "selection_full.npz", "pilot").take(np.arange(2))
    normalization = ExternalV46Normalization.fit(train)
    normalized = normalization.transform(pilot)
    normalized = type(normalized)(
        **{**normalized.__dict__, "teacher_dispatch": np.zeros((2, 4, 21), dtype=np.float32), "oracle_first_step_objective": np.zeros(2, dtype=np.float32)}
    )
    batch = ExternalBaselineBatch.from_split(normalized)
    assert batch.split == "pilot"
    assert tuple(batch.load_history.shape) == (2, 24, 4)
    assert tuple(batch.teacher_dispatch.shape) == (2, 4, 21)

