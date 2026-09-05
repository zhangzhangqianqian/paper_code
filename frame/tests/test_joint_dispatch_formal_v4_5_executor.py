from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.joint_dispatch.formal_v4_5_pilot_data import build_v45_loaders


def test_executor_exposes_disjoint_train_and_early_stop_loaders(monkeypatch) -> None:
    def fake_batches(collection, _normalization, indices, batch_size, teacher_dispatch=None):
        assert batch_size == 4
        result = []
        for start in range(0, len(indices), 2):
            stop = min(start + 2, len(indices))
            item = {"sample_indices": indices[start:stop]}
            if teacher_dispatch is not None:
                item["teacher_dispatch"] = teacher_dispatch[start:stop]
            result.append(item)
        return result

    monkeypatch.setattr("src.joint_dispatch.formal_v4_5_pilot_data.build_v44_batches", fake_batches)
    materialized = SimpleNamespace(train=SimpleNamespace(__len__=lambda self: 3), early_stop=SimpleNamespace(__len__=lambda self: 2))
    # Python special methods are looked up on the type, so use concrete arrays
    # for the fixture collections instead of relying on instance attributes.
    materialized.train = np.zeros((3, 1))
    materialized.early_stop = np.zeros((2, 1))
    teachers = {
        "train": np.zeros((3, 4, 21)),
        "early_stop": np.zeros((2, 4, 21)),
    }
    loaders = build_v45_loaders(materialized, normalization=None, batch_size=4, teacher=teachers)
    assert set(loaders) == {"train", "early_stop"}
    train_origins = {int(value) for batch in loaders["train"] for value in batch["origin_index"]}
    validation_origins = {int(value) for batch in loaders["early_stop"] for value in batch["origin_index"]}
    assert train_origins.isdisjoint(validation_origins)
    assert validation_origins == {3, 4}
