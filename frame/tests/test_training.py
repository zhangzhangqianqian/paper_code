import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_pipeline import HEEW_EXOG_COLUMNS, TASKS, build_windows
from src.external_models import DLinearBaseline, MMoELiteBaseline, SOFTSBaseline
from src.models import HardShareMTLModel
from src.training import (
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
)


def synthetic_frame(rows: int = 96) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=rows, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps})
    for task_index, task in enumerate(TASKS):
        frame[task] = 10.0 + task_index + np.arange(rows, dtype=np.float32) * 0.1
    for index, column in enumerate(HEEW_EXOG_COLUMNS):
        frame[column] = np.sin(np.arange(rows, dtype=np.float32) / (index + 2))
    return frame


class TrainingTest(unittest.TestCase):
    def test_train_only_standardization_and_inverse(self):
        frame = synthetic_frame()
        stats = StandardizationStats.fit(frame.iloc[:48], HEEW_EXOG_COLUMNS)
        windows = build_windows(
            frame.iloc[:48], lookback=24, horizon=4, exog_columns=HEEW_EXOG_COLUMNS
        )
        standardized = stats.transform_windows(windows)
        recovered = stats.inverse_targets(standardized["target"])
        np.testing.assert_allclose(recovered, windows["target"], rtol=1e-5, atol=1e-5)
        self.assertEqual(standardized["loads"].shape[-1], 3)
        self.assertEqual(standardized["exog"].shape[-1], len(HEEW_EXOG_COLUMNS))

    def test_standardization_stats_round_trip(self):
        frame = synthetic_frame()
        stats = StandardizationStats.fit(frame, HEEW_EXOG_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.npz"
            stats.save(path)
            restored = StandardizationStats.load(path)
        np.testing.assert_allclose(restored.load_mean, stats.load_mean)
        np.testing.assert_allclose(restored.exog_scale, stats.exog_scale)
        self.assertEqual(restored.exog_columns, stats.exog_columns)

    def test_cpu_fit_validation_checkpoint_and_evaluation(self):
        frame = synthetic_frame(128)
        stats = StandardizationStats.fit(frame.iloc[:80], HEEW_EXOG_COLUMNS)
        train = stats.transform_windows(
            build_windows(frame.iloc[:80], 24, 4, HEEW_EXOG_COLUMNS)
        )
        validation = stats.transform_windows(
            build_windows(frame.iloc[48:112], 24, 4, HEEW_EXOG_COLUMNS)
        )
        model = HardShareMTLModel(exog_dim=len(HEEW_EXOG_COLUMNS), hidden_dim=8, dropout=0.0)
        train_loader = make_dataloader(train, batch_size=8, shuffle=True)
        validation_loader = make_dataloader(validation, batch_size=8, shuffle=False)
        config = TrainerConfig(
            torch_threads=1,
            max_epochs=2,
            early_stopping_patience=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            history = fit_model(model, train_loader, validation_loader, config, checkpoint)
            loss, prediction, target = evaluate_model(model, validation_loader, "cpu")
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, target.shape)

    def test_loads_only_training_mode_supports_dlinear(self):
        frame = synthetic_frame(128)
        stats = StandardizationStats.fit(frame.iloc[:80], HEEW_EXOG_COLUMNS)
        train = stats.transform_windows(
            build_windows(frame.iloc[:80], 24, 4, HEEW_EXOG_COLUMNS)
        )
        validation = stats.transform_windows(
            build_windows(frame.iloc[48:112], 24, 4, HEEW_EXOG_COLUMNS)
        )
        model = DLinearBaseline(lookback=24, horizon=4, task_count=3)
        train_loader = make_dataloader(train, batch_size=8, shuffle=True)
        validation_loader = make_dataloader(validation, batch_size=8, shuffle=False)
        config = TrainerConfig(
            torch_threads=1,
            max_epochs=2,
            early_stopping_patience=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            history = fit_model(
                model,
                train_loader,
                validation_loader,
                config,
                checkpoint,
                input_mode="loads_only",
            )
            loss, prediction, target = evaluate_model(
                model,
                validation_loader,
                "cpu",
                input_mode="loads_only",
            )
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, target.shape)

    def test_loads_and_exog_training_mode_supports_mmoe_lite(self):
        frame = synthetic_frame(128)
        stats = StandardizationStats.fit(frame.iloc[:80], HEEW_EXOG_COLUMNS)
        train = stats.transform_windows(
            build_windows(frame.iloc[:80], 24, 4, HEEW_EXOG_COLUMNS)
        )
        validation = stats.transform_windows(
            build_windows(frame.iloc[48:112], 24, 4, HEEW_EXOG_COLUMNS)
        )
        model = MMoELiteBaseline(
            lookback=24,
            horizon=4,
            task_count=3,
            exog_dim=len(HEEW_EXOG_COLUMNS),
            expert_hidden_dim=8,
            representation_dim=8,
            head_hidden_dim=4,
            dropout=0.0,
        )
        train_loader = make_dataloader(train, batch_size=8, shuffle=True)
        validation_loader = make_dataloader(validation, batch_size=8, shuffle=False)
        config = TrainerConfig(
            torch_threads=1,
            max_epochs=2,
            early_stopping_patience=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            history = fit_model(
                model,
                train_loader,
                validation_loader,
                config,
                checkpoint,
                input_mode="loads_and_exog",
            )
            loss, prediction, target = evaluate_model(
                model,
                validation_loader,
                "cpu",
                input_mode="loads_and_exog",
            )
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, target.shape)

    def test_loads_only_training_mode_supports_softs(self):
        frame = synthetic_frame(128)
        stats = StandardizationStats.fit(frame.iloc[:80], HEEW_EXOG_COLUMNS)
        train = stats.transform_windows(
            build_windows(frame.iloc[:80], 24, 4, HEEW_EXOG_COLUMNS)
        )
        validation = stats.transform_windows(
            build_windows(frame.iloc[48:112], 24, 4, HEEW_EXOG_COLUMNS)
        )
        model = SOFTSBaseline(
            lookback=24,
            horizon=4,
            task_count=3,
            d_model=8,
            d_core=4,
            d_ff=16,
            e_layers=1,
            dropout=0.0,
            stochastic_pooling=False,
        )
        train_loader = make_dataloader(train, batch_size=8, shuffle=True)
        validation_loader = make_dataloader(validation, batch_size=8, shuffle=False)
        config = TrainerConfig(
            torch_threads=1,
            max_epochs=2,
            early_stopping_patience=1,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            history = fit_model(
                model,
                train_loader,
                validation_loader,
                config,
                checkpoint,
                input_mode="loads_only",
            )
            loss, prediction, target = evaluate_model(
                model,
                validation_loader,
                "cpu",
                input_mode="loads_only",
            )
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(len(history), 1)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(prediction.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
