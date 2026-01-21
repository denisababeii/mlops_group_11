from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf

import mlops_group_11.train as train_module


def test_train_returns_gracefully_when_data_missing(monkeypatch, tmp_path: Path) -> None:
    """
    Training test: verify training script behaves sensibly when processed data
    is missing.

    In train.py, poster_dataset(...) is called inside try/except FileNotFoundError,
    and the function returns (does not crash) if data isn't found.

    This test ensures that behavior doesn't accidentally break during refactors.
    """

    # Make poster_dataset raise FileNotFoundError as if processed data isn't there
    def _fake_poster_dataset(_path: Path):
        raise FileNotFoundError("fake missing data")

    monkeypatch.setattr(train_module, "poster_dataset", _fake_poster_dataset)

    # Mock wandb to avoid API key requirement.
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.config = {}
    mock_wandb.init.return_value = mock_run
    mock_wandb.config = {}
    monkeypatch.setattr(train_module, "wandb", mock_wandb)

    # Build a minimal Hydra-like config (only fields train() uses)
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "pretrained": False, "num_classes": 3},
            "data": {"processed_path": str(tmp_path / "does_not_exist")},
            "hyperparameters": {"batch_size": 2, "lr": 1e-3, "epochs": 1, "prob_threshold": 0.5},
            "logging": {"save_frequency": 1},
            "paths": {
                "best_model_file": str(tmp_path / "best.ckpt"),
                "checkpoint_file": str(tmp_path / "checkpoint.ckpt"),
                "reports_file": str(tmp_path / "report.png"),
            },
        }
    )

    # Call the underlying function
    # Expectation: it should NOT raise. It should return early.
    result = train_module.train.__wrapped__(cfg)
    assert result is None, "Expected train() to return None when data is missing (graceful exit)."
