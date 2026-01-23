from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
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


def test_set_seed_reproducibility() -> None:
    """
    Test that set_seed() produces reproducible random values.
    """
    # Set seed and generate random tensors
    train_module.set_seed(42)
    random_tensor_1 = torch.randn(3, 3)
    random_array_1 = np.random.randn(3, 3)

    # Set seed again and generate new random tensors
    train_module.set_seed(42)
    random_tensor_2 = torch.randn(3, 3)
    random_array_2 = np.random.randn(3, 3)

    # Verify reproducibility: same seed should produce same random values
    assert torch.allclose(random_tensor_1, random_tensor_2), "PyTorch random seed not reproducible"
    assert np.allclose(random_array_1, random_array_2), "NumPy random seed not reproducible"


def test_upload_to_gcs_handles_exception() -> None:
    """
    Test that upload_to_gcs handles GCS upload failures gracefully.

    """
    with patch("mlops_group_11.train.storage.Client") as mock_storage:
        # Make the client raise an exception
        mock_storage.side_effect = Exception("GCS connection failed")

        # Call upload_to_gcs; it should NOT raise, but log a warning instead
        result = train_module.upload_to_gcs(local_path="/fake/path/model.pth", gcs_path="models/test_model.pth")

        # Should return None and not raise
        assert result is None
