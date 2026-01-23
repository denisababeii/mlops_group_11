from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

import mlops_group_11.evaluate as evaluate_module


def test_evaluate_raises_error_when_checkpoint_missing(monkeypatch, tmp_path: Path) -> None:
    """Test that evaluate raises FileNotFoundError when checkpoint file is missing."""

    # Create config pointing to non-existent checkpoint
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "pretrained": False, "num_classes": 3},
            "data": {"processed_path": str(tmp_path / "processed")},
            "hyperparameters": {"batch_size": 2, "prob_threshold": 0.5},
            "paths": {
                "checkpoint_file": str(tmp_path / "nonexistent_checkpoint.pth"),
            },
        }
    )

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        evaluate_module.evaluate.__wrapped__(cfg)


def test_evaluate_raises_error_when_data_missing(monkeypatch, tmp_path: Path) -> None:
    """Test that evaluate raises FileNotFoundError when processed data is missing."""

    # Create a fake checkpoint file
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_text("fake checkpoint")

    # Mock load_model to avoid actually loading a model
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    def _fake_load_model(model_name, checkpoint_path, num_classes, device):
        return mock_model

    monkeypatch.setattr(evaluate_module, "load_model", _fake_load_model)

    # Make poster_dataset raise FileNotFoundError
    def _fake_poster_dataset(_path: Path):
        raise FileNotFoundError("Processed data not found")

    monkeypatch.setattr(evaluate_module, "poster_dataset", _fake_poster_dataset)

    # Create config
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "pretrained": False, "num_classes": 3},
            "data": {"processed_path": str(tmp_path / "does_not_exist")},
            "hyperparameters": {"batch_size": 2, "prob_threshold": 0.5},
            "paths": {
                "checkpoint_file": str(checkpoint_path),
            },
        }
    )

    # Should raise FileNotFoundError from poster_dataset
    with pytest.raises(FileNotFoundError, match="Processed data not found"):
        evaluate_module.evaluate.__wrapped__(cfg)
