"""Convert PyTorch model to ONNX format.

This script converts a trained PyTorch model to ONNX using Hydra configuration.

Usage:
    uv run scripts/convert_to_onnx.py

Requirements:
    - Trained model checkpoint at path specified in config

Output:
    - ONNX model saved to 'models/model.onnx'
"""

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mlops_group_11.model import experimental_convert_model_to_onnx


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs", version_base=None)
def main(cfg: DictConfig) -> None:
    """Convert trained model to ONNX format using Hydra config."""
    experimental_convert_model_to_onnx(
        model_name=cfg.model.name,
        checkpoint_path=Path(cfg.paths.best_model_file),
        output_path=Path('models/model.onnx'),
        num_classes=cfg.model.num_classes
    )


if __name__ == "__main__":
    main()