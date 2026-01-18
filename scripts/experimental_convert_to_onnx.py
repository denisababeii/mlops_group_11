"""Convert PyTorch model checkpoint to ONNX format.

This script converts a trained PyTorch model to ONNX (Open Neural Network Exchange)
format for deployment and inference in production environments. ONNX provides
interoperability between different deep learning frameworks and optimization tools.

Usage:
    uv run scripts/experimental_convert_to_onnx.py

Requirements:
    - Trained model checkpoint at 'models/best_model.pth'
    - PyTorch and ONNX dependencies installed

Output:
    - ONNX model saved to 'models/model.onnx'
    - Supports dynamic batch size for flexible inference
"""

from pathlib import Path
from mlops_group_11.model import experimental_convert_model_to_onnx

# Convert the trained model to ONNX format
experimental_convert_model_to_onnx(
    model_name='csatv2_21m.sw_r512_in1k',  # Model architecture from timm library
    checkpoint_path=Path('models/best_model.pth'),  # Path to trained PyTorch checkpoint
    output_path=Path('models/model.onnx'),  # Output path for ONNX model
    num_classes=24  # Number of output classes for multi-label classification
)