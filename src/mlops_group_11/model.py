"""Model architecture Module (timm library)

This module provide functions to create and manage deep learning models for
multi-label movie poster classification using the `timm` library.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import timm
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_timm_model(
    name: str = "csatv2_21m.sw_r512_in1k", pretrained: bool = True, num_classes: int = 24, **kwargs: Any
) -> nn.Module:
    """Create and return a timm model for multi-label classification.

    Args:
        name: model identifier for `timm.create_model`.
        pretrained: whether to load pretrained ImageNet weights (default: True).
        num_classes: number of output classes for the model (default: 24).
        **kwargs: additional keyword arguments for `timm.create_model`.

    Returns:
        A `torch.nn.Module` model instance configured for multi-label classification.

    Raises:
        RunTimeError: if model creation fails.
        ValueError: if num_classes is not a positive integer.

    """

    # Validate inputs
    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")
    try:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)

        # Log model statistics
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Created model '{name}' with {num_params} parameters, of which {num_trainable} are trainable.")
        logger.info(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")

        return model

    except Exception as e:
        logger.error(f"Error creating model '{name}': {e}")
        raise RuntimeError(f"Model creation failed for '{name}'") from e


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about the model architecture.

    Args:
        model: A `torch.nn.Module` model to analyze.

    Returns:
        A dictionary containing model statistics.

    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Get model size (MB)
    param_size_mb = total_params * 4 / 1024 / 1024  # float32

    # Get layer count
    num_layers = len(list(model.modules()))

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "model_size_mb": param_size_mb,
        "num_layers": num_layers,
    }


def list_available_models(filter_str: str = "", pretrained_only: bool = True) -> List[str]:
    """List available timm models.

    Args:
        filter_str: Optional substring to filter model names (default: "").
        pretrained_only: Whether to only list models with pretrained weights (deault: True).

    Returns:
        A list of available model names.
    """
    models = timm.list_models(filter_str, pretrained=pretrained_only)
    logger.info(f"Found {len(models)} available models matching filter '{filter_str}'")
    return models


def save_model(model: nn.Module, save_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save the model state dictionary to a file.

    Args:
        model: The `torch.nn.Module` model to save.
        save_path: Path to save the model file.
        metadata: Optional dictionary of metadata to save with the model.

    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {"model_state_dict": model.state_dict(), "model_info": get_model_info(model)}

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(
    model_name: str,
    checkpoint_path: Path,
    num_classes: int = 24,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a model from a checkpoint file.

    Args:
        model_name: Name of the timm model to load.
        checkpoint_path: Path to the saved checkpoint file.
        num_classes: Number of output classes for the model.
        device: Device to load the model on (default: auto-detect).

    Returns:
        A `torch.nn.Module` model instance with loaded weights.

    Raises:
        FileNotFoundError: if checkpoint file does not exist.

    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model architecture
    model = create_timm_model(
        name=model_name,
        pretrained=False,
        num_classes=num_classes,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "metadata" in checkpoint:
            logger.info(f"Loaded metadata: {checkpoint['metadata']}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    logger.info(f"Model loaded from {checkpoint_path} onto {device}")

    return model


def freeze_layers(model: nn.Module, freeze_until: Optional[int] = None) -> nn.Module:
    """Freeze layers of the model for transfer learning.

    Args:
        model: The `torch.nn.Module` model to freeze layers in.
        freeze_until: Name of layer to freeze until (default: None, freeze all).

    Returns:
        Model with frozen layers.

    """
    if freeze_until is None:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        logger.info("All layers frozen.")
    else:
        # Freeze layers until specified layer
        freeze = True
        for name, param in model.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

    # Count froze/trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    logger.info(f"Frozen {frozen:,} parameters; {trainable:,} remain trainable.")

    return model


def validate_model_output(model: nn.Module, input_shape: tuple = (1, 3, 224, 224), num_classes: int = 24) -> bool:
    """Validate that model produces expected output shape.

    Args:
        model: The `torch.nn.Module` model to validate.
        input_shape: Shape of the dummy input tensor (B, C, H, W).
        num_classes: Expected number of output classes.

    Returns:
        True if output shape is valid, False otherwise.

    Raises:
        AssertionError: if validation fails.

    """
    model.eval()
    with torch.no_grad():
        x = torch.rand(input_shape)
        out = model(x)

    # Check output shape
    expected_shape = (input_shape[0], num_classes)
    assert out.shape == expected_shape, f"Output shape {out.shape} does not match expected {expected_shape}"

    # Check for NaNs/Infs
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert not torch.isinf(out).any(), "Output contains Infs"

    # Check dtype
    assert out.dtype == torch.float32, f"Output dtype {out.dtype} is not float32"

    logger.info("Model output validation passed.")
    logger.info(f" Input shape: {input_shape}")
    logger.info(f" Output shape: {out.shape}")

    return True


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters in the model grouped by layer/module.

    Args:
        model: The `torch.nn.Module` model to analyze.

    Returns:
        A dictionary mapping layer/module names to parameter counts.

    """
    layer_params = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layer_params[name] = num_params

    return layer_params

def experimental_convert_model_to_onnx(
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    num_classes: int = 24,
    input_shape: tuple = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> None:
    """Convert a trained model to ONNX format.
    
    Args:
        model_name: Name of the timm model.
        checkpoint_path: Path to the model checkpoint.
        output_path: Path where ONNX model will be saved.
        num_classes: Number of output classes (default: 24).
        input_shape: Shape of dummy input tensor (B, C, H, W) (default: (1, 3, 224, 224)).
        device: Device to use for conversion (default: auto-detect).
    """
    logger.info("Starting ONNX conversion")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(
        model_name=model_name, checkpoint_path=checkpoint_path, num_classes=num_classes, device=device
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Create dummy input for tracing
    dummy_input = torch.randn(input_shape).to(device)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    
    logger.info(f"Model successfully exported to ONNX format: {output_path}")

if __name__ == "__main__":
    """ Example usage and quick validation """
    import sys

    print("Validating model module")

    try:
        # Create and validate model
        model = create_timm_model()
        info = get_model_info(model)

        print(f"Model: {info['total_params']:,} params, {info['model_size_mb']:.1f} MB")

        # Test forward pass
        x = torch.rand(1, 3, 224, 224)
        output = model(x)
        print(f"Forward pass: {x.shape} → {output.shape}")

        # Validate output
        validate_model_output(model)
        print("Validation passed")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
