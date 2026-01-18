# tests/test_model.py
"""
Model unit tests (M16 requirement: unit tests related to model construction).

These tests focus on:
1) Input validation (bad num_classes should raise ValueError)
2) Model output shape check (input -> output shape)

Corresponds to functions in mlops_group_11.model:
- create_timm_model
- validate_model_output
"""

import pytest
import torch
from mlops_group_11.model import create_timm_model, validate_model_output


def test_create_timm_model_invalid_num_classes_raises() -> None:
    """
    Model test: verify defensive programming works.
    If num_classes <= 0, create_timm_model should raise ValueError.
    """
    with pytest.raises(ValueError):
        create_timm_model(name="resnet18", pretrained=False, num_classes=0)


def test_model_forward_output_shape() -> None:
    """
    Model test: given an input batch (B, C, H, W),
    output must be (B, num_classes).

    We use a small model to keep tests fast.
    """
    num_classes = 7
    model = create_timm_model(name="resnet18", pretrained=False, num_classes=num_classes)
    model.eval()

    x = torch.rand(2, 3, 224, 224)  # (batch=2)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, num_classes), "Model output shape should be (batch_size, num_classes)."


def test_validate_model_output_helper_passes() -> None:
    """
    Check that validate_model_output returns True for a correctly built model.

    The helper should confirm that:
    - the model output has the correct shape
    - the output values are valid numbers
    - the output uses the correct data type

    """
    num_classes = 5
    model = create_timm_model(name="resnet18", pretrained=False, num_classes=num_classes)

    ok = validate_model_output(model, input_shape=(1, 3, 224, 224), num_classes=num_classes)
    assert ok is True, "validate_model_output should return True when the model behaves correctly."
