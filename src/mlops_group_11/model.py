import timm
import torch
import torch.nn as nn


def create_timm_model(name: str = "csatv2_21m.sw_r512_in1k", pretrained: bool = True, num_classes=24) -> nn.Module:
    """Create and return a timm model by name.

    Args:
        name: model identifier for `timm.create_model`.
        pretrained: whether to load pretrained weights.
        num_classes: number of output classes for the model.

    Returns:
        A `torch.nn.Module` model instance.
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


if __name__ == "__main__":
    model = create_timm_model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.rand(1, 3, 512, 512)
    out = model(x)
    print(f"Output shape of model: {out.shape}")
