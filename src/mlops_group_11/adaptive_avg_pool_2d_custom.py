import numpy as np
import timm
import torch
import torch.nn as nn

class AdaptiveAvgPool2dCustom(nn.Module):
    """Custom AdaptiveAvgPool2d for ONNX export compatibility.
    
    This replaces nn.AdaptiveAvgPool2d to fix ONNX export issues with dynamic input sizes.
    Solution from: https://github.com/pytorch/pytorch/issues/42653#issuecomment-1168816422
    
    Args:
        output_size: Target output size (H, W) or int for square output.
    """
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        if isinstance(output_size, int):
            self.output_size = np.array([output_size, output_size])
        else:
            self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        """Forward pass using regular AvgPool2d with computed kernel and stride sizes."""
        input_size = np.array(x.shape[-2:])
        
        # Compute stride and kernel sizes based on input shape
        stride_size = np.floor(input_size / self.output_size).astype(np.int32)
        kernel_size = input_size - (self.output_size - 1) * stride_size
        
        # Handle edge case where stride becomes 0 - use global pooling instead
        if np.any(stride_size == 0) or np.any(kernel_size <= 0):
            # If output size matches or exceeds input size, just return the input or use identity
            if np.all(self.output_size >= input_size):
                return x
            # Otherwise use a simple average pool with kernel = input size
            avg = nn.AvgPool2d(kernel_size=tuple(input_size.tolist()))
            return avg(x)
        
        # Use regular AvgPool2d with computed parameters
        avg = nn.AvgPool2d(kernel_size=tuple(kernel_size.tolist()), stride=tuple(stride_size.tolist()))
        x = avg(x)
        return x