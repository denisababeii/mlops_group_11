"""Convert PyTorch model checkpoint to TorchScript format.

TorchScript provides faster inference than regular PyTorch by optimizing the model
and removing Python overhead. It's compatible with complex models like CSATv2.

Usage:
    uv run scripts/convert_to_torchscript.py

Requirements:
    - Trained model checkpoint at 'models/best_model.pth'

Output:
    - TorchScript model saved to 'models/model_scripted.pt'
"""

import torch
from pathlib import Path
from mlops_group_11.model import load_model

print("Converting model to TorchScript...")

# Load the trained model
model_name = 'csatv2_21m.sw_r512_in1k'
checkpoint_path = Path('models/best_model.pth')
output_path = Path('models/model_scripted.pt')
device = torch.device('cpu')  # Use CPU for compatibility

print(f"Loading model from {checkpoint_path}...")
model = load_model(
    model_name=model_name,
    checkpoint_path=checkpoint_path,
    num_classes=24,
    device=device
)
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224, device=device)

# Convert to TorchScript using tracing
print("Converting to TorchScript (this may take a minute)...")
try:
    scripted_model = torch.jit.trace(model, example_input)
    
    # Save the scripted model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted_model, str(output_path))
    
    print(f"\n✓ TorchScript model saved to: {output_path}")
    print(f"✓ Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Test the scripted model
    print("\nTesting scripted model...")
    with torch.no_grad():
        output = scripted_model(example_input)
        print(f"✓ Test inference successful: {example_input.shape} → {output.shape}")
    
    print("\nTorchScript conversion complete!")
    print("Update your BentoML service to use this model for faster inference.")
    
except Exception as e:
    print(f"\n✗ TorchScript conversion failed: {e}")
    print("Falling back to regular PyTorch model...")
    raise
