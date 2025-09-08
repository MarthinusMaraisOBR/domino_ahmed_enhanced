import torch
import numpy as np
from enhanced_domino_model import DoMINOEnhanced
from omegaconf import DictConfig

# Setup
cfg = DictConfig({
    'model': {
        'enhanced_model': {'surface_input_features': 8, 'coarse_to_fine': {'hidden_layers': [256, 256]}},
        'activation': 'relu',
        'model_type': 'surface',
    }
})

model = DoMINOEnhanced(
    input_features=3,
    output_features_surf=4,
    model_parameters=cfg.model,
)

# Load checkpoint
checkpoint = torch.load('outputs/Ahmed_Dataset/11/models/DoMINOEnhanced.0.49.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Create dummy normalized input (similar to test data)
dummy_input = {
    'surface_fields': torch.randn(1, 1000, 4) * 0.15 + 0.72,  # Mean ~0.72, std ~0.15
    'geometry_encoding': torch.randn(1, 1000, 448),
}

with torch.no_grad():
    # Get raw model output
    _, raw_output = model(dummy_input)
    
print(f"Raw model output stats:")
print(f"  Shape: {raw_output.shape}")
print(f"  Mean: {raw_output.mean().item():.4f}")
print(f"  Std: {raw_output.std().item():.4f}")
print(f"  Min: {raw_output.min().item():.4f}")
print(f"  Max: {raw_output.max().item():.4f}")
