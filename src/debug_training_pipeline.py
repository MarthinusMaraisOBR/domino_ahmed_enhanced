import numpy as np
import torch
from omegaconf import OmegaConf

# First, check raw data
data = np.load('/data/ahmed_data/processed/train/run_1.npy', allow_pickle=True).item()
print("Raw processed data:")
print(f"  Surface fields shape: {data['surface_fields'].shape}")
print(f"  Mean of all 8 features: {data['surface_fields'].mean():.4f}")
print(f"  Fine features (0-3) mean: {data['surface_fields'][:, :4].mean():.4f}")
print(f"  Coarse features (4-7) mean: {data['surface_fields'][:, 4:].mean():.4f}")

# Now check what normalization the dataset applies
cfg = OmegaConf.load('conf/config.yaml')
print(f"\nConfig normalization setting: {cfg.model.get('normalization', 'not set')}")
print(f"Scaling type: {cfg.model.get('scaling_type', 'not set')}")

# Check if there's double normalization happening
sf = np.load('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy')
print(f"\nScaling factors shape: {sf.shape}")

# Manually normalize to see expected values
fine_norm = (data['surface_fields'][:, :4] - sf[1, :4]) / (sf[0, :4] - sf[1, :4])
print(f"Manually normalized fine features mean: {fine_norm.mean():.4f}")
print("This is what the model SHOULD be learning to output")

# The model outputs 0.39, so there's likely another transform
print(f"\nDifference: {fine_norm.mean() - 0.39:.4f}")
print("This offset suggests double normalization or wrong scaling factors")
