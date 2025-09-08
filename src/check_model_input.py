import torch
import numpy as np

# Check what input size the model expects
checkpoint = torch.load('outputs/Ahmed_Dataset/11/models/DoMINOEnhanced.0.49.pt', map_location='cpu')

# Look for layers that process surface features
print("Checking model layers for surface feature processing:")
for key in sorted(checkpoint.keys()):
    if 'surface' in key.lower() or 'coarse_to_fine' in key:
        if 'weight' in key and len(checkpoint[key].shape) >= 2:
            print(f"{key}: shape {checkpoint[key].shape}")
            if 'coarse_to_fine' in key and '0.weight' in key:
                print(f"  -> Coarse-to-fine expects {checkpoint[key].shape[1]} input features")

# Check our actual input
surf_factors = np.load('outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors_inference.npy')
print(f"\nScaling factors shape: {surf_factors.shape}")
print(f"We're providing {surf_factors.shape[1]} features to the model")

# Check training scaling factors
train_factors = np.load('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy')
print(f"\nTraining scaling factors shape: {train_factors.shape}")
print(f"Training used {train_factors.shape[1]} features")
