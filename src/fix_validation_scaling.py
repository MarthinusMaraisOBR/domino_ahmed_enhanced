import numpy as np
import os

# Load the 8-feature scaling factors
enhanced_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy"
if os.path.exists(enhanced_path):
    factors_8 = np.load(enhanced_path)
    print(f"Loaded enhanced scaling factors: {factors_8.shape}")
    
    # Extract just the first 4 features for validation
    factors_4 = factors_8[:, :4]
    
    # Save standard 4-feature scaling factors
    standard_path = "outputs/Ahmed_Dataset/surface_scaling_factors.npy"
    np.save(standard_path, factors_4)
    print(f"Created standard scaling factors: {factors_4.shape}")
    
    # Also create a validation-specific version
    val_path = "outputs/Ahmed_Dataset/surface_scaling_factors_val.npy"
    np.save(val_path, factors_4)
    print(f"Created validation scaling factors: {factors_4.shape}")
else:
    print("Enhanced scaling factors not found!")
