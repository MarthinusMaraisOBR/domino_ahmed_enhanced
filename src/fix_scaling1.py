import numpy as np

# Load the 8-feature factors and extract first 4
enhanced = np.load("outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy")
inference = enhanced[:, :4]  # Extract first 4 features only

# Save the corrected inference factors
np.save("outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy", inference)

print(f"Fixed inference factors:")
print(f"  Shape: {inference.shape}")
print(f"  Max: {inference[0]}")
print(f"  Min: {inference[1]}")
