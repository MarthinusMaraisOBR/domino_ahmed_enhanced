import numpy as np

# Load scaling factors
surf_factors = np.load('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy')
print("Scaling factors (4 features):")
print(f"  Max: {surf_factors[0]}")
print(f"  Min: {surf_factors[1]}")

# Test normalization with actual data ranges
test_pressure = -0.11  # Typical test data pressure mean
test_shear = -0.0005   # Typical test data shear mean

# Method 1: (data - min) / (max - min) - standard min-max normalization
normalized_p = (test_pressure - surf_factors[1][0]) / (surf_factors[0][0] - surf_factors[1][0])
print(f"\nNormalized pressure (mean -0.11): {normalized_p:.4f}")

# Test with actual coarse data range
coarse_p_min = -0.8218
coarse_p_max = 0.5174
norm_min = (coarse_p_min - surf_factors[1][0]) / (surf_factors[0][0] - surf_factors[1][0])
norm_max = (coarse_p_max - surf_factors[1][0]) / (surf_factors[0][0] - surf_factors[1][0])
print(f"Coarse pressure range normalized: [{norm_min:.4f}, {norm_max:.4f}]")

# This should give values in [0, 1] range if scaling factors are correct
