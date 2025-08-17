import numpy as np

# Check all scaling factor shapes
files = [
    "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy",
    "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_enhanced.npy", 
    "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
]

for file in files:
    factors = np.load(file)
    print(f"{file}: {factors.shape}")
    print(f"  Max: {factors[0]}")
    print(f"  Min: {factors[1]}")
    print()
