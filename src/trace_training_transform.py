import numpy as np

# Raw data
data = np.load('/data/ahmed_data/processed/train/run_1.npy', allow_pickle=True).item()
raw_fine = data['surface_fields'][:, :4]

# What we expected (manual calculation)
sf = np.load('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy')
expected_norm = (raw_fine - sf[1, :4]) / (sf[0, :4] - sf[1, :4])
print(f"Expected normalized mean: {expected_norm.mean():.4f}")

# What the dataset actually provides
print(f"Actual dataset mean: 0.4485")

# The difference
diff = expected_norm.mean() - 0.4485
print(f"Difference: {diff:.4f}")

# This means there's an additional transformation
# Let's figure out what would cause this
# If x_expected = 0.7294 and x_actual = 0.4485
# Then x_actual = scale * x_expected + shift
# Assuming linear: 0.4485 = scale * 0.7294 + shift

# Try scale = 1, shift = -0.28
shift = 0.4485 - 0.7294
print(f"\nPossible transformation: x_actual = x_expected + {shift:.4f}")

# Or there could be different scaling factors being used
print("\nOr the dataset might be using different scaling factors entirely")
