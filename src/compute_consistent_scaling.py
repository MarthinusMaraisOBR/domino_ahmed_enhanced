import numpy as np
import os

# Compute scaling factors from raw training data
all_data = []
for i in range(1, 401):  # All training files
    path = f'/data/ahmed_data/processed/train/run_{i}.npy'
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True).item()
        all_data.append(data['surface_fields'])

all_data = np.vstack(all_data)
sf_max = np.max(all_data, axis=0)
sf_min = np.min(all_data, axis=0)

scaling_factors = np.array([sf_max, sf_min])
np.save('outputs/Ahmed_Dataset/consistent_scaling.npy', scaling_factors)
print(f"Scaling factors saved")
print(f"Max: {sf_max}")
print(f"Min: {sf_min}")
