import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def compute_enhanced_scaling_factors(data_dir, output_dir):
    """Compute scaling factors for enhanced 8-feature surface fields"""
    
    print("Computing scaling factors for enhanced dataset...")
    
    # Get all NPY files
    npy_files = list(Path(data_dir).glob("*.npy"))
    print(f"Found {len(npy_files)} NPY files")
    
    # Initialize min/max trackers for 8 features
    global_min = np.full(8, np.inf)
    global_max = np.full(8, -np.inf)
    
    # Process each file
    for npy_file in tqdm(npy_files, desc="Processing files"):
        data = np.load(npy_file, allow_pickle=True).item()
        
        if 'surface_fields' in data:
            surface_fields = data['surface_fields']
            
            # Check shape
            if surface_fields.shape[1] == 8:
                # Update min/max for each feature
                for i in range(8):
                    field_data = surface_fields[:, i]
                    global_min[i] = min(global_min[i], field_data.min())
                    global_max[i] = max(global_max[i], field_data.max())
            else:
                print(f"Warning: {npy_file.name} has {surface_fields.shape[1]} features, expected 8")
    
    # Create scaling factors array [2, 8]
    # Row 0: max values, Row 1: min values
    scaling_factors = np.array([global_max, global_min], dtype=np.float32)
    
    print("\nScaling factors computed:")
    print("Feature | Min | Max")
    print("-" * 30)
    for i in range(8):
        feature_name = f"Feature {i}"
        if i < 4:
            feature_name = f"Fine {i}"
        else:
            feature_name = f"Coarse {i-4}"
        print(f"{feature_name:10} | {global_min[i]:.6f} | {global_max[i]:.6f}")
    
    # Save scaling factors
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "surface_scaling_factors_enhanced.npy")
    np.save(save_path, scaling_factors)
    print(f"\nSaved to: {save_path}")
    
    # Also save in the expected location for training
    standard_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy"
    os.makedirs(os.path.dirname(standard_path), exist_ok=True)
    np.save(standard_path, scaling_factors)
    print(f"Also saved to: {standard_path}")
    
    return scaling_factors

# Run computation
data_dir = "/data/ahmed_data/ahmed_data/train/"
output_dir = "outputs/Ahmed_Dataset/enhanced_1/"

scaling_factors = compute_enhanced_scaling_factors(data_dir, output_dir)
