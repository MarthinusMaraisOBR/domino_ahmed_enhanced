import torch
import yaml
from omegaconf import DictConfig
import sys
import os

# Add paths
sys.path.append('/mnt/windows/ahmed-ml-project/domino_enhanced/PhysicsNeMo')

# Test imports
try:
    from enhanced_domino_model_v3 import DoMINOEnhancedV3
    print("✓ V3 model imported")
except ImportError as e:
    print(f"✗ V3 model import failed: {e}")
    exit(1)

try:
    from coefficient_data_loader import EnhancedDatasetWithCoefficients
    print("✓ Coefficient data loader imported")
except ImportError as e:
    print(f"✗ Data loader import failed: {e}")
    exit(1)

try:
    from multitask_loss import MultiTaskLoss
    print("✓ Multi-task loss imported")
except ImportError as e:
    print(f"✗ Loss import failed: {e}")
    exit(1)

# Load config
with open('conf/config_v3.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("✓ Config loaded")

# Test dataset
print("\nTesting dataset with coefficients...")
dataset = EnhancedDatasetWithCoefficients(
    npy_data_path="/data/ahmed_data/processed/train/",
    coefficient_path="/mnt/windows/ahmed-ml-project/ahmed_data/organized/",
    split="train",
    use_enhanced_features=True,
    normalize_coefficients=True,
)

sample = dataset[0]
print(f"✓ Dataset created with {len(dataset)} samples")
print(f"  Surface fields shape: {sample['surface_fields'].shape}")
if 'fine_coefficients' in sample:
    print(f"  Coefficients found: Cd={sample['fine_coefficients'][0]:.4f}, Cl={sample['fine_coefficients'][1]:.4f}")
else:
    print("  ✗ No coefficients in dataset!")

# Test model
print("\nTesting V3 model...")
model_config = DictConfig(config['model'])
model = DoMINOEnhancedV3(
    input_features=3,
    output_features_vol=None,
    output_features_surf=4,
    model_parameters=model_config
).cuda()
print("✓ Model created on GPU")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 1
n_points = 1000
test_data = {
    'surface_fields': torch.randn(batch_size, n_points, 8).cuda(),
    'geometry_coordinates': torch.randn(batch_size, n_points, 3).cuda(),
    'surf_grid': torch.randn(batch_size, 128, 64, 64, 3).cuda(),
    'sdf_surf_grid': torch.randn(batch_size, 128, 64, 64).cuda(),
    'surface_mesh_centers': torch.randn(batch_size, n_points, 3).cuda(),
    'surface_min_max': torch.randn(batch_size, 2, 3).cuda(),
}

with torch.no_grad():
    vol_pred, surf_pred = model(test_data)
    coeffs = model.get_coefficients()

print(f"✓ Forward pass complete")
print(f"  Surface prediction: {surf_pred.shape}")
print(f"  Coefficients: {coeffs.shape if coeffs is not None else 'None'}")

print("\n✅ All tests passed! Ready to train with coefficients.")
