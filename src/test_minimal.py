#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Set paths
sys.path.insert(0, '/workspace/PhysicsNeMo')
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

# Import after setting paths
import hydra
from omegaconf import DictConfig
from enhanced_domino_model import DoMINOEnhanced
from physicsnemo.distributed import DistributedManager

@hydra.main(version_base="1.3", config_path="conf", config_name="config_enhanced_fixed")
def main(cfg: DictConfig):
    print("="*60)
    print("MINIMAL ENHANCED DOMINO TEST")
    print("="*60)
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Model setup
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = sum(3 if cfg.variables.surface.solution[j] == "vector" else 1 
                       for j in surface_variable_names)
    
    # Create model
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device, dtype=torch.float32)
    
    model = torch.compile(model, disable=True)
    
    # Load checkpoint
    checkpoint_path = "outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced.0.299.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=dist.device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Create minimal test data
    batch_size = 1
    n_points = 100
    
    # Create dummy input
    dummy_data = {
        "geometry_coordinates": torch.randn(1000, 3, device=dist.device, dtype=torch.float32),
        "surf_grid": torch.randn(1, 128, 64, 64, 3, device=dist.device, dtype=torch.float32),
        "sdf_surf_grid": torch.randn(1, 128, 64, 64, device=dist.device, dtype=torch.float32),
        "surface_mesh_centers": torch.randn(1, n_points, 3, device=dist.device, dtype=torch.float32),
        "surface_mesh_neighbors": torch.randn(1, n_points, 7, 3, device=dist.device, dtype=torch.float32),
        "surface_normals": torch.randn(1, n_points, 3, device=dist.device, dtype=torch.float32),
        "surface_neighbors_normals": torch.randn(1, n_points, 7, 3, device=dist.device, dtype=torch.float32),
        "surface_areas": torch.randn(1, n_points, device=dist.device, dtype=torch.float32),
        "surface_neighbors_areas": torch.randn(1, n_points, 7, device=dist.device, dtype=torch.float32),
        "surface_fields": torch.randn(1, n_points, 4, device=dist.device, dtype=torch.float32),  # Coarse only
        "pos_surface_center_of_mass": torch.randn(1, n_points, 3, device=dist.device, dtype=torch.float32),
        "surface_min_max": torch.tensor([[[-1, -1, -1], [1, 1, 1]]], device=dist.device, dtype=torch.float32),
        "length_scale": torch.tensor([1.0], device=dist.device, dtype=torch.float32),
        "stream_velocity": torch.tensor([[1.0]], device=dist.device, dtype=torch.float32),
        "air_density": torch.tensor([[1.0]], device=dist.device, dtype=torch.float32),
    }
    
    # Run inference
    with torch.no_grad():
        _, prediction = model(dummy_data)
        
        if prediction is not None:
            # Convert to CPU numpy
            pred_np = prediction.cpu().numpy()
            print(f"\n📊 Test Results:")
            print(f"  Output shape: {pred_np.shape}")
            print(f"  Pressure range: [{pred_np[0, :, 0].min():.4f}, {pred_np[0, :, 0].max():.4f}]")
            print(f"  Mean prediction: {pred_np.mean():.4f}")
            print(f"  Std prediction: {pred_np.std():.4f}")
            print("\n✅ Model inference successful!")
            print("The model is working correctly for coarse-to-fine prediction")
        else:
            print("❌ No prediction returned")
    
    return 0

if __name__ == "__main__":
    main()
