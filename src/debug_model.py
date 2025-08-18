import torch
import hydra
from omegaconf import DictConfig
from enhanced_domino_model import DoMINOEnhanced

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== MODEL DEBUG ===")
    
    # Create model
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=4,
        model_parameters=cfg.model,
    )
    
    print(f"Model enhanced features: {model.use_enhanced_features}")
    print(f"Has coarse_to_fine_model: {hasattr(model, 'coarse_to_fine_model')}")
    
    # Check checkpoint path
    checkpoint_path = f"{cfg.resume_dir}/{cfg.eval.checkpoint_name}"
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint exists: {torch.load(checkpoint_path) is not None}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    print("✅ Model loaded successfully")
    
    # Test with dummy data
    dummy_input = {
        'surface_fields': torch.randn(1, 100, 4),  # Only 4 features (coarse only)
        'geometry_coordinates': torch.randn(1000, 3),
        'surface_mesh_centers': torch.randn(1, 100, 3),
        'surface_normals': torch.randn(1, 100, 3),
        'stream_velocity': torch.tensor([[1.0]]),
        'air_density': torch.tensor([[1.0]]),
    }
    
    # Add minimal required fields
    dummy_input.update({
        'surf_grid': torch.randn(1, 32, 32, 32, 3),
        'sdf_surf_grid': torch.randn(1, 32, 32, 32),
        'surface_areas': torch.randn(1, 100),
        'surface_mesh_neighbors': torch.randn(1, 100, 7, 3),
        'surface_neighbors_normals': torch.randn(1, 100, 7, 3),
        'surface_neighbors_areas': torch.randn(1, 100, 7),
        'pos_surface_center_of_mass': torch.randn(1, 100, 3),
        'surface_min_max': torch.randn(1, 2, 3),
        'length_scale': torch.tensor([1.0]),
    })
    
    with torch.no_grad():
        _, output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model output range: [{output.min():.3f}, {output.max():.3f}]")

if __name__ == "__main__":
    main()
