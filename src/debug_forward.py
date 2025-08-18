import torch
import hydra
from omegaconf import DictConfig
from enhanced_domino_model import DoMINOEnhanced

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== FORWARD PASS DEBUG ===")
    
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=4,
        model_parameters=cfg.model,
    )
    
    checkpoint = torch.load(f"{cfg.resume_dir}/{cfg.eval.checkpoint_name}", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Minimal working input
    dummy_input = {
        'surface_fields': torch.randn(1, 100, 4),
        'geometry_coordinates': torch.randn(1000, 3),
        'surface_mesh_centers': torch.randn(1, 100, 3),
        'surface_normals': torch.randn(1, 100, 3),
        'stream_velocity': torch.tensor([[1.0]]),
        'air_density': torch.tensor([[1.0]]),
        'surf_grid': torch.randn(1, 32, 32, 32, 3),
        'sdf_surf_grid': torch.randn(1, 32, 32, 32),
        'surface_areas': torch.randn(1, 100),
        'surface_mesh_neighbors': torch.randn(1, 100, 7, 3),
        'surface_neighbors_normals': torch.randn(1, 100, 7, 3),
        'surface_neighbors_areas': torch.randn(1, 100, 7),
        'pos_surface_center_of_mass': torch.randn(1, 100, 3),
        'surface_min_max': torch.randn(1, 2, 3),
        'length_scale': torch.tensor([1.0]),
    }
    
    print("Testing forward pass step by step...")
    
    try:
        with torch.no_grad():
            # Check what the model thinks about input
            surface_fields = dummy_input["surface_fields"]
            print(f"Input shape: {surface_fields.shape}")
            print(f"Model expects enhanced: {model.use_enhanced_features}")
            
            # Call forward
            vol_pred, surf_pred = model(dummy_input)
            
            print(f"Volume output: {vol_pred}")
            print(f"Surface output: {surf_pred}")
            
            if surf_pred is not None:
                print(f"Surface output shape: {surf_pred.shape}")
            else:
                print("❌ Surface output is None!")
                
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
