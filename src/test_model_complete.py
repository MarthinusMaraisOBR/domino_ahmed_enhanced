import torch
import numpy as np
from enhanced_domino_model import DoMINOEnhanced
from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yaml')
model = DoMINOEnhanced(
    input_features=3,
    output_features_surf=4,
    model_parameters=cfg.model,
).cuda()

checkpoint = torch.load('outputs/Ahmed_Dataset/11/models/DoMINOEnhanced.0.49.pt')
model.load_state_dict(checkpoint)
model.eval()

# Create complete test input
batch_size = 1
n_points = 1000

test_input = {
    'surface_fields': torch.ones(batch_size, n_points, 4).cuda() * 0.774,  # Training mean
    'geometry_coordinates': torch.randn(batch_size, n_points, 3).cuda(),
    'surface_mesh_centers': torch.randn(batch_size, n_points, 3).cuda(),
    'surface_normals': torch.randn(batch_size, n_points, 3).cuda(),
    'surface_areas': torch.ones(batch_size, n_points, 1).cuda(),
    # Add other required fields based on what the model expects
}

with torch.no_grad():
    try:
        _, output = model(test_input)
        print(f"Input mean: 0.774 (training coarse mean)")
        print(f"Expected output mean: 0.773 (training fine mean)")
        print(f"Actual output mean: {output.mean().item():.4f}")
        print(f"Output shape: {output.shape}")
        
        # Check each feature
        for i in range(4):
            print(f"  Feature {i}: mean={output[0,:,i].mean().item():.4f}, std={output[0,:,i].std().item():.4f}")
    except KeyError as e:
        print(f"Missing required input: {e}")
        print("Add this to test_input dict")
