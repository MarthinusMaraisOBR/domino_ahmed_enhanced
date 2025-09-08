import torch
import numpy as np
from enhanced_domino_model import DoMINOEnhanced
from omegaconf import OmegaConf

# Load config and model
cfg = OmegaConf.load('conf/config.yaml')
model = DoMINOEnhanced(
    input_features=3,
    output_features_surf=4,
    model_parameters=cfg.model,
).cuda()

checkpoint = torch.load('outputs/Ahmed_Dataset/11/models/DoMINOEnhanced.0.49.pt')
model.load_state_dict(checkpoint)
model.eval()

# Create test input with known normalized values
# Using the mean values from training
test_input = {
    'surface_fields': torch.ones(1, 1000, 4).cuda() * 0.726,  # Coarse input
    'geometry_encoding': torch.randn(1, 1000, 448).cuda(),
}

with torch.no_grad():
    _, output = model(test_input)
    print(f"Input mean: 0.726")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
# The output should be around 0.765 for correct pressure
# If it's still 0.48, the model weights are wrong
