from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
import numpy as np
from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yaml')

# Create dataset without cache
dataset = DoMINODataPipe(
    input_path="/data/ahmed_data/processed/train/",
    phase="train",
    grid_resolution=cfg.model.interp_res,
    volume_variables=None,
    surface_variables=['pMean', 'wallShearStressMean'],
    normalize_coordinates=True,
    sampling=True,
    sample_in_bbox=False,
    surface_points_sample=cfg.model.surface_points_sample,
    surface_factors=np.load('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy'),
    scaling_type=cfg.model.normalization,
    model_type="surface"
)

# Get one sample
sample = dataset[0]
print(f"Surface fields from dataset: mean={sample['surface_fields'].mean():.4f}")
print(f"Shape: {sample['surface_fields'].shape}")
