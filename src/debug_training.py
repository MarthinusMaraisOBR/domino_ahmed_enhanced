import numpy as np
import torch
from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== TRAINING DEBUG ===")
    
    # Check config
    enhanced = cfg.data_processor.get('use_enhanced_features', False)
    print(f"Enhanced features enabled: {enhanced}")
    
    # Load one training sample
    train_dataset = create_domino_dataset(cfg, "train", [], ["pMean", "wallShearStressMean"], None, None)
    sample = train_dataset[0]
    
    print(f"Training surface_fields shape: {sample['surface_fields'].shape}")
    print(f"Expected: (N, 8) for enhanced, (N, 4) for standard")
    
    # Check field ranges
    fields = sample['surface_fields']
    if fields.shape[1] == 8:
        print(f"Fine pressure range: [{fields[:, 0].min():.3f}, {fields[:, 0].max():.3f}]")
        print(f"Coarse pressure range: [{fields[:, 4].min():.3f}, {fields[:, 4].max():.3f}]")

if __name__ == "__main__":
    main()
