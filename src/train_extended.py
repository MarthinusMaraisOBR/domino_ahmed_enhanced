import torch
import os
import sys
sys.path.insert(0, '/workspace/PhysicsNeMo')
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

from train import main
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="conf", config_name="config_enhanced_fixed")
def train_extended(cfg: DictConfig):
    # Modify config for extended training
    cfg.train.epochs = 500
    cfg.exp_tag = "enhanced_extended"
    
    # Set very low learning rate for fine-tuning
    print("Starting extended training from epoch 300...")
    print("This will refine the model with a low learning rate")
    
    # Run main training
    return main(cfg)

if __name__ == "__main__":
    train_extended()
