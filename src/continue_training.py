#!/usr/bin/env python3
import os
import sys
import torch
import shutil
from pathlib import Path

# Setup paths
sys.path.insert(0, '/workspace/PhysicsNeMo')
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

# Copy checkpoint to allow continuation
src_dir = Path("outputs/Ahmed_Dataset/enhanced_fixed/models")
dst_dir = Path("outputs/Ahmed_Dataset/enhanced_continued/models")
dst_dir.mkdir(parents=True, exist_ok=True)

# Copy the checkpoint with adjusted epoch number
src_checkpoint = src_dir / "checkpoint.0.299.pt"
dst_checkpoint = dst_dir / "checkpoint.0.299.pt"

if src_checkpoint.exists():
    shutil.copy2(src_checkpoint, dst_checkpoint)
    
    # Also copy the model
    src_model = src_dir / "DoMINOEnhanced.0.299.pt"
    dst_model = dst_dir / "DoMINOEnhanced.0.299.pt"
    shutil.copy2(src_model, dst_model)
    
    print("Checkpoints copied for continued training")
    print(f"Ready to train from epoch 300 to 500")
else:
    print("Source checkpoint not found")
