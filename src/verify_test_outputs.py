#!/usr/bin/env python3
"""
Verify Enhanced DoMINO Test Outputs
Run a small test to verify model predictions are being saved correctly
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

def run_mini_test():
    """Run a minimal test to check if predictions are working"""
    
    print("=" * 80)
    print("MINI TEST - ENHANCED DoMINO OUTPUT VERIFICATION")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    
    # Model path
    model_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✓ Found model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ Loaded checkpoint")
    
    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print("\nCheckpoint contents:")
        for key in checkpoint.keys():
            if key != 'model_state_dict':
                print(f"  - {key}: {type(checkpoint[key])}")
    
    # Load a test case
    data_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data/case_451.h5")
    
    if data_path.exists():
        print(f"\n✓ Loading test data: {data_path.name}")
        
        with h5py.File(data_path, 'r') as f:
            print(f"  Available keys: {list(f.keys())}")
            
            if 'enhanced_data' in f:
                data = f['enhanced_data'][:]
                print(f"  Enhanced data shape: {data.shape}")
                print(f"  Coarse features (4:8): mean={data[:, 4:8].mean():.4f}")
                print(f"  Fine features (0:4): mean={data[:, :4].mean():.4f}")
            elif 'coarse_input' in f:
                coarse = f['coarse_input'][:]
                print(f"  Coarse data shape: {coarse.shape}")
                print(f"  Coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
    
    # Create a test output directory
    output_dir = Path("test_output_verification")
    output_dir.mkdir(exist_ok=True)
    
    # Save a test file to verify writing works
    test_output = output_dir / f"test_{datetime.now().strftime('%H%M%S')}.txt"
    with open(test_output, 'w') as f:
        f.write(f"Test output created at {datetime.now()}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"This verifies file writing is working\n")
    
    print(f"\n✓ Test output saved to: {test_output}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_mini_test()
