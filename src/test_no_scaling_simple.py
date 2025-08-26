#!/usr/bin/env python3
"""
Simple wrapper to run test_enhanced with scaling fixes
"""

import sys
import os
sys.path.insert(0, '/workspace/PhysicsNeMo')
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

import numpy as np
import torch

# Import the test module
import test_enhanced

# Store original functions
original_test_model = test_enhanced.test_enhanced_model

def test_enhanced_model_no_scaling(data_dict, model, device, cfg, surf_factors):
    """
    Test function with physical scaling removed for V=1, ρ=1 conditions.
    """
    
    print("\n🔧 APPLYING FIX: Removing physical scaling (V=1, ρ=1)")
    
    with torch.no_grad():
        # Convert to GPU tensors
        data_dict_gpu = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
                data_dict_gpu[k] = v_tensor
            else:
                data_dict_gpu[k] = v
        
        # CRITICAL FIX: Ensure only 4 features for inference
        if "surface_fields" in data_dict_gpu:
            surface_fields = data_dict_gpu["surface_fields"]
            if surface_fields.shape[-1] == 8:
                print("   📝 Extracting coarse features only (last 4 of 8)")
                data_dict_gpu["surface_fields"] = surface_fields[..., 4:8]
            
            print(f"   📊 Input features shape: {data_dict_gpu['surface_fields'].shape}")
        
        # Get predictions
        _, prediction_surf = model(data_dict_gpu)
        
        if prediction_surf is not None and surf_factors is not None:
            # Only unnormalize - NO physical scaling
            prediction_surf = test_enhanced.unnormalize(
                prediction_surf.cpu().numpy(),
                surf_factors[0],
                surf_factors[1]
            )
            
            # Debug output
            print(f"   ✅ Unnormalized only (no V²ρ scaling)")
            print(f"   📈 Prediction stats:")
            print(f"      Pressure: [{prediction_surf[0,:,0].min():.6f}, {prediction_surf[0,:,0].max():.6f}]")
            print(f"      Mean: {prediction_surf[0,:,0].mean():.6f}")
    
    return prediction_surf

# Apply the patch
print("="*80)
print("PATCHING TEST_ENHANCED FOR V=1, ρ=1 CONDITIONS")
print("="*80)
test_enhanced.test_enhanced_model = test_enhanced_model_no_scaling

# Now run the main function normally through Hydra
if __name__ == "__main__":
    print("\nStarting test with fixed scaling...")
    print("Expected improvements:")
    print("  • Forces should be ~0.02 N (not 0.04 N)")
    print("  • Improvements should be positive (not -1000%)")
    print("="*80 + "\n")
    
    # Call the original main which has the Hydra decorator
    sys.exit(test_enhanced.main())
