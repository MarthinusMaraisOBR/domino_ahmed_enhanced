#!/usr/bin/env python3
"""
Fix Scaling Factors for Enhanced DoMINO Inference
"""
import torch
import numpy as np
from pathlib import Path

def extract_inference_scaling():
    """Extract correct scaling factors for 4-feature inference"""
    
    # Path to your 8-feature scaling from training
    train_scaling_path = Path("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_enhanced.pt")
    
    if not train_scaling_path.exists():
        print("❌ Training scaling factors not found!")
        return
    
    # Load 8-feature scaling
    scaling_8feat = torch.load(train_scaling_path)
    print(f"Loaded training scaling: shape {scaling_8feat.shape}")
    print(f"  Full scaling matrix:")
    print(f"    Fine features (0:4) mean: {scaling_8feat[0, :4].numpy()}")
    print(f"    Fine features (0:4) std:  {scaling_8feat[1, :4].numpy()}")
    print(f"    Coarse features (4:8) mean: {scaling_8feat[0, 4:].numpy()}")
    print(f"    Coarse features (4:8) std:  {scaling_8feat[1, 4:].numpy()}")
    
    # Extract coarse features only (last 4) for inference
    scaling_inference = scaling_8feat[:, 4:8].clone()
    
    # Save inference scaling
    inference_scaling_path = Path("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_inference.pt")
    torch.save(scaling_inference, inference_scaling_path)
    
    print(f"\n✅ Created inference scaling: shape {scaling_inference.shape}")
    print(f"  Saved to: {inference_scaling_path}")
    print(f"\n  Inference scaling values:")
    print(f"    Mean: {scaling_inference[0].numpy()}")
    print(f"    Std:  {scaling_inference[1].numpy()}")
    
    return scaling_inference

def verify_scaling_consistency():
    """Verify that scaling is consistent with data"""
    import h5py
    
    # Load a test file
    test_file = Path("dataset/Ahmed_Full/ahmed_data/case_451.h5")
    
    with h5py.File(test_file, 'r') as f:
        if 'coarse_input' in f:
            coarse = f['coarse_input'][:]
            print(f"\n📊 Test data statistics (case_451):")
            print(f"  Raw coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
            
            # Apply inference scaling
            scaling = torch.load("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_inference.pt")
            scaled = (coarse - scaling[0].numpy()) / scaling[1].numpy()
            
            print(f"  Scaled pressure: mean={scaled[:, 0].mean():.4f}, std={scaled[:, 0].std():.4f}")
            
            if np.abs(scaled).max() > 10:
                print("  ⚠️ WARNING: Scaled values exceed ±10 std!")
            else:
                print("  ✅ Scaling looks reasonable")

if __name__ == "__main__":
    print("=" * 80)
    print("FIXING SCALING FACTORS FOR INFERENCE")
    print("=" * 80)
    
    scaling = extract_inference_scaling()
    verify_scaling_consistency()
    
    print("\n" + "=" * 80)
    print("DONE! Use scaling_factors_inference.pt for testing")
    print("=" * 80)
