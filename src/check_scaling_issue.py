#!/usr/bin/env python3
"""check_scaling_issue.py - Verify scaling factor application"""

import numpy as np
import torch

def check_scaling_problem():
    print("="*80)
    print("SCALING FACTOR DIAGNOSIS")
    print("="*80)
    
    # Load scaling factors
    train_scale_path = "outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy"
    scales = np.load(train_scale_path)
    
    print(f"\nScaling factors shape: {scales.shape}")
    print("\nFine features (first 4 columns):")
    for i in range(4):
        print(f"  Feature {i}: max={scales[0,i]:.6f}, min={scales[1,i]:.6f}")
    
    print("\nCoarse features (last 4 columns):")
    for i in range(4, 8):
        print(f"  Feature {i-4}: max={scales[0,i]:.6f}, min={scales[1,i]:.6f}")
    
    # Simulate normalization and denormalization
    print("\n" + "="*40)
    print("NORMALIZATION TEST")
    print("="*40)
    
    # Real data values
    real_value = -0.11  # Typical pressure
    
    # Normalize using fine scaling
    normalized = (real_value - scales[1,0]) / (scales[0,0] - scales[1,0])
    print(f"\nReal value: {real_value:.6f}")
    print(f"Normalized: {normalized:.6f}")
    
    # If model outputs this normalized value but we denormalize with wrong scale
    denorm_correct = normalized * (scales[0,0] - scales[1,0]) + scales[1,0]
    denorm_wrong = normalized * (scales[0,4] - scales[1,4]) + scales[1,4]  # Using coarse scale
    
    print(f"\nDenormalized (correct scale): {denorm_correct:.6f}")
    print(f"Denormalized (wrong scale):   {denorm_wrong:.6f}")
    
    # Check if this matches our observed shift
    observed_shift = -0.79 - (-0.11)
    print(f"\nObserved shift in predictions: {observed_shift:.6f}")
    
    # Test with inference scaling
    if scales.shape[1] == 8:
        print("\n⚠️ Training used 8-feature scaling")
        print("   Inference should use only first 4 columns!")
        
        # What happens if we use all 8?
        mean_scale_max = scales[0,:4].mean()
        mean_scale_min = scales[1,:4].mean()
        
        print(f"\n  Fine-only scale range: [{mean_scale_min:.4f}, {mean_scale_max:.4f}]")
        
if __name__ == "__main__":
    check_scaling_problem()
