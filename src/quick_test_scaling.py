#!/usr/bin/env python3
"""
Quick test to verify scaling fix works.
"""

import numpy as np
import os

def test_scaling_fix():
    """Test if the scaling fix resolves the 500x error."""
    
    print("🧪 TESTING SCALING FIX")
    print("="*40)
    
    # Check if inference factors exist
    inference_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
    
    if not os.path.exists(inference_path):
        print("❌ Inference scaling factors missing!")
        return False
    
    factors = np.load(inference_path)
    print(f"✅ Loaded scaling factors: {factors.shape}")
    print(f"   Max: {factors[0]}")
    print(f"   Min: {factors[1]}")
    
    # Test unnormalization with sample data
    # Simulate model output (normalized predictions)
    sample_predictions = np.array([
        [0.1, 0.01, 0.005, -0.002],  # Sample normalized values
        [-0.2, -0.005, 0.01, 0.003]
    ])
    
    print(f"\nSample normalized predictions:")
    print(f"   {sample_predictions}")
    
    # Apply fixed unnormalization
    unnormalized = sample_predictions * (factors[0] - factors[1]) + factors[1]
    
    print(f"\nAfter unnormalization:")
    print(f"   {unnormalized}")
    
    # Check if values are in reasonable range for Ahmed body
    pressure_range = np.abs(unnormalized[:, 0]).max()
    shear_range = np.abs(unnormalized[:, 1:]).max()
    
    print(f"\n📊 VALIDATION:")
    print(f"   Max pressure coefficient: {pressure_range:.3f}")
    print(f"   Max wall shear stress: {shear_range:.6f}")
    
    # Reasonable ranges for Ahmed body
    pressure_ok = 0.1 < pressure_range < 5.0  # Pressure coefficient
    shear_ok = 0.001 < shear_range < 0.1      # Wall shear stress
    
    if pressure_ok and shear_ok:
        print("   ✅ Values in reasonable range!")
        return True
    else:
        print("   ⚠️  Values may still be incorrect")
        print(f"   Expected pressure: 0.1-5.0, got {pressure_range:.3f}")
        print(f"   Expected shear: 0.001-0.1, got {shear_range:.6f}")
        return False

if __name__ == "__main__":
    success = test_scaling_fix()
    
    if success:
        print("\n🎉 SCALING FIX VALIDATED!")
        print("You can now run:")
        print("   python test_enhanced_patched.py")
    else:
        print("\n❌ SCALING FIX NEEDS MORE WORK")
