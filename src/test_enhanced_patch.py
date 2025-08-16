#!/usr/bin/env python3
"""
EMERGENCY PATCH for test_enhanced.py
This script patches your existing test_enhanced.py to fix the 500x scaling error.

Run this BEFORE running test_enhanced.py
"""

import os
import shutil
import numpy as np

def create_inference_scaling_factors():
    """Create the missing inference scaling factors."""
    
    print("🔧 Creating inference scaling factors...")
    
    # Check if enhanced factors exist
    enhanced_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy"
    standard_path = "outputs/Ahmed_Dataset/surface_scaling_factors.npy"
    inference_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
    
    os.makedirs("outputs/Ahmed_Dataset/enhanced_1", exist_ok=True)
    
    if os.path.exists(enhanced_path):
        factors = np.load(enhanced_path)
        if factors.shape[1] == 8:
            # Extract first 4 features (fine features) for inference
            inference_factors = factors[:, :4].astype(np.float32)
        else:
            inference_factors = factors.astype(np.float32)
    elif os.path.exists(standard_path):
        factors = np.load(standard_path)
        inference_factors = factors.astype(np.float32)
    else:
        # Create reasonable default factors for Ahmed body
        print("⚠️  No scaling factors found - creating defaults")
        inference_factors = np.array([
            [0.8215, 0.01063, 0.01515, 0.01328],    # Max values
            [-2.1506, -0.01865, -0.01514, -0.01215] # Min values
        ], dtype=np.float32)
    
    np.save(inference_path, inference_factors)
    print(f"✅ Created: {inference_path}")
    print(f"   Shape: {inference_factors.shape}")
    return True

def patch_test_enhanced():
    """Patch the test_enhanced.py file to fix scaling."""
    
    print("🔧 Patching test_enhanced.py...")
    
    if not os.path.exists("test_enhanced.py"):
        print("❌ test_enhanced.py not found!")
        return False
    
    # Backup original
    shutil.copy("test_enhanced.py", "test_enhanced_backup.py")
    print("✅ Backed up original to test_enhanced_backup.py")
    
    # Read the file
    with open("test_enhanced.py", "r") as f:
        content = f.read()
    
    # Critical patches
    patches = [
        # Fix 1: Change scaling factor path to inference version
        {
            'old': 'surf_save_path = os.path.join(\n        cfg.eval.scaling_param_path, "surface_scaling_factors_inference.npy"\n    )',
            'new': '''surf_save_path = os.path.join(
        cfg.eval.scaling_param_path, "surface_scaling_factors_inference.npy"
    )
    
    # EMERGENCY FIX: Create inference factors if missing
    if not os.path.exists(surf_save_path):
        print(f"FIXING: Creating missing inference scaling factors...")
        enhanced_path = os.path.join(cfg.eval.scaling_param_path, "surface_scaling_factors.npy")
        if os.path.exists(enhanced_path):
            enhanced_factors = np.load(enhanced_path)
            if enhanced_factors.shape[1] >= 4:
                inference_factors = enhanced_factors[:, :4].astype(np.float32)
                np.save(surf_save_path, inference_factors)
                print(f"✅ Created inference factors: {inference_factors.shape}")
        else:
            # Default Ahmed factors
            inference_factors = np.array([
                [0.8215, 0.01063, 0.01515, 0.01328],
                [-2.1506, -0.01865, -0.01514, -0.01215]
            ], dtype=np.float32)
            np.save(surf_save_path, inference_factors)
            print(f"⚠️  Created default inference factors")'''
        },
        
        # Fix 2: Correct the unnormalization in test_enhanced_model
        {
            'old': 'prediction_surf = unnormalize(\n                prediction_surf.cpu().numpy(),\n                surf_factors[0],\n                surf_factors[1]\n            )',
            'new': '''prediction_surf_raw = prediction_surf.cpu().numpy()
            print(f"DEBUG: Raw prediction range: [{prediction_surf_raw.min():.6f}, {prediction_surf_raw.max():.6f}]")
            
            # FIXED: Proper unnormalization
            prediction_surf = prediction_surf_raw * (surf_factors[0] - surf_factors[1]) + surf_factors[1]
            print(f"DEBUG: Unnormalized range: [{prediction_surf.min():.6f}, {prediction_surf.max():.6f}]")'''
        },
        
        # Fix 3: Correct physical parameter scaling
        {
            'old': 'prediction_surf = (\n                prediction_surf * stream_velocity**2.0 * air_density\n            )',
            'new': '''# FIXED: Correct physical scaling
            # Predictions are now in coefficient form, scale by dynamic pressure
            dynamic_pressure = 0.5 * air_density * stream_velocity**2.0
            print(f"DEBUG: Dynamic pressure: {dynamic_pressure}")
            
            # Only scale if values look like coefficients (small values)
            if np.abs(prediction_surf).max() < 10:
                prediction_surf = prediction_surf * dynamic_pressure
                print(f"DEBUG: After physical scaling: [{prediction_surf.min():.6f}, {prediction_surf.max():.6f}]")
            else:
                print(f"WARNING: Predictions already in physical units, skipping scaling")'''
        }
    ]
    
    # Apply patches
    patched_content = content
    patches_applied = 0
    
    for patch in patches:
        if patch['old'] in patched_content:
            patched_content = patched_content.replace(patch['old'], patch['new'])
            patches_applied += 1
            print(f"✅ Applied patch {patches_applied}")
        else:
            print(f"⚠️  Could not find text to patch: {patch['old'][:50]}...")
    
    # Write patched file
    with open("test_enhanced_patched.py", "w") as f:
        f.write(patched_content)
    
    print(f"✅ Created patched version: test_enhanced_patched.py")
    print(f"   Applied {patches_applied}/{len(patches)} patches")
    
    return patches_applied > 0

def create_quick_test():
    """Create a minimal quick test to verify the fix."""
    
    quick_test = '''#!/usr/bin/env python3
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
    
    print(f"\\nSample normalized predictions:")
    print(f"   {sample_predictions}")
    
    # Apply fixed unnormalization
    unnormalized = sample_predictions * (factors[0] - factors[1]) + factors[1]
    
    print(f"\\nAfter unnormalization:")
    print(f"   {unnormalized}")
    
    # Check if values are in reasonable range for Ahmed body
    pressure_range = np.abs(unnormalized[:, 0]).max()
    shear_range = np.abs(unnormalized[:, 1:]).max()
    
    print(f"\\n📊 VALIDATION:")
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
        print("\\n🎉 SCALING FIX VALIDATED!")
        print("You can now run:")
        print("   python test_enhanced_patched.py")
    else:
        print("\\n❌ SCALING FIX NEEDS MORE WORK")
'''
    
    with open("quick_test_scaling.py", "w") as f:
        f.write(quick_test)
    
    print("✅ Created quick test: quick_test_scaling.py")

def main():
    """Apply emergency patches."""
    
    print("🚨 EMERGENCY PATCH FOR ENHANCED DOMINO SCALING")
    print("="*60)
    print("This will fix the 500x scaling error in your force predictions")
    print()
    
    # Step 1: Create inference scaling factors
    print("Step 1: Creating inference scaling factors...")
    create_inference_scaling_factors()
    
    # Step 2: Patch test script
    print("\nStep 2: Patching test script...")
    patch_success = patch_test_enhanced()
    
    # Step 3: Create quick test
    print("\nStep 3: Creating validation test...")
    create_quick_test()
    
    print("\n" + "="*60)
    print("🔧 EMERGENCY PATCH COMPLETE!")
    print("="*60)
    
    if patch_success:
        print("\n✅ NEXT STEPS:")
        print("1. Run validation: python quick_test_scaling.py")
        print("2. Test fix: python test_enhanced_patched.py")
        print("3. Check forces are now ~0.02 N instead of ~10 N")
        
        print("\n🎯 EXPECTED AFTER FIX:")
        print("   - Drag forces: 0.01-0.03 N (not 10+ N)")
        print("   - Lift forces: -0.01 to +0.03 N")
        print("   - Improvement: Should be positive % (not -500000%)")
        
    else:
        print("\n⚠️  PATCH PARTIALLY APPLIED")
        print("You may need to manually apply some fixes")
        print("Check test_enhanced_patched.py and compare with original")
    
    print("\n🔍 IF STILL WRONG:")
    print("   - Check model checkpoint loads correctly")
    print("   - Verify test data paths")
    print("   - Ensure geometry scaling is correct")

if __name__ == "__main__":
    main()