#!/usr/bin/env python3
"""
Complete investigation of scaling factors to identify the mismatch
between training and testing
"""

import torch
import numpy as np
from pathlib import Path
import pyvista as pv

def investigate_all_scaling():
    """
    Comprehensively check all scaling factors and their usage
    """
    print("=" * 80)
    print("COMPLETE SCALING FACTOR INVESTIGATION")
    print("=" * 80)
    
    # Define all paths
    base_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_v2_physics")
    
    # Check ALL scaling files
    print("\n1. CHECKING ALL SCALING FILES:")
    print("-" * 40)
    
    scaling_files = {
        'enhanced': base_path / "scaling_factors_enhanced.pt",
        'inference': base_path / "scaling_factors_inference.pt",
        'target': base_path / "scaling_factors_target.pt",
        'surface_npy': base_path / "surface_scaling_factors_inference.npy",
        'enhanced_npy': base_path / "scaling_factors_enhanced.npy"
    }
    
    all_scalings = {}
    
    for name, path in scaling_files.items():
        if path.exists():
            if path.suffix == '.pt':
                data = torch.load(path, map_location='cpu')
            else:
                data = np.load(path)
            
            all_scalings[name] = data
            
            print(f"\n{name}: {path.name}")
            print(f"  Shape: {data.shape}")
            if len(data.shape) == 2:
                print(f"  Mean row: {data[0]}")
                print(f"  Std row:  {data[1]}")
                
                # Check pressure component specifically
                print(f"  Pressure (index 0):")
                print(f"    Mean: {data[0][0]:.6f}")
                print(f"    Std:  {data[1][0]:.6f}")
    
    return all_scalings

def simulate_normalization_pipeline(all_scalings):
    """
    Simulate what happens during normalization and unnormalization
    """
    print("\n" + "=" * 80)
    print("2. SIMULATING NORMALIZATION PIPELINE")
    print("-" * 40)
    
    # Test with actual coarse values from your test data
    test_coarse_pressure = -0.1257  # From run_454
    
    print(f"\nStarting with coarse pressure: {test_coarse_pressure:.4f}")
    
    # During INFERENCE, what scaling is used for input normalization?
    if 'inference' in all_scalings:
        input_scaling = all_scalings['inference']
        print(f"\nInput normalization using 'inference' scaling:")
        print(f"  Mean: {input_scaling[0][0]:.6f}, Std: {input_scaling[1][0]:.6f}")
        
        normalized_input = (test_coarse_pressure - input_scaling[0][0]) / input_scaling[1][0]
        print(f"  Normalized input: {normalized_input:.4f}")
    
    # What would a passthrough model output?
    # With 91.5% residual weight
    model_output = 0.915 * normalized_input + 0.0269  # Adding pressure bias
    print(f"\nModel output (91.5% passthrough + bias): {model_output:.4f}")
    
    # During UNNORMALIZATION, what scaling is used?
    if 'surface_npy' in all_scalings:
        output_scaling = all_scalings['surface_npy']
        print(f"\nOutput unnormalization using 'surface_npy' scaling:")
        print(f"  Mean: {output_scaling[0][0]:.6f}, Std: {output_scaling[1][0]:.6f}")
        
        unnormalized_output = model_output * output_scaling[1][0] + output_scaling[0][0]
        print(f"  Unnormalized output: {unnormalized_output:.4f}")
        print(f"  Expected: ~-0.126 (fine pressure)")
        print(f"  Actually getting: +0.0586")
        
        # What normalized value would give us +0.0586?
        needed_normalized = (0.0586 - output_scaling[0][0]) / output_scaling[1][0]
        print(f"\n  To get +0.0586, need normalized: {needed_normalized:.4f}")

def check_training_vs_test_scaling():
    """
    Check if training used different scaling than testing
    """
    print("\n" + "=" * 80)
    print("3. TRAINING VS TESTING SCALING CHECK")
    print("-" * 40)
    
    # During TRAINING with 8 features
    enhanced_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_v2_physics/scaling_factors_enhanced.pt")
    
    if enhanced_path.exists():
        enhanced = torch.load(enhanced_path, map_location='cpu')
        print("\nTraining scaling (8 features):")
        print(f"  Fine features (0:4) for targets:")
        print(f"    Pressure mean: {enhanced[0][0]:.6f}, std: {enhanced[1][0]:.6f}")
        print(f"  Coarse features (4:8) for inputs:")
        print(f"    Pressure mean: {enhanced[0][4]:.6f}, std: {enhanced[1][4]:.6f}")
        
        # Check if there's a mismatch
        if enhanced.shape[1] == 8:
            input_train_mean = enhanced[0][4]
            input_train_std = enhanced[1][4]
            output_train_mean = enhanced[0][0]
            output_train_std = enhanced[1][0]
            
            print("\n⚠️ CRITICAL CHECK:")
            print(f"  During training:")
            print(f"    Input normalized with: mean={input_train_mean:.6f}, std={input_train_std:.6f}")
            print(f"    Output unnormalized with: mean={output_train_mean:.6f}, std={output_train_std:.6f}")
            
            # Check what testing is using
            test_input = torch.load(Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_v2_physics/scaling_factors_inference.pt"), map_location='cpu')
            test_output = np.load(Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_v2_physics/surface_scaling_factors_inference.npy"))
            
            print(f"\n  During testing:")
            print(f"    Input normalized with: mean={test_input[0][0]:.6f}, std={test_input[1][0]:.6f}")
            print(f"    Output unnormalized with: mean={test_output[0][0]:.6f}, std={test_output[1][0]:.6f}")
            
            # Check if they match
            input_match = np.allclose(input_train_mean, test_input[0][0]) and np.allclose(input_train_std, test_input[1][0])
            output_match = np.allclose(output_train_mean, test_output[0][0]) and np.allclose(output_train_std, test_output[1][0])
            
            if not input_match:
                print("\n❌ INPUT SCALING MISMATCH DETECTED!")
                print("   Training and testing use different input normalization!")
            
            if not output_match:
                print("\n❌ OUTPUT SCALING MISMATCH DETECTED!")
                print("   Training and testing use different output unnormalization!")

def propose_fix():
    """
    Propose the correct scaling setup
    """
    print("\n" + "=" * 80)
    print("4. PROPOSED FIX")
    print("-" * 40)
    
    print("""
The issue is likely that during training, the model learned with:
- Inputs normalized using COARSE statistics (indices 4:8 from enhanced)
- Outputs compared to targets normalized with FINE statistics (indices 0:4)

But during testing, you might be:
- Normalizing inputs with wrong statistics
- Unnormalizing outputs with wrong statistics

THE FIX:
--------
In your test_enhanced.py, ensure the scaling is loaded correctly:

# For normalizing the coarse INPUT:
enhanced_scaling = torch.load('scaling_factors_enhanced.pt')
input_mean = enhanced_scaling[0, 4:8]  # COARSE features from training
input_std = enhanced_scaling[1, 4:8]

# For unnormalizing the PREDICTIONS:
output_mean = enhanced_scaling[0, :4]  # FINE features from training
output_std = enhanced_scaling[1, :4]

# The model predicts normalized FINE values, so unnormalize with FINE stats:
predictions = model_output * output_std + output_mean
""")

def main():
    all_scalings = investigate_all_scaling()
    simulate_normalization_pipeline(all_scalings)
    check_training_vs_test_scaling()
    propose_fix()
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    
    print("""
KEY FINDINGS:
1. Your model was trained with 8-feature enhanced scaling
2. Testing must use the EXACT same scaling as training
3. Input normalization: Use coarse stats (indices 4:8)
4. Output unnormalization: Use fine stats (indices 0:4)

The model is likely correct, but the scaling pipeline is mismatched.
""")

if __name__ == "__main__":
    main()
