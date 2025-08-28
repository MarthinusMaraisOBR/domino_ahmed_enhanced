#!/usr/bin/env python3
"""
Diagnose exactly what's happening in the prediction pipeline
"""

import torch
import numpy as np
from pathlib import Path
import h5py
import pyvista as pv

def check_model_output_directly():
    """
    Check what the model outputs BEFORE any unnormalization
    """
    print("=" * 80)
    print("CHECKING RAW MODEL OUTPUT")
    print("=" * 80)
    
    # Load model
    model_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Direct state dict'}")
    
    # Check model weights related to output
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print("\nOutput-related weights:")
    for key in state_dict.keys():
        if 'output' in key or 'final' in key or 'projection' in key:
            param = state_dict[key]
            if param.numel() < 20:  # Only print small tensors
                print(f"  {key}: {param}")
            else:
                print(f"  {key}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")

def test_unnormalization():
    """
    Test the unnormalization process step by step
    """
    print("\n" + "=" * 80)
    print("TESTING UNNORMALIZATION")
    print("=" * 80)
    
    # Load scaling factors
    scaling_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy")
    
    if scaling_path.exists():
        scaling = np.load(scaling_path)
        print(f"Scaling shape: {scaling.shape}")
        print(f"Scaling mean: {scaling[0]}")
        print(f"Scaling std: {scaling[1]}")
    else:
        print("Scaling factors not found!")
        return
    
    # Simulate normalized predictions (what model might output)
    print("\n--- Simulating unnormalization ---")
    
    # Test case 1: Zero normalized output
    normalized = np.zeros(4)
    unnormalized = normalized * scaling[1] + scaling[0]
    print(f"\nNormalized zeros -> Unnormalized: {unnormalized}")
    print(f"  Expected pressure: {unnormalized[0]:.4f} (should be ~{scaling[0][0]:.4f})")
    
    # Test case 2: Typical normalized output range (-2 to 2)
    normalized = np.array([-2.0, 0.5, 0.5, 0.5])
    unnormalized = normalized * scaling[1] + scaling[0]
    print(f"\nNormalized [-2,0.5,0.5,0.5] -> Unnormalized: {unnormalized}")
    print(f"  Pressure: {unnormalized[0]:.4f}")
    
    # What would give us -0.6 pressure?
    target_pressure = -0.6
    needed_normalized = (target_pressure - scaling[0][0]) / scaling[1][0]
    print(f"\nTo get pressure=-0.6, need normalized value: {needed_normalized:.4f}")
    
    # Check if this is reasonable
    if abs(needed_normalized) > 5:
        print(f"  ⚠️ WARNING: This is {abs(needed_normalized):.1f} standard deviations!")
        print(f"     Model is outputting extreme values!")

def check_scaling_mismatch():
    """
    Check if there's a mismatch between what scaling is used
    """
    print("\n" + "=" * 80)
    print("CHECKING SCALING MISMATCH")
    print("=" * 80)
    
    # Check all scaling files
    base_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1")
    
    scaling_files = {
        'enhanced': base_dir / "scaling_factors_enhanced.pt",
        'inference': base_dir / "scaling_factors_inference.pt", 
        'target': base_dir / "scaling_factors_target.pt",
        'surface': base_dir / "surface_scaling_factors_inference.npy"
    }
    
    for name, path in scaling_files.items():
        if path.exists():
            if path.suffix == '.pt':
                data = torch.load(path)
            else:
                data = np.load(path)
            
            print(f"\n{name} ({path.name}):")
            if hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}")
                if data.shape[0] == 2:  # mean/std format
                    print(f"  Mean: {data[0]}")
                    print(f"  Std:  {data[1]}")
                    
                    # Check for issues
                    if data.shape[1] == 8:
                        print(f"  Note: This has 8 features (enhanced training format)")
                        print(f"    Fine features (0:4): mean={data[0][:4]}")
                        print(f"    Coarse features (4:8): mean={data[0][4:]}")

def analyze_actual_prediction():
    """
    Load an actual prediction and analyze what went wrong
    """
    print("\n" + "=" * 80)
    print("ANALYZING ACTUAL PREDICTION")
    print("=" * 80)
    
    # Load a prediction VTP
    vtp_path = Path("/data/ahmed_data/predictions_v2/predictions_20250828_140715/run_451_prediction.vtp")
    
    if not vtp_path.exists():
        print(f"VTP not found: {vtp_path}")
        return
        
    mesh = pv.read(str(vtp_path))
    
    # Analyze fields
    for field_name in ['Coarse_Pressure', 'Fine_Pressure_GroundTruth_Interpolated', 'Predicted_Pressure']:
        if field_name in mesh.cell_data:
            data = mesh.cell_data[field_name]
            print(f"\n{field_name}:")
            print(f"  Mean: {data.mean():.4f}")
            print(f"  Std:  {data.std():.4f}")
            print(f"  Min:  {data.min():.4f}")
            print(f"  Max:  {data.max():.4f}")
            
            # Check distribution
            percentiles = np.percentile(data, [5, 25, 50, 75, 95])
            print(f"  Percentiles [5,25,50,75,95]: {percentiles}")

def propose_fixes():
    """
    Propose specific fixes based on the diagnosis
    """
    print("\n" + "=" * 80)
    print("PROPOSED FIXES")
    print("=" * 80)
    
    print("\nThe problem is clear: Predicted pressure is ~-0.6 instead of ~-0.11")
    print("This is a ~5x factor with wrong sign!")
    
    print("\nPossible causes:")
    print("1. ❌ Wrong scaling used for unnormalization")
    print("   - Using coarse scaling instead of fine")
    print("   - Or using wrong indices from 8-feature scaling")
    
    print("\n2. ❌ Model outputs are in wrong range")
    print("   - Model learned incorrect mapping")
    print("   - Residual connection has wrong weight")
    
    print("\n3. ❌ Sign flip in model")
    print("   - Model learned inverse relationship")
    
    print("\nIMPORTANT FIX TO TRY:")
    print("-" * 40)
    
    # Create a fix script
    fix_script = '''
# In test_enhanced.py, find the unnormalization line and replace:

# WRONG (might be using wrong scaling):
prediction_surf = unnormalize(
    prediction_surf.cpu().numpy(),
    surf_factors_inference[0],
    surf_factors_inference[1]
)

# CORRECT (use fine/target scaling for output):
# The model predicts FINE features, so unnormalize with FINE statistics
if surf_factors.shape[1] == 8:
    # Use fine features (first 4) from enhanced scaling
    fine_mean = surf_factors[0, :4]  
    fine_std = surf_factors[1, :4]
else:
    # Or load target scaling explicitly
    target_scaling = np.load("scaling_factors_target.pt")
    fine_mean = target_scaling[0]
    fine_std = target_scaling[1]

prediction_surf = unnormalize(
    prediction_surf.cpu().numpy(),
    fine_mean,
    fine_std
)
'''
    
    print(fix_script)
    
    print("\nAlternative fix if that doesn't work:")
    print("The model may have learned to predict normalized COARSE instead of FINE")
    print("In that case, check the training code for where targets are defined")

def main():
    check_model_output_directly()
    test_unnormalization()
    check_scaling_mismatch()
    analyze_actual_prediction()
    propose_fixes()

if __name__ == "__main__":
    main()
