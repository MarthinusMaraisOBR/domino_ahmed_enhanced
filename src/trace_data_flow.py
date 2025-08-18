#!/usr/bin/env python3
"""
Step 3: Trace the exact data flow from training to inference
This is the most critical diagnostic - we'll trace exactly what happens to the data
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pyvista as pv

def trace_training_data_flow():
    """Trace exactly what happens during training."""
    
    print("="*80)
    print("STEP 3: TRACING DATA FLOW - TRAINING")
    print("="*80)
    
    # Load actual training data
    train_dir = Path("/data/ahmed_data/processed/train/")
    sample_file = list(train_dir.glob("*.npy"))[0]
    
    print(f"\n📊 Loading training sample: {sample_file.name}")
    data = np.load(sample_file, allow_pickle=True).item()
    
    surface_fields = data['surface_fields']
    print(f"Raw surface_fields shape: {surface_fields.shape}")
    
    if surface_fields.shape[1] == 8:
        # Exactly what the model sees
        print("\n🔍 TRAINING DATA BREAKDOWN:")
        print("="*60)
        
        # Features 0-3: FINE (target)
        fine_features = surface_fields[:, :4]
        print(f"\nFINE FEATURES (indices 0-3) - TARGET:")
        print(f"  Shape: {fine_features.shape}")
        print(f"  Pressure (idx 0): [{fine_features[:, 0].min():.6f}, {fine_features[:, 0].max():.6f}]")
        print(f"  Shear X  (idx 1): [{fine_features[:, 1].min():.6f}, {fine_features[:, 1].max():.6f}]")
        print(f"  Shear Y  (idx 2): [{fine_features[:, 2].min():.6f}, {fine_features[:, 2].max():.6f}]")
        print(f"  Shear Z  (idx 3): [{fine_features[:, 3].min():.6f}, {fine_features[:, 3].max():.6f}]")
        
        # Features 4-7: COARSE (input)
        coarse_features = surface_fields[:, 4:8]
        print(f"\nCOARSE FEATURES (indices 4-7) - INPUT:")
        print(f"  Shape: {coarse_features.shape}")
        print(f"  Pressure (idx 4): [{coarse_features[:, 0].min():.6f}, {coarse_features[:, 0].max():.6f}]")
        print(f"  Shear X  (idx 5): [{coarse_features[:, 1].min():.6f}, {coarse_features[:, 1].max():.6f}]")
        print(f"  Shear Y  (idx 6): [{coarse_features[:, 2].min():.6f}, {coarse_features[:, 2].max():.6f}]")
        print(f"  Shear Z  (idx 7): [{coarse_features[:, 3].min():.6f}, {coarse_features[:, 3].max():.6f}]")
        
        # Check difference
        print(f"\n📊 FINE vs COARSE COMPARISON:")
        pressure_diff = fine_features[:, 0] - coarse_features[:, 0]
        print(f"  Pressure difference stats:")
        print(f"    Mean: {pressure_diff.mean():.6f}")
        print(f"    Std:  {pressure_diff.std():.6f}")
        print(f"    Min:  {pressure_diff.min():.6f}")
        print(f"    Max:  {pressure_diff.max():.6f}")
        
        # What the model should learn
        print(f"\n🎯 MODEL OBJECTIVE:")
        print(f"  Input:  Coarse features (indices 4-7)")
        print(f"  Output: Fine features (indices 0-3)")
        print(f"  Learn:  Coarse → Fine mapping")
        
        return fine_features, coarse_features
    else:
        print(f"❌ Unexpected shape: {surface_fields.shape}")
        return None, None

def trace_inference_data_flow():
    """Trace exactly what happens during inference."""
    
    print("\n" + "="*80)
    print("TRACING DATA FLOW - INFERENCE")
    print("="*80)
    
    # Check test data
    test_coarse_path = Path("/data/ahmed_data/organized/test/coarse/run_451/boundary_451.vtp")
    test_fine_path = Path("/data/ahmed_data/organized/test/fine/run_451/boundary_451.vtp")
    
    if not test_coarse_path.exists():
        print(f"❌ Test coarse file not found: {test_coarse_path}")
        return None, None
    
    print(f"\n📊 Loading test data:")
    print(f"  Coarse: {test_coarse_path}")
    print(f"  Fine:   {test_fine_path}")
    
    # Load coarse test data
    coarse_mesh = pv.read(str(test_coarse_path))
    print(f"\nCoarse mesh arrays: {coarse_mesh.array_names}")
    
    # Find coarse pressure field
    coarse_pressure = None
    for name in ['p', 'pressure', 'Pressure', 'pMean']:
        if name in coarse_mesh.point_data:
            coarse_pressure = np.array(coarse_mesh.point_data[name])
            print(f"  Found coarse pressure in point_data['{name}']")
            break
        elif name in coarse_mesh.cell_data:
            coarse_pressure = np.array(coarse_mesh.cell_data[name])
            print(f"  Found coarse pressure in cell_data['{name}']")
            break
    
    if coarse_pressure is not None:
        print(f"\n🔍 INFERENCE INPUT (COARSE):")
        print(f"  Shape: {coarse_pressure.shape}")
        print(f"  Range: [{coarse_pressure.min():.6f}, {coarse_pressure.max():.6f}]")
        print(f"  Mean:  {coarse_pressure.mean():.6f}")
        print(f"  Std:   {coarse_pressure.std():.6f}")
    
    # Load fine test data (ground truth)
    if test_fine_path.exists():
        fine_mesh = pv.read(str(test_fine_path))
        
        fine_pressure = None
        for name in ['pMean', 'p', 'pressure', 'Pressure']:
            if name in fine_mesh.point_data:
                fine_pressure = np.array(fine_mesh.point_data[name])
                print(f"\n  Found fine pressure in point_data['{name}']")
                break
            elif name in fine_mesh.cell_data:
                fine_pressure = np.array(fine_mesh.cell_data[name])
                print(f"\n  Found fine pressure in cell_data['{name}']")
                break
        
        if fine_pressure is not None:
            print(f"\n🎯 EXPECTED OUTPUT (FINE):")
            print(f"  Shape: {fine_pressure.shape}")
            print(f"  Range: [{fine_pressure.min():.6f}, {fine_pressure.max():.6f}]")
            print(f"  Mean:  {fine_pressure.mean():.6f}")
            print(f"  Std:   {fine_pressure.std():.6f}")
    
    # Check predictions
    pred_path = Path("/data/ahmed_data/predictions/boundary_451_comprehensive_comparison.vtp")
    if pred_path.exists():
        pred_mesh = pv.read(str(pred_path))
        
        if 'Predicted_Pressure' in pred_mesh.array_names:
            pred_pressure = pred_mesh['Predicted_Pressure']
            
            print(f"\n❌ ACTUAL OUTPUT (PREDICTED):")
            print(f"  Shape: {pred_pressure.shape}")
            print(f"  Range: [{pred_pressure.min():.6f}, {pred_pressure.max():.6f}]")
            print(f"  Mean:  {pred_pressure.mean():.6f}")
            print(f"  Std:   {pred_pressure.std():.6f}")
            
            # Diagnose the problem
            print(f"\n⚠️  PROBLEM DIAGNOSIS:")
            if pred_pressure.mean() < 0 and coarse_pressure.mean() > 0:
                print("  - Predictions have wrong sign!")
            if pred_pressure.std() < coarse_pressure.std() * 0.1:
                print("  - Predictions have too low variance!")
            if abs(pred_pressure.mean()) > abs(coarse_pressure.mean()) * 10:
                print("  - Predictions are scaled incorrectly!")
    
    return coarse_pressure, fine_pressure

def check_model_internals():
    """Check what's happening inside the model."""
    
    print("\n" + "="*80)
    print("CHECKING MODEL INTERNALS")
    print("="*80)
    
    try:
        from enhanced_domino_model import DoMINOEnhanced
        
        # Load checkpoint
        ckpt_path = Path("outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt")
        
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return
        
        print(f"\n📊 Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check for enhanced model keys
        enhanced_keys = [k for k in checkpoint.keys() if 'coarse_to_fine' in k]
        print(f"  Found {len(enhanced_keys)} coarse_to_fine keys")
        
        # Check specific components
        if 'coarse_to_fine_model.residual_projection.weight' in checkpoint:
            residual_weight = checkpoint['coarse_to_fine_model.residual_projection.weight']
            print(f"\n🔍 Residual Projection Analysis:")
            print(f"  Shape: {residual_weight.shape}")
            print(f"  Norm: {residual_weight.norm():.4f}")
            print(f"  Mean: {residual_weight.mean():.6f}")
            print(f"  Std:  {residual_weight.std():.6f}")
            
            # Check if it's near identity
            if residual_weight.shape[0] == residual_weight.shape[1]:
                identity_error = (residual_weight - torch.eye(residual_weight.shape[0])).norm()
                print(f"  Distance from identity: {identity_error:.4f}")
                
                if identity_error < 0.5:
                    print("  ⚠️ WARNING: Residual is nearly identity - just passing input through!")
        
        # Check output projection
        if 'coarse_to_fine_model.output_projection.0.weight_v' in checkpoint:
            output_weight = checkpoint['coarse_to_fine_model.output_projection.0.weight_v']
            print(f"\n🔍 Output Projection Analysis:")
            print(f"  Shape: {output_weight.shape}")
            print(f"  Norm: {output_weight.norm():.4f}")
            print(f"  Mean: {output_weight.mean():.6f}")
            print(f"  Std:  {output_weight.std():.6f}")
            
            if output_weight.std() < 0.01:
                print("  ⚠️ WARNING: Output weights have low variance - may output constants!")
        
        # Test the model behavior
        print(f"\n🧪 Testing Model Behavior:")
        
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters={
                'enhanced_model': {'surface_input_features': 8},
                'activation': 'relu',
                'model_type': 'surface',
                'interp_res': [32, 32, 32]
            }
        )
        
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Test with controlled input
        with torch.no_grad():
            # Create test data - only coarse features
            test_coarse = torch.tensor([
                [0.5, 0.01, 0.01, 0.01],   # Moderate positive pressure
                [-0.5, -0.01, -0.01, -0.01], # Moderate negative pressure
            ]).unsqueeze(0).repeat(1, 100, 1)  # (1, 100, 4)
            
            dummy_geom = torch.randn(1, 100, 3)
            
            test_data = {
                'surface_fields': test_coarse,  # Only 4 features for inference
                'geometry_coordinates': dummy_geom,
                'surface_mesh_centers': torch.randn(1, 100, 3),
                'surf_grid': torch.randn(1, 32, 32, 32, 3),
                'sdf_surf_grid': torch.randn(1, 32, 32, 32),
            }
            
            _, output = model(test_data)
            
            if output is not None:
                output_np = output[0].numpy()  # Remove batch
                
                print(f"\nInput coarse pressure: [0.5, -0.5]")
                print(f"Output predictions:")
                print(f"  Point 0: {output_np[0]}")
                print(f"  Point 1: {output_np[1]}")
                print(f"  Mean: {output_np.mean(axis=0)}")
                print(f"  Std:  {output_np.std(axis=0)}")
                
                # Check if output is reasonable
                if np.all(output_np[:, 0] < 0):
                    print("  ❌ All pressures negative regardless of input!")
                if output_np.std() < 0.01:
                    print("  ❌ Output has no variation!")
                    
    except Exception as e:
        print(f"❌ Error checking model: {str(e)}")
        import traceback
        traceback.print_exc()

def identify_root_cause():
    """Synthesize findings and identify root cause."""
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("\n🔍 KEY OBSERVATIONS:")
    print("1. Model trains and loss decreases ✅")
    print("2. But predictions are worse than baseline ❌")
    print("3. All predictions are negative ❌")
    print("4. Predictions have low variance ❌")
    
    print("\n💡 MOST LIKELY CAUSES:")
    print("\n1. **Residual Connection Dominance**")
    print("   The residual connection might be too strong, causing:")
    print("   - Model just passes coarse through with minor modifications")
    print("   - If residual has wrong sign/scale, all outputs are wrong")
    print("   FIX: Remove or reduce residual connection weight")
    
    print("\n2. **Feature Index Mismatch**")
    print("   During training vs inference, features might be swapped:")
    print("   - Training: expects coarse at indices 4-7")
    print("   - Inference: provides coarse at indices 0-3")
    print("   FIX: Verify enhanced_domino_model._forward_surface_enhanced()")
    
    print("\n3. **Scaling Factor Mismatch**")
    print("   Different scaling applied during training vs inference:")
    print("   - Training uses 8-feature scaling factors")
    print("   - Inference uses 4-feature scaling factors")
    print("   FIX: Ensure consistent scaling between train and test")
    
    print("\n4. **Wrong Training Objective**")
    print("   Model might be learning the wrong mapping:")
    print("   - Should learn: coarse → fine")
    print("   - Might be learning: fine → coarse or coarse → coarse")
    print("   FIX: Check compute_loss_dict_enhanced in train.py")

def proposed_fixes():
    """Generate specific code fixes."""
    
    print("\n" + "="*80)
    print("PROPOSED FIXES")
    print("="*80)
    
    print("\n🔧 FIX 1: Disable Residual Connection")
    print("-"*40)
    print("In enhanced_domino_model.py, CoarseToFineModel.__init__:")
    print("```python")
    print("# Change this:")
    print("use_residual: bool = True")
    print("# To this:")
    print("use_residual: bool = False  # DISABLED - was causing wrong predictions")
    print("```")
    
    print("\n🔧 FIX 2: Verify Feature Extraction")
    print("-"*40)
    print("In enhanced_domino_model.py, _forward_surface_enhanced:")
    print("```python")
    print("# Add debugging:")
    print("if surface_fields.shape[-1] == 8:")
    print("    fine_features = surface_fields[..., :4]")
    print("    coarse_features = surface_fields[..., 4:8]")
    print("    print(f'Training - Fine mean: {fine_features.mean():.4f}')")
    print("    print(f'Training - Coarse mean: {coarse_features.mean():.4f}')")
    print("else:")
    print("    coarse_features = surface_fields  # This should be 4 features")
    print("    print(f'Inference - Coarse mean: {coarse_features.mean():.4f}')")
    print("```")
    
    print("\n🔧 FIX 3: Fix Scaling Factors")
    print("-"*40)
    print("Create separate inference scaling factors:")
    print("```python")
    print("# In test_enhanced.py, before loading model:")
    print("# Extract only coarse scaling factors for inference")
    print("full_factors = np.load('surface_scaling_factors.npy')")
    print("if full_factors.shape[1] == 8:")
    print("    # Use coarse factors (indices 4-7) for inference")
    print("    inference_factors = full_factors[:, 4:8]")
    print("else:")
    print("    inference_factors = full_factors")
    print("```")
    
    print("\n🔧 FIX 4: Simplify Architecture")
    print("-"*40)
    print("Consider a simpler model first:")
    print("```python")
    print("# Reduce hidden layers:")
    print("hidden_layers=[128, 128]  # Was [512, 512, 512]")
    print("# Disable spectral features:")
    print("use_spectral=False  # Was True")
    print("# Disable residual:")
    print("use_residual=False  # Was True")
    print("```")

def main():
    """Run complete diagnostic."""
    
    print("\n🔬 COMPLETE DATA FLOW ANALYSIS\n")
    
    # Trace training flow
    train_fine, train_coarse = trace_training_data_flow()
    
    # Trace inference flow
    test_coarse, test_fine = trace_inference_data_flow()
    
    # Check model internals
    check_model_internals()
    
    # Identify root cause
    identify_root_cause()
    
    # Propose fixes
    proposed_fixes()
    
    print("\n" + "="*80)
    print("RECOMMENDED ACTION PLAN")
    print("="*80)
    print("\n1. First, disable residual connection (most likely culprit)")
    print("2. Add debug prints to verify feature extraction")
    print("3. Retrain with simpler architecture")
    print("4. Test on training data to verify model can reproduce training samples")
    print("5. If still failing, check data preprocessing pipeline")
    
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
