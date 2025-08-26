#!/usr/bin/env python3
"""
Debug script to check what features are being passed to the model
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, '/workspace/PhysicsNeMo')
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

# Import test_enhanced to monkey-patch it
import test_enhanced
from enhanced_domino_model import DoMINOEnhanced

# Store the original forward function
original_forward = DoMINOEnhanced._forward_surface_enhanced

def debug_forward_surface_enhanced(self, inputs_dict):
    """Debug version to see what's being passed."""
    
    surface_fields = inputs_dict["surface_fields"]
    
    print("\n" + "="*60)
    print("🔍 DEBUG: Enhanced Model Forward Pass")
    print("="*60)
    
    print(f"Surface fields shape: {surface_fields.shape}")
    print(f"Surface fields dtype: {surface_fields.dtype}")
    
    # Check the actual values
    if len(surface_fields.shape) >= 2:
        n_features = surface_fields.shape[-1]
        print(f"Number of features: {n_features}")
        
        # Print statistics for each feature
        for i in range(min(n_features, 8)):
            if len(surface_fields.shape) == 3:
                feature_data = surface_fields[0, :, i]  # Batch dimension
            else:
                feature_data = surface_fields[:, i]
            
            print(f"\nFeature {i}:")
            print(f"  Mean: {feature_data.mean():.6f}")
            print(f"  Std:  {feature_data.std():.6f}")
            print(f"  Min:  {feature_data.min():.6f}")
            print(f"  Max:  {feature_data.max():.6f}")
    
    # Check what the model thinks it's doing
    if n_features == 8:
        print("\n⚠️ TRAINING MODE DETECTED (8 features)")
        print("  Features 0-3: Fine (target)")
        print("  Features 4-7: Coarse (input)")
        coarse_features = surface_fields[..., 4:8]
        print(f"  Extracting coarse features: shape {coarse_features.shape}")
    elif n_features == 4:
        print("\n✅ INFERENCE MODE DETECTED (4 features)")
        print("  All 4 features should be coarse input")
        coarse_features = surface_fields
    else:
        print(f"\n❌ UNEXPECTED NUMBER OF FEATURES: {n_features}")
        coarse_features = surface_fields
    
    # Call original with debug info
    result = original_forward(self, inputs_dict)
    
    # Check the output
    if result is not None:
        print(f"\nModel output shape: {result.shape}")
        print(f"Output statistics:")
        print(f"  Mean: {result.mean():.6f}")
        print(f"  Std:  {result.std():.6f}")
        print(f"  Min:  {result.min():.6f}")
        print(f"  Max:  {result.max():.6f}")
    
    print("="*60 + "\n")
    
    return result

# Apply the debug patch
DoMINOEnhanced._forward_surface_enhanced = debug_forward_surface_enhanced

# Also debug what's being loaded
original_load_coarse = test_enhanced.load_coarse_vtp_data_robust

def debug_load_coarse(vtp_path, surface_variables):
    """Debug version of coarse data loading."""
    print("\n🔍 DEBUG: Loading Coarse Data")
    print(f"Path: {vtp_path}")
    
    result = original_load_coarse(vtp_path, surface_variables)
    
    print(f"\nCoarse data statistics:")
    fields = result['fields']
    print(f"  Shape: {fields.shape}")
    for i in range(min(4, fields.shape[1])):
        print(f"  Feature {i}: mean={fields[:, i].mean():.6f}, std={fields[:, i].std():.6f}")
    
    return result

test_enhanced.load_coarse_vtp_data_robust = debug_load_coarse

# Run the test
if __name__ == "__main__":
    print("Running test with debug output...")
    # Don't call main directly - it needs Hydra decoration
    # Instead, just run a focused test on one case
    
    from pathlib import Path
    from omegaconf import DictConfig
    import pyvista as pv
    from physicsnemo.distributed import DistributedManager
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Create a minimal config
    cfg = DictConfig({
        'eval': {
            'coarse_test_path': '/data/ahmed_data/organized/test/coarse/',
            'test_path': '/data/ahmed_data/organized/test/fine/',
            'save_path': '/data/ahmed_data/debug_test/',
            'checkpoint_name': 'DoMINOEnhanced.0.499.pt',
            'scaling_param_path': 'outputs/Ahmed_Dataset/enhanced_1',
            'stencil_size': 7
        },
        'model': {
            'model_type': 'surface',
            'interp_res': [128, 64, 64],
            'enhanced_model': {
                'surface_input_features': 8,
                'debug': True
            }
        },
        'variables': {
            'surface': {
                'solution': {
                    'pMean': 'scalar',
                    'wallShearStressMean': 'vector'
                }
            }
        },
        'data': {
            'bounding_box_surface': {
                'min': [-1.5, -0.4, 0.0],
                'max': [1.0, 0.4, 0.5]
            }
        },
        'resume_dir': 'outputs/Ahmed_Dataset/enhanced_1/models'
    })
    
    print("\n" + "="*60)
    print("DEBUG TEST - Single Case Analysis")
    print("="*60)
    
    # Load one test case to debug
    test_case = "run_451"
    coarse_vtp = f"/data/ahmed_data/organized/test/coarse/{test_case}/boundary_451.vtp"
    fine_vtp = f"/data/ahmed_data/organized/test/fine/{test_case}/boundary_451.vtp"
    
    print(f"\nTest case: {test_case}")
    print(f"Coarse: {coarse_vtp}")
    print(f"Fine: {fine_vtp}")
    
    # Load the data
    surface_variable_names = ['pMean', 'wallShearStressMean']
    
    print("\n1. Loading coarse data...")
    coarse_data = test_enhanced.load_coarse_vtp_data_robust(coarse_vtp, surface_variable_names)
    
    print("\n2. Loading fine data...")
    fine_data = test_enhanced.load_fine_vtp_data_and_interpolate(
        fine_vtp, surface_variable_names, coarse_data['coordinates']
    )
    
    print("\n3. Data comparison:")
    print(f"   Coarse shape: {coarse_data['fields'].shape}")
    print(f"   Fine shape: {fine_data['fields'].shape}")
    
    # Compare statistics
    print("\n4. Field statistics:")
    print("   Pressure (feature 0):")
    print(f"     Coarse: mean={coarse_data['fields'][:,0].mean():.6f}, std={coarse_data['fields'][:,0].std():.6f}")
    print(f"     Fine:   mean={fine_data['fields'][:,0].mean():.6f}, std={fine_data['fields'][:,0].std():.6f}")
    
    print("\n   Shear X (feature 1):")
    print(f"     Coarse: mean={coarse_data['fields'][:,1].mean():.6f}, std={coarse_data['fields'][:,1].std():.6f}")
    print(f"     Fine:   mean={fine_data['fields'][:,1].mean():.6f}, std={fine_data['fields'][:,1].std():.6f}")
    
    # Check what would be passed to model
    print("\n5. What gets passed to model during inference:")
    print(f"   Only coarse fields: shape {coarse_data['fields'].shape}")
    print(f"   These are the 4 input features")
    
    # Load and check the model
    print("\n6. Loading model...")
    model_path = "outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt"
    
    if Path(model_path).exists():
        print(f"   Found checkpoint: {model_path}")
        
        # Create the model
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters=cfg.model
        ).to(dist.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=dist.device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("   Model loaded successfully")
        
        # Check coarse_to_fine model
        if hasattr(model, 'coarse_to_fine_model'):
            print("\n7. Coarse-to-fine model analysis:")
            total_params = 0
            for name, param in model.coarse_to_fine_model.named_parameters():
                total_params += param.numel()
                if 'weight' in name and 'output' in name:
                    print(f"   {name}:")
                    print(f"     Shape: {param.shape}")
                    print(f"     Mean: {param.mean():.6f}")
                    print(f"     Std:  {param.std():.6f}")
                    if param.std() < 1e-5:
                        print("     ⚠️ WARNING: Very low std - possibly not trained!")
            print(f"   Total parameters: {total_params:,}")
    else:
        print(f"   ❌ Checkpoint not found: {model_path}")
    
    print("\n" + "="*60)
    print("Debug complete!")
    print("="*60)