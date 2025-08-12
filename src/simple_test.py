#!/usr/bin/env python3
"""
Simple test script to verify Enhanced DoMINO model works
Run this before full training to catch any issues early
"""

import torch
import numpy as np
from pathlib import Path
import sys

def test_enhanced_model():
    """Simple test of the enhanced model with dummy data"""
    
    print("="*60)
    print("Simple Enhanced DoMINO Model Test")
    print("="*60)
    
    try:
        # Import the enhanced model
        print("\n1. Importing Enhanced Model...")
        from enhanced_domino_model import DoMINOEnhanced
        print("   ✅ Import successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    try:
        # Create a minimal model
        print("\n2. Creating Enhanced Model...")
        model_params = {
            'activation': 'relu',
            'model_type': 'surface',
            'interp_res': [32, 32, 32],  # Small for testing
            'positional_encoding': False,
            'use_sdf_in_basis_func': True,
            'surface_points_sample': 100,
            'geom_points_sample': 1000,
            'num_surface_neighbors': 7,
            'normalization': 'min_max_scaling',
            'surf_loss_scaling': 1.0,
            'integral_loss_scaling_factor': 100,
            'enhanced_model': {
                'surface_input_features': 8,
                'coarse_to_fine': {
                    'hidden_layers': [128, 128],  # Smaller for testing
                    'use_spectral': True,
                    'use_residual': True,
                }
            },
            # Add other required parameters with defaults
            'use_surface_normals': True,
            'use_surface_area': True,
            'surface_neighbors': True,
            'encode_parameters': False,
            'geometry_encoding_type': 'both',
            'solution_calculation_mode': 'two-loop',
            'geometry_rep': {
                'geo_conv': {
                    'base_neurons': 32,
                    'base_neurons_out': 1,
                    'volume_radii': [0.1, 0.5, 2.5, 5.0],
                    'surface_radii': [0.01, 0.05, 0.1],
                    'hops': 1,
                    'activation': 'relu'
                },
                'geo_processor': {
                    'base_filters': 8,
                    'activation': 'relu'
                },
                'geo_processor_sdf': {
                    'base_filters': 8
                }
            },
            'nn_basis_functions': {
                'base_layer': 128,  # Smaller for testing
                'fourier_features': False,
                'num_modes': 5,
                'activation': 'relu'
            },
            'local_point_conv': {
                'activation': 'relu'
            },
            'aggregation_model': {
                'base_layer': 128,  # Smaller for testing
                'activation': 'relu'
            },
            'position_encoder': {
                'base_neurons': 128  # Smaller for testing
            },
            'geometry_local': {
                'volume_neighbors_in_radius': [64, 128, 256],
                'surface_neighbors_in_radius': [64, 128, 256],
                'volume_radii': [0.05, 0.25, 1.0],
                'surface_radii': [0.05, 0.25, 1.0],
                'base_layer': 128  # Smaller for testing
            },
            'parameter_model': {
                'base_layer': 128,  # Smaller for testing
                'scaling_params': [1.0, 1.0],
                'fourier_features': False,
                'num_modes': 5
            }
        }
        
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters=model_params
        )
        print("   ✅ Model created successfully")
        
        # Check if coarse-to-fine model exists
        if hasattr(model, 'coarse_to_fine_model'):
            print("   ✅ Coarse-to-fine model initialized")
        else:
            print("   ❌ Coarse-to-fine model not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Create dummy input data
        print("\n3. Creating Dummy Input Data...")
        batch_size = 1
        num_points = 100
        grid_res = model_params['interp_res']
        
        # Training mode: 8 features (4 fine + 4 coarse)
        inputs_train = {
            'geometry_coordinates': torch.randn(batch_size, 1000, 3),
            'surface_mesh_centers': torch.randn(batch_size, num_points, 3),
            'surface_fields': torch.randn(batch_size, num_points, 8),  # 8 features!
            'surface_normals': torch.randn(batch_size, num_points, 3),
            'surface_areas': torch.rand(batch_size, num_points),
            'surf_grid': torch.randn(batch_size, *grid_res, 3),
            'sdf_surf_grid': torch.randn(batch_size, *grid_res),
            'surface_min_max': torch.tensor([[[0, 0, 0], [1, 1, 1]]], dtype=torch.float32),
            'stream_velocity': torch.ones(batch_size, 1),
            'air_density': torch.ones(batch_size, 1),
        }
        
        # Add surface neighbors (required by model)
        inputs_train['surface_mesh_neighbors'] = torch.randn(batch_size, num_points, 6, 3)
        inputs_train['surface_neighbors_normals'] = torch.randn(batch_size, num_points, 6, 3)
        inputs_train['surface_neighbors_areas'] = torch.rand(batch_size, num_points, 6)
        inputs_train['pos_surface_center_of_mass'] = torch.randn(batch_size, num_points, 3)
        
        print(f"   Created training input with {inputs_train['surface_fields'].shape[-1]} surface features")
        
        # Inference mode: 4 features (coarse only)
        inputs_test = inputs_train.copy()
        inputs_test['surface_fields'] = torch.randn(batch_size, num_points, 4)  # 4 features!
        
        print(f"   Created inference input with {inputs_test['surface_fields'].shape[-1]} surface features")
        
    except Exception as e:
        print(f"   ❌ Input creation failed: {e}")
        return False
    
    try:
        # Test forward pass - training mode
        print("\n4. Testing Forward Pass (Training Mode)...")
        model.eval()  # Set to eval to avoid issues with batch norm
        with torch.no_grad():
            vol_out, surf_out = model(inputs_train)
        
        if surf_out is not None:
            print(f"   ✅ Training output shape: {surf_out.shape}")
            print(f"   Expected: torch.Size([{batch_size}, {num_points}, 4])")
            if surf_out.shape == (batch_size, num_points, 4):
                print("   ✅ Output shape correct!")
            else:
                print("   ⚠️  Output shape mismatch")
        else:
            print("   ❌ No surface output")
            return False
            
    except Exception as e:
        print(f"   ❌ Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test forward pass - inference mode
        print("\n5. Testing Forward Pass (Inference Mode)...")
        with torch.no_grad():
            vol_out, surf_out = model(inputs_test)
        
        if surf_out is not None:
            print(f"   ✅ Inference output shape: {surf_out.shape}")
            print(f"   Expected: torch.Size([{batch_size}, {num_points}, 4])")
            if surf_out.shape == (batch_size, num_points, 4):
                print("   ✅ Output shape correct!")
            else:
                print("   ⚠️  Output shape mismatch")
        else:
            print("   ❌ No surface output")
            return False
            
    except Exception as e:
        print(f"   ❌ Inference forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test loss calculation
        print("\n6. Testing Loss Calculation...")
        
        # Create target (fine features)
        target = torch.randn(batch_size, num_points, 4)
        
        # Simple MSE loss
        loss = torch.nn.functional.mse_loss(surf_out, target)
        print(f"   ✅ Loss calculated: {loss.item():.6f}")
        
        # Check if loss is reasonable
        if torch.isnan(loss):
            print("   ❌ Loss is NaN")
            return False
        elif torch.isinf(loss):
            print("   ❌ Loss is infinite")
            return False
        else:
            print("   ✅ Loss is valid")
            
    except Exception as e:
        print(f"   ❌ Loss calculation failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour Enhanced DoMINO model is working correctly!")
    print("You can now proceed with full training using:")
    print("  python train.py")
    
    return True


def main():
    """Main function"""
    success = test_enhanced_model()
    
    if not success:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Check enhanced_domino_model.py for syntax errors")
        print("2. Ensure all required parameters are in model_params")
        print("3. Verify DoMINOEnhanced inherits from DoMINO correctly")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
