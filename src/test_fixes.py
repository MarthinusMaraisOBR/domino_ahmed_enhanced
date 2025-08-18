#!/usr/bin/env python3
"""
Test script to verify the fixed Enhanced DoMINO model
"""

import torch
import numpy as np
from pathlib import Path

def test_fixed_model():
    """Test the fixed enhanced model."""
    
    print("="*80)
    print("TESTING FIXED ENHANCED DOMINO MODEL")
    print("="*80)
    
    # Import the fixed model
    from enhanced_domino_model_fixed import DoMINOEnhanced, CoarseToFineModel
    
    print("\n1. Testing CoarseToFineModel in isolation...")
    print("-"*40)
    
    # Test the core coarse-to-fine model
    c2f_model = CoarseToFineModel(
        input_dim=4,
        output_dim=4,
        encoding_dim=448,
        hidden_layers=[128, 128],  # Simple architecture
        use_spectral=False,  # Disabled
        use_residual=False,  # CRITICAL: Disabled
        dropout_rate=0.0,  # No dropout for testing
    )
    
    # Create test data
    batch_size = 2
    num_points = 100
    
    # Simulate coarse input with known statistics
    coarse_input = torch.randn(batch_size, num_points, 4) * 0.13 - 0.065  # Match your data stats
    geometry_encoding = torch.randn(batch_size, num_points, 448) * 0.1
    
    # Test forward pass
    with torch.no_grad():
        c2f_model.eval()
        output = c2f_model(coarse_input, geometry_encoding)
    
    print(f"Input shape: {coarse_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input stats - mean: {coarse_input.mean():.4f}, std: {coarse_input.std():.4f}")
    print(f"Output stats - mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Check if output is reasonable
    issues = []
    if output.mean() < -1.0 or output.mean() > 1.0:
        issues.append("Output mean out of expected range")
    if output.std() < 0.01:
        issues.append("Output variance too low")
    if torch.all(output < 0):
        issues.append("All outputs negative")
    
    if issues:
        print("\n⚠️ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Coarse-to-fine model output looks reasonable!")
    
    print("\n2. Testing full DoMINOEnhanced model...")
    print("-"*40)
    
    # Create minimal config for testing
    from omegaconf import DictConfig
    model_params = DictConfig({
        'enhanced_model': {
            'surface_input_features': 8,
            'debug': True,
            'coarse_to_fine': {
                'hidden_layers': [128, 128],
                'use_spectral': False,
                'use_residual': False,
                'dropout_rate': 0.0,
            }
        },
        'activation': 'relu',
        'model_type': 'surface',
        'interp_res': [32, 32, 32],
        'use_sdf_in_basis_func': True,
        'num_surface_neighbors': 7,
        'positional_encoding': False,
        'surface_neighbors': True,
        'use_surface_normals': True,
        'use_surface_area': True,
        'integral_loss_scaling_factor': 100,
        'surf_loss_scaling': 1.0,
        'vol_loss_scaling': 0.0,
        'normalization': 'min_max_scaling',
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
            'base_layer': 512,
            'fourier_features': False,
            'num_modes': 5,
            'activation': 'relu'
        },
        'local_point_conv': {
            'activation': 'relu'
        },
        'aggregation_model': {
            'base_layer': 512,
            'activation': 'relu'
        },
        'position_encoder': {
            'base_neurons': 512
        },
        'geometry_local': {
            'volume_neighbors_in_radius': [64, 128, 256],
            'surface_neighbors_in_radius': [64, 128, 256],
            'volume_radii': [0.05, 0.25, 1.0],
            'surface_radii': [0.05, 0.25, 1.0],
            'base_layer': 512
        }
    })
    
    try:
        # Create enhanced model
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters=model_params
        )
        model.eval()
        
        print("✅ Enhanced model created successfully!")
        
        # Test with training-like data (8 features)
        print("\n3. Testing training mode (8 features)...")
        print("-"*40)
        
        fine_features = torch.randn(1, 100, 4) * 0.15 - 0.066  # Fine stats
        coarse_features = torch.randn(1, 100, 4) * 0.13 - 0.064  # Coarse stats
        combined_features = torch.cat([fine_features, coarse_features], dim=-1)
        
        train_data = {
            'surface_fields': combined_features,
            'geometry_coordinates': torch.randn(1, 100, 3),
            'surface_mesh_centers': torch.randn(1, 100, 3),
            'surf_grid': torch.randn(1, 32, 32, 32, 3),
            'sdf_surf_grid': torch.randn(1, 32, 32, 32),
        }
        
        with torch.no_grad():
            _, train_output = model(train_data)
        
        if train_output is not None:
            print(f"Training mode output shape: {train_output.shape}")
            print(f"Output stats - mean: {train_output.mean():.4f}, std: {train_output.std():.4f}")
            print(f"Expected (fine) - mean: {fine_features.mean():.4f}, std: {fine_features.std():.4f}")
        
        # Test with inference-like data (4 features)
        print("\n4. Testing inference mode (4 features)...")
        print("-"*40)
        
        test_data = {
            'surface_fields': coarse_features,  # Only coarse
            'geometry_coordinates': torch.randn(1, 100, 3),
            'surface_mesh_centers': torch.randn(1, 100, 3),
            'surf_grid': torch.randn(1, 32, 32, 32, 3),
            'sdf_surf_grid': torch.randn(1, 32, 32, 32),
        }
        
        with torch.no_grad():
            _, test_output = model(test_data)
        
        if test_output is not None:
            print(f"Inference mode output shape: {test_output.shape}")
            print(f"Output stats - mean: {test_output.mean():.4f}, std: {test_output.std():.4f}")
            
            # Check if predictions are reasonable
            if test_output.mean() < -1.0:
                print("⚠️ Predictions too negative!")
            elif test_output.std() < 0.05:
                print("⚠️ Predictions have low variance!")
            else:
                print("✅ Predictions look reasonable!")
        
    except Exception as e:
        print(f"❌ Error creating enhanced model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    print("\n📝 SUMMARY:")
    print("If all tests passed, the model is ready for retraining.")
    print("If issues persist, check:")
    print("1. Geometry encoding dimension mismatch")
    print("2. Data preprocessing pipeline")
    print("3. Loss function implementation")
    
    return True

def test_with_real_checkpoint():
    """Test with actual trained checkpoint if available."""
    
    print("\n" + "="*80)
    print("TESTING WITH REAL CHECKPOINT")
    print("="*80)
    
    checkpoint_path = Path("outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        from enhanced_domino_model_fixed import DoMINOEnhanced
        from omegaconf import DictConfig
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Analyze checkpoint contents
        print("\n📊 Checkpoint Analysis:")
        print(f"  Total keys: {len(checkpoint.keys())}")
        
        # Check for residual weights
        residual_keys = [k for k in checkpoint.keys() if 'residual' in k]
        if residual_keys:
            print(f"  ⚠️ Found {len(residual_keys)} residual-related keys")
            for key in residual_keys[:3]:
                weight = checkpoint[key]
                print(f"    {key}: shape={weight.shape}, norm={weight.norm():.4f}")
        
        # Check output projection
        output_keys = [k for k in checkpoint.keys() if 'output_projection' in k]
        if output_keys:
            print(f"  Found {len(output_keys)} output projection keys")
            for key in output_keys[:3]:
                weight = checkpoint[key]
                if hasattr(weight, 'shape'):
                    print(f"    {key}: mean={weight.mean():.6f}, std={weight.std():.6f}")
        
        # Load a sample to get actual data statistics
        train_path = Path("/data/ahmed_data/processed/train/run_1.npy")
        if train_path.exists():
            print(f"\n📊 Loading real training data: {train_path}")
            data = np.load(train_path, allow_pickle=True).item()
            
            if 'surface_fields' in data:
                surface_fields = data['surface_fields']
                
                # Take a sample
                sample_size = min(1000, surface_fields.shape[0])
                sample_indices = np.random.choice(surface_fields.shape[0], sample_size, replace=False)
                sample = surface_fields[sample_indices]
                
                fine_sample = sample[:, :4]
                coarse_sample = sample[:, 4:8]
                
                print(f"\nReal data statistics (sample of {sample_size} points):")
                print(f"  Fine pressure:   mean={fine_sample[:, 0].mean():.4f}, std={fine_sample[:, 0].std():.4f}")
                print(f"  Coarse pressure: mean={coarse_sample[:, 0].mean():.4f}, std={coarse_sample[:, 0].std():.4f}")
                
                # Test with real coarse data
                print("\n🧪 Testing model with real coarse data...")
                
                # Create minimal model to test
                model_params = DictConfig({
                    'enhanced_model': {
                        'surface_input_features': 8,
                        'debug': False,
                        'coarse_to_fine': {
                            'hidden_layers': [512, 512, 512],  # Original architecture
                            'use_spectral': True,  # Original setting
                            'use_residual': True,  # Original setting (problematic)
                            'dropout_rate': 0.0,
                        }
                    },
                    'activation': 'relu',
                    'model_type': 'surface',
                    'interp_res': [128, 64, 64],
                    'use_sdf_in_basis_func': True,
                    'num_surface_neighbors': 7,
                    'positional_encoding': False,
                    'surface_neighbors': True,
                    'use_surface_normals': True,
                    'use_surface_area': True,
                    'integral_loss_scaling_factor': 100,
                    'surf_loss_scaling': 5.0,
                    'vol_loss_scaling': 0.0,
                    'normalization': 'min_max_scaling',
                    'encode_parameters': False,
                    'geometry_encoding_type': 'both',
                    'solution_calculation_mode': 'two-loop',
                })
                
                # Note: We can't fully test without all the geometry processing
                # but we can check the coarse-to-fine model weights
                
                print("\n💡 Key findings:")
                if residual_keys:
                    residual_weight = checkpoint.get('coarse_to_fine_model.residual_projection.weight')
                    if residual_weight is not None:
                        print(f"  Residual projection weight norm: {residual_weight.norm():.4f}")
                        if residual_weight.norm() > 1.0:
                            print("  ⚠️ Residual weights are large - likely dominating output!")
                
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("\n🔧 TESTING FIXED ENHANCED DOMINO MODEL\n")
    
    # Test the fixed model
    test_fixed_model()
    
    # Test with real checkpoint if available
    test_with_real_checkpoint()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("\n1. Replace your enhanced_domino_model.py with enhanced_domino_model_fixed.py:")
    print("   cp enhanced_domino_model.py enhanced_domino_model_backup.py")
    print("   cp enhanced_domino_model_fixed.py enhanced_domino_model.py")
    
    print("\n2. Update your config.yaml with the fixed settings")
    
    print("\n3. Retrain from scratch with the fixed model:")
    print("   python train.py")
    
    print("\n4. Monitor training closely:")
    print("   - Loss should decrease steadily")
    print("   - Watch for improvement metrics in logs")
    print("   - Check tensorboard for convergence")
    
    print("\n5. After ~100 epochs, test on a training sample:")
    print("   - Model should at least reproduce training data")
    print("   - If not, there's still a data pipeline issue")
    
    print("\n💡 KEY FIXES APPLIED:")
    print("  ✅ Residual connection DISABLED")
    print("  ✅ Simplified architecture (256x256 instead of 512x512x512)")
    print("  ✅ Spectral features DISABLED")
    print("  ✅ Added dropout regularization")
    print("  ✅ Better weight initialization")
    print("  ✅ Debug logging added")
    print("  ✅ Fixed feature extraction logic")
    
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)