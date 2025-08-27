#!/usr/bin/env python3
"""inspect_raw_outputs_complete.py - Complete config version"""

import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from enhanced_domino_model import DoMINOEnhanced

def inspect_raw_outputs():
    """Inspect raw model outputs with complete configuration."""
    
    print("="*80)
    print("RAW MODEL OUTPUT INSPECTION - COMPLETE VERSION")
    print("="*80)
    
    # Load the actual config used during training
    import yaml
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create complete model config
    model_config = DictConfig({
        'enhanced_model': config['model']['enhanced_model'],
        'interp_res': config['model']['interp_res'],
        'activation': config['model']['activation'],
        'model_type': config['model']['model_type'],
        'use_sdf_in_basis_func': config['model']['use_sdf_in_basis_func'],
        'positional_encoding': config['model']['positional_encoding'],
        'surface_neighbors': config['model']['surface_neighbors'],
        'num_surface_neighbors': config['model']['num_surface_neighbors'],
        'use_surface_normals': config['model']['use_surface_normals'],
        'use_surface_area': config['model']['use_surface_area'],
        'integral_loss_scaling_factor': config['model']['integral_loss_scaling_factor'],
        'normalization': config['model']['normalization'],
        'encode_parameters': config['model']['encode_parameters'],
        'loss_function': config['model']['loss_function'],
        'surf_loss_scaling': config['model']['surf_loss_scaling'],
        'vol_loss_scaling': config['model']['vol_loss_scaling'],
        'volume_points_sample': config['model']['volume_points_sample'],
        'surface_points_sample': config['model']['surface_points_sample'],
        'geom_points_sample': config['model']['geom_points_sample'],
        'surface_sampling_algorithm': config['model']['surface_sampling_algorithm'],
        'geometry_encoding_type': config['model']['geometry_encoding_type'],
        'solution_calculation_mode': config['model']['solution_calculation_mode'],
        'resampling_surface_mesh': config['model']['resampling_surface_mesh'],
        'geometry_rep': config['model']['geometry_rep'],
        'nn_basis_functions': config['model']['nn_basis_functions'],
        'local_point_conv': config['model']['local_point_conv'],
        'aggregation_model': config['model']['aggregation_model'],
        'position_encoder': config['model']['position_encoder'],
        'geometry_local': config['model']['geometry_local'],
        'parameter_model': config['model']['parameter_model']  # This was missing!
    })
    
    # Load model
    checkpoint_path = Path("outputs/Ahmed_Dataset/enhanced_fixed/models/best_model/DoMINOEnhanced.0.8.821020098063551e-06.pt")
    
    print("Creating model...")
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=4,
        model_parameters=model_config
    )
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create realistic test input
    batch_size = 1
    n_points = 1000
    
    # Load scaling factors to create properly scaled input
    scales = np.load("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
    
    # Create normalized input (as the model expects)
    coarse_features = torch.zeros(batch_size, n_points, 4)
    
    # Generate realistic normalized values
    for i in range(4):
        # Use coarse scaling (columns 4-7)
        max_val = scales[0, i+4]
        min_val = scales[1, i+4]
        
        # Generate normalized values between 0 and 1
        normalized = torch.rand(batch_size, n_points)
        
        # Add some structure (not just random)
        if i == 0:  # Pressure
            # Add stagnation point (high pressure)
            normalized[:, :100] = 0.9 + torch.rand(batch_size, 100) * 0.1
            # Add wake (low pressure) 
            normalized[:, 900:] = torch.rand(batch_size, 100) * 0.2
        
        coarse_features[:, :, i] = normalized
    
    geometry_encoding = torch.randn(batch_size, n_points, 448) * 0.1
    
    # Get the coarse-to-fine model
    c2f = model.coarse_to_fine_model
    
    print("\nModel Configuration:")
    print(f"  Use residual: {c2f.use_residual}")
    if hasattr(c2f, 'residual_weight'):
        print(f"  Residual weight: {c2f.residual_weight.item():.6f}")
        print(f"  Correction weight: {c2f.correction_weight.item():.6f}")
    
    # Forward pass through coarse-to-fine model
    with torch.no_grad():
        # Get intermediate outputs
        coarse_processed = c2f.coarse_feature_extractor(coarse_features)
        combined = torch.cat([coarse_processed, geometry_encoding], dim=-1)
        processed = c2f.main_network(combined)
        correction = c2f.output_projection(processed)
        
        # Final output
        output = c2f(coarse_features, geometry_encoding)
    
    print("\n" + "="*60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("="*60)
    
    print("\n1. Input (Normalized Coarse):")
    print(f"   Mean: {coarse_features[0, :, 0].mean():.6f}")
    print(f"   Std:  {coarse_features[0, :, 0].std():.6f}")
    print(f"   Min:  {coarse_features[0, :, 0].min():.6f}")
    print(f"   Max:  {coarse_features[0, :, 0].max():.6f}")
    
    print("\n2. After Feature Extraction:")
    print(f"   Mean: {coarse_processed.mean():.6f}")
    print(f"   Std:  {coarse_processed.std():.6f}")
    
    print("\n3. After Main Network:")
    print(f"   Mean: {processed.mean():.6f}")
    print(f"   Std:  {processed.std():.6f}")
    
    print("\n4. Correction (from output_projection):")
    print(f"   Mean: {correction[0, :, 0].mean():.6f}")
    print(f"   Std:  {correction[0, :, 0].std():.6f}")
    print(f"   Min:  {correction[0, :, 0].min():.6f}")
    print(f"   Max:  {correction[0, :, 0].max():.6f}")
    
    print("\n5. Final Output (after residual):")
    print(f"   Mean: {output[0, :, 0].mean():.6f}")
    print(f"   Std:  {output[0, :, 0].std():.6f}")
    print(f"   Min:  {output[0, :, 0].min():.6f}")
    print(f"   Max:  {output[0, :, 0].max():.6f}")
    
    # Analyze the transformation
    print("\n" + "="*60)
    print("TRANSFORMATION ANALYSIS")
    print("="*60)
    
    if c2f.use_residual:
        # What should the output be?
        expected = c2f.residual_weight * coarse_features + c2f.correction_weight * correction
        
        print(f"\nExpected output based on weights:")
        print(f"  {c2f.residual_weight.item():.4f} * input + {c2f.correction_weight.item():.4f} * correction")
        
        print(f"\nExpected vs Actual (pressure channel):")
        print(f"  Expected mean: {expected[0, :, 0].mean():.6f}")
        print(f"  Actual mean:   {output[0, :, 0].mean():.6f}")
        print(f"  Difference:    {(output[0, :, 0].mean() - expected[0, :, 0].mean()):.6f}")
    
    # Now denormalize to see what physical values we get
    print("\n" + "="*60)
    print("DENORMALIZED OUTPUT (Physical Values)")
    print("="*60)
    
    # Use FINE scaling for denormalization (first 4 columns)
    output_denorm = output.clone()
    for i in range(4):
        max_val = scales[0, i]  # Fine max
        min_val = scales[1, i]  # Fine min
        output_denorm[:, :, i] = output[:, :, i] * (max_val - min_val) + min_val
    
    print(f"\nDenormalized Pressure:")
    print(f"  Mean: {output_denorm[0, :, 0].mean():.6f}")
    print(f"  Std:  {output_denorm[0, :, 0].std():.6f}")
    print(f"  Min:  {output_denorm[0, :, 0].min():.6f}")
    print(f"  Max:  {output_denorm[0, :, 0].max():.6f}")
    
    print(f"\nExpected physical range (from data):")
    print(f"  Mean: ~-0.11")
    print(f"  Range: ~[-0.87, 0.52]")
    
    # Check if the issue is in normalization
    print("\n" + "="*60)
    print("NORMALIZATION DIAGNOSIS")
    print("="*60)
    
    # What happens if model outputs are in wrong range?
    if abs(output_denorm[0, :, 0].mean() + 0.79) < 0.1:
        print("🔴 CRITICAL: Model outputs match the problematic -0.79 mean!")
        print("   The model learned to predict in normalized space incorrectly!")
        
        # What normalized value gives -0.79 in physical space?
        problematic_physical = -0.79
        normalized_problematic = (problematic_physical - scales[1, 0]) / (scales[0, 0] - scales[1, 0])
        print(f"\n   Physical -0.79 corresponds to normalized: {normalized_problematic:.4f}")
        print(f"   Model is outputting around: {output[0, :, 0].mean():.4f}")
        
        if abs(output[0, :, 0].mean() - normalized_problematic) < 0.1:
            print("\n   🎯 CONFIRMED: Model learned wrong normalized target!")
            print("      It's predicting constant normalized value that denormalizes to -0.79")
    
    return output, output_denorm

if __name__ == "__main__":
    output_norm, output_denorm = inspect_raw_outputs()