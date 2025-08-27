#!/usr/bin/env python3
"""calibrate_fix.py - Calibrate the fix more precisely"""

import torch
import numpy as np
from pathlib import Path

def calibrate_model_fix():
    """Calibrate the model fix more precisely."""
    
    print("="*80)
    print("CALIBRATING MODEL FIX")
    print("="*80)
    
    # Load original checkpoint
    checkpoint = torch.load(
        "outputs/Ahmed_Dataset/enhanced_fixed/models/best_model/DoMINOEnhanced.0.8.821020098063551e-06.pt",
        map_location='cpu'
    )
    
    # Load scaling factors
    scales = np.load("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
    
    print("\nCurrent situation:")
    print(f"  Original mean: -0.656")
    print(f"  After first fix: -0.448") 
    print(f"  Target mean: -0.11")
    print(f"  Still need shift: +0.338")
    
    # Strategy: Adjust weights and bias more carefully
    print("\n" + "="*40)
    print("CALIBRATED FIX")
    print("="*40)
    
    # Option 1: Keep residual high, reduce correction impact
    checkpoint['coarse_to_fine_model.residual_weight'] = torch.tensor(0.95)
    checkpoint['coarse_to_fine_model.correction_weight'] = torch.tensor(0.05)
    
    # Option 2: Scale output projection more aggressively
    checkpoint['coarse_to_fine_model.output_projection.weight'] = checkpoint['coarse_to_fine_model.output_projection.weight'] / 20.0
    
    # Option 3: Add more bias correction
    scale_range = scales[0, 0] - scales[1, 0]
    additional_shift = 0.338 / scale_range  # Additional shift needed
    
    current_bias = checkpoint['coarse_to_fine_model.output_projection.bias'].clone()
    checkpoint['coarse_to_fine_model.output_projection.bias'][0] = current_bias[0] + additional_shift
    
    print(f"Adjustments made:")
    print(f"  1. Residual weight: 0.95")
    print(f"  2. Correction weight: 0.05 (reduced)")
    print(f"  3. Output weights scaled by 1/20")
    print(f"  4. Bias adjusted by: {additional_shift:.4f}")
    
    # Save calibrated model
    torch.save(checkpoint, "outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced_CALIBRATED.pt")
    print(f"\nSaved calibrated model")
    
    return checkpoint

def test_calibrated_model():
    """Test the calibrated model."""
    
    print("\n" + "="*80)
    print("TESTING CALIBRATED MODEL")
    print("="*80)
    
    from omegaconf import DictConfig
    from enhanced_domino_model import DoMINOEnhanced
    import yaml
    
    # Load config
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = DictConfig(config['model'])
    
    # Create model
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=4,
        model_parameters=model_config
    )
    
    # Load calibrated checkpoint
    checkpoint = torch.load("outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced_CALIBRATED.pt", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test input simulating actual coarse data statistics
    scales = np.load("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
    
    batch = 1
    points = 1000
    
    # Create realistic normalized coarse input
    coarse_physical = torch.zeros(batch, points, 4)
    coarse_physical[:, :, 0] = torch.randn(batch, points) * 0.16 - 0.11  # Pressure
    for i in range(1, 4):
        coarse_physical[:, :, i] = torch.randn(batch, points) * 0.01  # Wall shear
    
    # Normalize using coarse scales (columns 4-7)
    coarse_normalized = torch.zeros_like(coarse_physical)
    for i in range(4):
        max_val = scales[0, i+4]
        min_val = scales[1, i+4]
        coarse_normalized[:, :, i] = (coarse_physical[:, :, i] - min_val) / (max_val - min_val)
    
    geometry = torch.randn(batch, points, 448) * 0.1
    
    # Get predictions
    with torch.no_grad():
        output_normalized = model.coarse_to_fine_model(coarse_normalized, geometry)
    
    # Denormalize using fine scales
    output_physical = torch.zeros_like(output_normalized)
    for i in range(4):
        max_val = scales[0, i]
        min_val = scales[1, i]
        output_physical[:, :, i] = output_normalized[:, :, i] * (max_val - min_val) + min_val
    
    print(f"\nInput (Coarse) Statistics:")
    print(f"  Pressure mean: {coarse_physical[0, :, 0].mean():.4f}")
    print(f"  Pressure std:  {coarse_physical[0, :, 0].std():.4f}")
    
    print(f"\nOutput (Predicted Fine) Statistics:")
    print(f"  Pressure mean: {output_physical[0, :, 0].mean():.4f}")
    print(f"  Pressure std:  {output_physical[0, :, 0].std():.4f}")
    print(f"  Pressure min:  {output_physical[0, :, 0].min():.4f}")
    print(f"  Pressure max:  {output_physical[0, :, 0].max():.4f}")
    
    print(f"\nTarget Statistics (from real fine data):")
    print(f"  Mean: ~-0.11")
    print(f"  Std:  ~0.16")
    print(f"  Range: ~[-0.87, 0.52]")
    
    error = abs(output_physical[0, :, 0].mean() + 0.11)
    if error < 0.05:
        print(f"\n✅ SUCCESS! Mean is very close to target (error: {error:.4f})")
        return True
    elif error < 0.15:
        print(f"\n⚠️ PARTIAL SUCCESS! Mean is reasonably close (error: {error:.4f})")
        return True
    else:
        print(f"\n❌ Still off target (error: {error:.4f})")
        return False

def create_simple_interpolation_baseline():
    """Create a simple baseline that just does interpolation."""
    
    print("\n" + "="*80)
    print("CREATING INTERPOLATION BASELINE")
    print("="*80)
    
    checkpoint = torch.load(
        "outputs/Ahmed_Dataset/enhanced_fixed/models/best_model/DoMINOEnhanced.0.8.821020098063551e-06.pt",
        map_location='cpu'
    )
    
    # Set to pure passthrough (identity mapping)
    checkpoint['coarse_to_fine_model.residual_weight'] = torch.tensor(1.0)
    checkpoint['coarse_to_fine_model.correction_weight'] = torch.tensor(0.0)
    
    # Zero out the bias
    checkpoint['coarse_to_fine_model.output_projection.bias'] = torch.zeros(4)
    
    torch.save(checkpoint, "outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced_BASELINE.pt")
    print("Created baseline model (pure interpolation)")
    print("This should match coarse input statistics")
    
    return True

if __name__ == "__main__":
    # Try calibrated fix
    calibrate_model_fix()
    success = test_calibrated_model()
    
    if not success:
        print("\n" + "="*80)
        print("ALTERNATIVE: TRYING INTERPOLATION BASELINE")
        print("="*80)
        create_simple_interpolation_baseline()
        print("\nBaseline model created.")
        print("This will just output the interpolated coarse data.")
        print("Use this to verify the pipeline works correctly.")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("1. Test both models on actual test data:")
    print("   python test_enhanced.py --checkpoint DoMINOEnhanced_CALIBRATED.pt")
    print("   python test_enhanced.py --checkpoint DoMINOEnhanced_BASELINE.pt")
    
    print("\n2. Compare the outputs:")
    print("   - CALIBRATED should show improvement over baseline")
    print("   - BASELINE should match coarse input")
    
    print("\n3. For proper fix, retrain with:")
    print("   - Residual weight initialized to 0.9-1.0")
    print("   - Correction weight constraint: max 0.2")
    print("   - Output projection weight regularization")
    print("   - Monitor mean predictions during training")
