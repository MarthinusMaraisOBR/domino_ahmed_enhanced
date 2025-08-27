#!/usr/bin/env python3
"""analyze_and_fix_weights.py - Analyze and potentially fix the weight issue"""

import torch
import numpy as np
from pathlib import Path

def analyze_weight_problem():
    """Deep analysis of the weight problem."""
    
    print("="*80)
    print("WEIGHT PROBLEM ANALYSIS")
    print("="*80)
    
    # Load checkpoint
    checkpoint_path = Path("outputs/Ahmed_Dataset/enhanced_fixed/models/best_model/DoMINOEnhanced.0.8.821020098063551e-06.pt")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Find all relevant weights
    print("\n1. Residual Connection Weights:")
    print("-"*40)
    for key in checkpoint.keys():
        if 'residual_weight' in key:
            print(f"  {key}: {checkpoint[key].item():.6f}")
        if 'correction_weight' in key:
            print(f"  {key}: {checkpoint[key].item():.6f}")
    
    # Check output projection
    print("\n2. Output Projection Analysis:")
    print("-"*40)
    for key in checkpoint.keys():
        if 'output_projection' in key:
            tensor = checkpoint[key]
            if 'weight' in key:
                print(f"  {key} shape: {tensor.shape}")
                print(f"    Mean weight: {tensor.mean():.6f}")
                print(f"    Std weight: {tensor.std():.6f}")
                print(f"    Max weight: {tensor.max():.6f}")
                print(f"    Min weight: {tensor.min():.6f}")
            if 'bias' in key:
                print(f"  {key}: {tensor.numpy()}")
    
    # Analyze main network weights
    print("\n3. Main Network Final Layer:")
    print("-"*40)
    for key in checkpoint.keys():
        if 'main_network' in key and '4' in key:  # Last layer
            tensor = checkpoint[key]
            if tensor.dim() >= 2:
                print(f"  {key} shape: {tensor.shape}")
                print(f"    Mean: {tensor.mean():.6f}")
                print(f"    Std: {tensor.std():.6f}")
    
    return checkpoint

def create_fixed_model():
    """Create a fixed version of the model."""
    
    print("\n" + "="*80)
    print("CREATING FIXED MODEL")
    print("="*80)
    
    checkpoint = torch.load(
        "outputs/Ahmed_Dataset/enhanced_fixed/models/best_model/DoMINOEnhanced.0.8.821020098063551e-06.pt",
        map_location='cpu'
    )
    
    # Fix 1: Adjust residual weights to more reasonable values
    for key in checkpoint.keys():
        if 'residual_weight' in key:
            old_val = checkpoint[key].item()
            # Increase to give more weight to input
            checkpoint[key] = torch.tensor(0.95)
            print(f"Fixed residual_weight: {old_val:.4f} → 0.95")
        
        if 'correction_weight' in key:
            old_val = checkpoint[key].item()
            # Increase to actually use the correction
            checkpoint[key] = torch.tensor(0.20)
            print(f"Fixed correction_weight: {old_val:.4f} → 0.20")
    
    # Fix 2: Scale down the output projection weights
    for key in checkpoint.keys():
        if 'output_projection.weight' in key:
            # Scale down by factor of 10 since corrections are too large
            checkpoint[key] = checkpoint[key] / 10.0
            print(f"Scaled down output_projection weights by 10x")
    
    # Fix 3: Adjust bias to correct the mean
    for key in checkpoint.keys():
        if 'output_projection.bias' in key:
            # Current mean is -0.656, want -0.11
            # Shift needed: +0.546 in physical space
            scales = np.load("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
            scale_range = scales[0, 0] - scales[1, 0]  # Fine pressure scale
            shift_normalized = 0.546 / scale_range
            
            checkpoint[key][0] += shift_normalized
            print(f"Added bias correction: {shift_normalized:.4f}")
    
    # Save fixed model
    fixed_path = Path("outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced_FIXED.pt")
    torch.save(checkpoint, fixed_path)
    print(f"\nSaved fixed model to: {fixed_path}")
    
    return fixed_path

def test_fixed_model():
    """Quick test of the fixed model."""
    
    print("\n" + "="*80)
    print("TESTING FIXED MODEL")
    print("="*80)
    
    from omegaconf import DictConfig
    from enhanced_domino_model import DoMINOEnhanced
    import yaml
    
    # Load config
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_config = DictConfig(config['model'])
    
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=4,
        model_parameters=model_config
    )
    
    # Load fixed checkpoint
    checkpoint = torch.load("outputs/Ahmed_Dataset/enhanced_fixed/models/DoMINOEnhanced_FIXED.pt", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Test with normalized input
    test_input = torch.rand(1, 100, 4) * 0.8 + 0.1  # Normalized range
    geometry = torch.randn(1, 100, 448) * 0.1
    
    with torch.no_grad():
        output = model.coarse_to_fine_model(test_input, geometry)
    
    # Denormalize
    scales = np.load("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
    output_physical = output.clone()
    for i in range(4):
        max_val = scales[0, i]
        min_val = scales[1, i]
        output_physical[:, :, i] = output[:, :, i] * (max_val - min_val) + min_val
    
    print(f"\nFixed Model Output (Pressure):")
    print(f"  Normalized mean: {output[0, :, 0].mean():.4f}")
    print(f"  Physical mean: {output_physical[0, :, 0].mean():.4f}")
    print(f"  Physical std: {output_physical[0, :, 0].std():.4f}")
    print(f"\nTarget: mean ~-0.11, std ~0.16")
    
    if abs(output_physical[0, :, 0].mean() + 0.11) < 0.2:
        print("\n✅ Fixed model produces reasonable mean!")
    else:
        print("\n⚠️ Still needs adjustment")

if __name__ == "__main__":
    # Analyze the problem
    checkpoint = analyze_weight_problem()
    
    # Create fixed model
    fixed_path = create_fixed_model()
    
    # Test it
    test_fixed_model()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Test the fixed model on actual test data:")
    print("   python test_enhanced.py --checkpoint DoMINOEnhanced_FIXED.pt")
    print("\n2. If it works, the issue was learned weights")
    print("   Need to retrain with:")
    print("   - Better weight initialization")
    print("   - Constraints on weight magnitudes")
    print("   - Regularization on correction magnitude")
    print("\n3. If it doesn't work, the training data itself may be corrupted")
