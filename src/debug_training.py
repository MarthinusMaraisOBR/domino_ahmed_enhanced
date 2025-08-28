#!/usr/bin/env python3
"""
debug_training.py - Debug why training shows negative improvements
This script checks data alignment, model behavior, and loss calculations
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def debug_data_alignment():
    """Check if coarse and fine data are properly aligned."""
    
    print("="*80)
    print("DATA ALIGNMENT CHECK")
    print("="*80)
    
    # Load a sample training file
    train_file = Path("/data/ahmed_data/processed/train/run_1.npy")
    
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return False
    
    data = np.load(train_file, allow_pickle=True).item()
    
    # Check surface fields
    if 'surface_fields' not in data:
        print("❌ No surface fields in data")
        return False
    
    surface_fields = data['surface_fields']
    print(f"Surface fields shape: {surface_fields.shape}")
    
    if surface_fields.shape[1] != 8:
        print(f"❌ Expected 8 features, got {surface_fields.shape[1]}")
        return False
    
    # Extract fine and coarse
    n_points = min(1000, surface_fields.shape[0])
    fine_pressure = surface_fields[:n_points, 0]
    coarse_pressure = surface_fields[:n_points, 4]
    
    fine_shear_x = surface_fields[:n_points, 1]
    coarse_shear_x = surface_fields[:n_points, 5]
    
    print("\n📊 Field Statistics:")
    print(f"Fine pressure:   mean={fine_pressure.mean():.6f}, std={fine_pressure.std():.6f}")
    print(f"Coarse pressure: mean={coarse_pressure.mean():.6f}, std={coarse_pressure.std():.6f}")
    print(f"Difference:      mean={np.mean(fine_pressure - coarse_pressure):.6f}")
    
    # Check correlation
    corr_p, _ = pearsonr(fine_pressure, coarse_pressure)
    corr_s, _ = pearsonr(fine_shear_x, coarse_shear_x)
    
    print(f"\n📈 Correlations:")
    print(f"Pressure correlation: {corr_p:.4f}")
    print(f"Shear X correlation:  {corr_s:.4f}")
    
    if corr_p < 0.5:
        print("⚠️ WARNING: Low pressure correlation - possible misalignment!")
    
    # Check if coarse is actually worse than fine
    # In a properly interpolated dataset, coarse should have less detail
    fine_range = fine_pressure.max() - fine_pressure.min()
    coarse_range = coarse_pressure.max() - coarse_pressure.min()
    
    print(f"\n📏 Dynamic Range:")
    print(f"Fine range:   {fine_range:.6f}")
    print(f"Coarse range: {coarse_range:.6f}")
    print(f"Range ratio:  {coarse_range/fine_range:.2f}")
    
    if coarse_range > fine_range:
        print("⚠️ WARNING: Coarse has larger range than fine - unexpected!")
    
    # Visualize alignment
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pressure scatter
    axes[0].scatter(coarse_pressure, fine_pressure, alpha=0.5, s=1)
    axes[0].plot([coarse_pressure.min(), coarse_pressure.max()],
                [coarse_pressure.min(), coarse_pressure.max()], 'r--', label='y=x')
    axes[0].set_xlabel('Coarse Pressure')
    axes[0].set_ylabel('Fine Pressure')
    axes[0].set_title(f'Pressure Alignment (corr={corr_p:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Shear scatter
    axes[1].scatter(coarse_shear_x, fine_shear_x, alpha=0.5, s=1)
    axes[1].plot([coarse_shear_x.min(), coarse_shear_x.max()],
                [coarse_shear_x.min(), coarse_shear_x.max()], 'r--', label='y=x')
    axes[1].set_xlabel('Coarse Shear X')
    axes[1].set_ylabel('Fine Shear X')
    axes[1].set_title(f'Shear X Alignment (corr={corr_s:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Coarse-Fine Data Alignment Check')
    plt.tight_layout()
    plt.savefig('data_alignment_check.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved alignment plot to: data_alignment_check.png")
    
    return corr_p > 0.7 and corr_s > 0.5


def debug_model_behavior():
    """Test model behavior with controlled inputs."""
    
    print("\n" + "="*80)
    print("MODEL BEHAVIOR TEST")
    print("="*80)
    
    from enhanced_domino_model import DoMINOEnhanced
    from omegaconf import DictConfig
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
    model.eval()
    
    # Test 1: Check if model can learn identity
    print("\n1. Identity Test (output should equal input):")
    
    batch_size, n_points, n_features = 1, 100, 4
    
    # Create identical input and target
    test_input = torch.randn(batch_size, n_points, n_features) * 0.1
    geometry = torch.randn(batch_size, n_points, 448) * 0.01
    
    with torch.no_grad():
        output = model.coarse_to_fine_model(test_input, geometry)
    
    diff = (output - test_input).abs().mean()
    print(f"   Input mean:  {test_input.mean():.6f}")
    print(f"   Output mean: {output.mean():.6f}")
    print(f"   Difference:  {diff:.6f}")
    
    if diff > 0.5:
        print("   ❌ Model not preserving input - large transformation!")
    
    # Test 2: Check gradient flow
    print("\n2. Gradient Flow Test:")
    
    test_input.requires_grad = True
    output = model.coarse_to_fine_model(test_input, geometry)
    loss = output.mean()
    loss.backward()
    
    if test_input.grad is not None:
        grad_norm = test_input.grad.norm().item()
        print(f"   Input gradient norm: {grad_norm:.6f}")
        if grad_norm < 1e-6:
            print("   ⚠️ Very small gradients - possible vanishing gradient!")
        elif grad_norm > 100:
            print("   ⚠️ Very large gradients - possible exploding gradient!")
        else:
            print("   ✅ Gradient flow looks normal")
    
    # Test 3: Check weight distribution
    print("\n3. Weight Analysis:")
    
    c2f = model.coarse_to_fine_model
    
    # Check output projection weights
    output_weight = c2f.output_projection.weight
    print(f"   Output weight shape: {output_weight.shape}")
    print(f"   Output weight mean:  {output_weight.mean():.6f}")
    print(f"   Output weight std:   {output_weight.std():.6f}")
    
    if output_weight.std() < 1e-4:
        print("   ⚠️ Output weights nearly constant - not learning!")
    
    # Check if residual connection exists and its weight
    if hasattr(c2f, 'residual_weight'):
        print(f"   Residual weight: {c2f.residual_weight.item():.4f}")
        print(f"   Correction weight: {c2f.correction_weight.item():.4f}")
    
    return True


def debug_loss_calculation():
    """Test if loss calculation is correct."""
    
    print("\n" + "="*80)
    print("LOSS CALCULATION TEST")
    print("="*80)
    
    from physics_loss_fixed import PhysicsAwareLossFixed
    
    # Create loss function
    loss_fn = PhysicsAwareLossFixed(force_scale=1e6, debug=True)
    
    # Create dummy data
    batch_size, n_points = 1, 1000
    
    # Scenario 1: Perfect prediction (loss should be ~0)
    print("\n1. Perfect Prediction Test:")
    
    target = torch.randn(batch_size, n_points, 4) * 0.1
    prediction = target.clone()
    coarse = target + torch.randn_like(target) * 0.05  # Slightly worse
    
    areas = torch.ones(batch_size, n_points, 1)
    normals = torch.randn(batch_size, n_points, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    
    # Mock model
    class MockModel:
        pass
    
    model = MockModel()
    
    loss, components = loss_fn(prediction, target, coarse, areas, normals, model)
    
    print(f"   Total loss: {loss:.6f}")
    print(f"   MSE: {components['mse']:.6f}")
    print(f"   Improvement: {components['improvement']*100:.1f}%")
    
    if loss > 0.01:
        print("   ⚠️ Loss too high for perfect prediction!")
    
    # Scenario 2: Worse than baseline (negative improvement)
    print("\n2. Worse Than Baseline Test:")
    
    bad_prediction = target + torch.randn_like(target) * 0.2  # Much worse
    
    loss, components = loss_fn(bad_prediction, target, coarse, areas, normals, model)
    
    print(f"   Total loss: {loss:.6f}")
    print(f"   Improvement: {components['improvement']*100:.1f}%")
    
    if components['improvement'] >= 0:
        print("   ❌ Should show negative improvement!")
    
    # Scenario 3: Check force calculation
    print("\n3. Force Calculation Test:")
    
    # Create data with known forces
    pressure = torch.ones(batch_size, n_points, 1) * 0.1
    shear = torch.zeros(batch_size, n_points, 3)
    test_fields = torch.cat([pressure, shear], dim=-1)
    
    drag, lift = loss_fn._compute_forces(test_fields, areas, normals)
    
    print(f"   Drag force: {drag:.6f}")
    print(f"   Lift force: {lift:.6f}")
    
    # Forces should be non-zero for non-zero pressure
    if abs(drag) < 1e-6 and abs(lift) < 1e-6:
        print("   ⚠️ Forces too small - check calculation!")
    
    return True


def main():
    """Run all debugging checks."""
    
    print("🔍 ENHANCED DOMINO TRAINING DEBUGGER")
    print("="*80)
    print("This will diagnose why your model shows negative improvements\n")
    
    # Run checks
    checks = {
        "Data Alignment": debug_data_alignment,
        "Model Behavior": debug_model_behavior,
        "Loss Calculation": debug_loss_calculation
    }
    
    results = {}
    for name, check_fn in checks.items():
        print(f"\n{'='*40}")
        print(f"Running: {name}")
        print('='*40)
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20} {status}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    
    if not results.get("Data Alignment", False):
        print("\n1. Data Alignment Issues Detected:")
        print("   - Check interpolation in process_data.py")
        print("   - Verify coarse and fine VTP files match")
        print("   - Ensure same mesh ordering")
    
    if not results.get("Model Behavior", False):
        print("\n2. Model Behavior Issues:")
        print("   - Reduce model complexity (use [128, 64] layers)")
        print("   - Check weight initialization")
        print("   - Verify geometry encoding dimension")
    
    if not results.get("Loss Calculation", False):
        print("\n3. Loss Calculation Issues:")
        print("   - Check force calculation units")
        print("   - Verify area and normal vectors")
        print("   - Adjust force_scale parameter")
    
    print("\n💡 MOST LIKELY ISSUE:")
    print("Based on your symptoms (negative improvement, low physics loss),")
    print("the coarse interpolated data might be misaligned with fine data.")
    print("This would cause the model to learn corrections that make things worse.")
    print("\nTry:")
    print("1. Visualize a few training samples in ParaView")
    print("2. Check if coarse and fine pressure patterns match")
    print("3. Verify the interpolation preserves key features")
    
    return all(results.values())


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
