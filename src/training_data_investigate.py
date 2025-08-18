#!/usr/bin/env python3
"""
Step 1: Verify that training data is correctly structured
This script checks if fine/coarse features are in the right order
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def verify_training_data_structure():
    """Check the structure of processed training data."""
    
    print("="*80)
    print("STEP 1: VERIFYING TRAINING DATA STRUCTURE")
    print("="*80)
    
    # Check processed training data
    train_dir = Path("/data/ahmed_data/processed/train/")
    
    if not train_dir.exists():
        print(f"❌ Training data directory not found: {train_dir}")
        return False
    
    # Load a few samples
    npy_files = list(train_dir.glob("*.npy"))[:5]
    
    if not npy_files:
        print("❌ No processed NPY files found")
        return False
    
    print(f"\n📊 Checking {len(npy_files)} training samples...")
    
    all_fine_ranges = []
    all_coarse_ranges = []
    
    for i, npy_file in enumerate(npy_files):
        print(f"\n Sample {i+1}: {npy_file.name}")
        data = np.load(npy_file, allow_pickle=True).item()
        
        if 'surface_fields' in data:
            surface_fields = data['surface_fields']
            print(f"  Surface fields shape: {surface_fields.shape}")
            
            if surface_fields.shape[1] == 8:
                # Expected: [fine_pressure, fine_shear_x, fine_shear_y, fine_shear_z,
                #            coarse_pressure, coarse_shear_x, coarse_shear_y, coarse_shear_z]
                fine_features = surface_fields[:, :4]
                coarse_features = surface_fields[:, 4:8]
                
                print(f"  Fine features (0:4):")
                print(f"    Pressure range: [{fine_features[:, 0].min():.6f}, {fine_features[:, 0].max():.6f}]")
                print(f"    Shear X range:  [{fine_features[:, 1].min():.6f}, {fine_features[:, 1].max():.6f}]")
                
                print(f"  Coarse features (4:8):")
                print(f"    Pressure range: [{coarse_features[:, 0].min():.6f}, {coarse_features[:, 0].max():.6f}]")
                print(f"    Shear X range:  [{coarse_features[:, 1].min():.6f}, {coarse_features[:, 1].max():.6f}]")
                
                # Check if coarse is actually different from fine
                pressure_diff = np.mean(np.abs(fine_features[:, 0] - coarse_features[:, 0]))
                print(f"  Mean pressure difference (fine-coarse): {pressure_diff:.6f}")
                
                # Collect ranges for analysis
                all_fine_ranges.append([fine_features.min(), fine_features.max()])
                all_coarse_ranges.append([coarse_features.min(), coarse_features.max()])
                
                # Check for obvious issues
                if np.allclose(fine_features, coarse_features):
                    print("  ⚠️ WARNING: Fine and coarse features are identical!")
                
                if np.abs(fine_features).max() > 10:
                    print("  ⚠️ WARNING: Fine features seem unnormalized (>10)")
                
                if np.abs(coarse_features).max() > 10:
                    print("  ⚠️ WARNING: Coarse features seem unnormalized (>10)")
                    
            else:
                print(f"  ❌ Wrong number of features: {surface_fields.shape[1]} (expected 8)")
                return False
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    fine_ranges = np.array(all_fine_ranges)
    coarse_ranges = np.array(all_coarse_ranges)
    
    print(f"\nFine features across all samples:")
    print(f"  Global min: {fine_ranges[:, 0].min():.6f}")
    print(f"  Global max: {fine_ranges[:, 1].max():.6f}")
    
    print(f"\nCoarse features across all samples:")
    print(f"  Global min: {coarse_ranges[:, 0].min():.6f}")
    print(f"  Global max: {coarse_ranges[:, 1].max():.6f}")
    
    # Check scaling factors
    print("\n" + "="*60)
    print("CHECKING SCALING FACTORS")
    print("="*60)
    
    scaling_paths = [
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/surface_scaling_factors.npy"
    ]
    
    for path in scaling_paths:
        if Path(path).exists():
            factors = np.load(path)
            print(f"\n{path}:")
            print(f"  Shape: {factors.shape}")
            if factors.shape[1] == 8:
                print(f"  Fine max:   {factors[0, :4]}")
                print(f"  Fine min:   {factors[1, :4]}")
                print(f"  Coarse max: {factors[0, 4:]}")
                print(f"  Coarse min: {factors[1, 4:]}")
            else:
                print(f"  Max: {factors[0]}")
                print(f"  Min: {factors[1]}")
    
    return True

def check_enhanced_model_forward():
    """Test the enhanced model's forward pass logic."""
    
    print("\n" + "="*80)
    print("CHECKING ENHANCED MODEL FORWARD PASS")
    print("="*80)
    
    try:
        from enhanced_domino_model import DoMINOEnhanced
        
        # Create a minimal test
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters={
                'enhanced_model': {'surface_input_features': 8},
                'activation': 'relu',
                'model_type': 'surface'
            }
        )
        
        # Create dummy data
        batch_size = 2
        num_points = 100
        
        # Training mode: 8 features
        train_data = {
            'surface_fields': torch.randn(batch_size, num_points, 8),
            'geometry_coordinates': torch.randn(batch_size, num_points, 3),
            'surface_mesh_centers': torch.randn(batch_size, num_points, 3),
        }
        
        print("\nTesting training mode (8 features)...")
        with torch.no_grad():
            _, train_output = model(train_data)
        
        if train_output is not None:
            print(f"  ✅ Training output shape: {train_output.shape}")
            print(f"  Expected: ({batch_size}, {num_points}, 4)")
        
        # Inference mode: 4 features
        test_data = {
            'surface_fields': torch.randn(batch_size, num_points, 4),
            'geometry_coordinates': torch.randn(batch_size, num_points, 3),
            'surface_mesh_centers': torch.randn(batch_size, num_points, 3),
        }
        
        print("\nTesting inference mode (4 features)...")
        with torch.no_grad():
            _, test_output = model(test_data)
        
        if test_output is not None:
            print(f"  ✅ Inference output shape: {test_output.shape}")
            print(f"  Expected: ({batch_size}, {num_points}, 4)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def analyze_loss_calculation():
    """Analyze if the loss is calculated correctly."""
    
    print("\n" + "="*80)
    print("ANALYZING LOSS CALCULATION")
    print("="*80)
    
    print("\nExpected behavior during training:")
    print("1. Model receives 8 features: [fine(0:4), coarse(4:8)]")
    print("2. Model uses coarse(4:8) as input")
    print("3. Model predicts fine features")
    print("4. Loss compares predictions against fine(0:4)")
    
    print("\nChecking train.py loss implementation...")
    
    # Check if enhanced loss function exists
    train_path = Path("train.py")
    if train_path.exists():
        with open(train_path, 'r') as f:
            content = f.read()
            
        if 'compute_loss_dict_enhanced' in content:
            print("  ✅ Enhanced loss function found")
            
            # Check key parts
            if 'target_surf = surface_fields_all[..., :4]' in content:
                print("  ✅ Loss targets fine features (0:4)")
            else:
                print("  ⚠️ Cannot verify if loss targets correct features")
                
            if 'coarse_features = surface_fields_all[..., 4:8]' in content:
                print("  ✅ Coarse features extracted (4:8)")
            else:
                print("  ⚠️ Cannot verify coarse feature extraction")
                
        else:
            print("  ❌ Enhanced loss function not found!")
            print("  This is critical - model may be training incorrectly")
    
    return True

def visualize_feature_distributions():
    """Create visualization of feature distributions."""
    
    print("\n" + "="*80)
    print("VISUALIZING FEATURE DISTRIBUTIONS")
    print("="*80)
    
    train_dir = Path("/data/ahmed_data/processed/train/")
    sample = list(train_dir.glob("*.npy"))[0]
    data = np.load(sample, allow_pickle=True).item()
    
    if 'surface_fields' not in data:
        print("❌ No surface fields found")
        return
    
    surface_fields = data['surface_fields']
    
    if surface_fields.shape[1] != 8:
        print(f"❌ Wrong shape: {surface_fields.shape}")
        return
    
    fine_pressure = surface_fields[:, 0]
    coarse_pressure = surface_fields[:, 4]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram comparison
    axes[0].hist(fine_pressure, bins=50, alpha=0.5, label='Fine', color='blue')
    axes[0].hist(coarse_pressure, bins=50, alpha=0.5, label='Coarse', color='red')
    axes[0].set_xlabel('Pressure Coefficient')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Pressure Distribution')
    axes[0].legend()
    
    # Scatter plot
    axes[1].scatter(coarse_pressure[:1000], fine_pressure[:1000], alpha=0.5, s=1)
    axes[1].plot([coarse_pressure.min(), coarse_pressure.max()], 
                 [coarse_pressure.min(), coarse_pressure.max()], 
                 'r--', label='y=x')
    axes[1].set_xlabel('Coarse Pressure')
    axes[1].set_ylabel('Fine Pressure')
    axes[1].set_title('Coarse vs Fine Correlation')
    axes[1].legend()
    
    # Difference histogram
    diff = fine_pressure - coarse_pressure
    axes[2].hist(diff, bins=50, color='green', alpha=0.7)
    axes[2].set_xlabel('Fine - Coarse')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Pressure Difference (mean={diff.mean():.4f})')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150)
    print(f"  ✅ Saved visualization to feature_distributions.png")
    
    # Print statistics
    print(f"\nPressure Statistics:")
    print(f"  Fine:   mean={fine_pressure.mean():.4f}, std={fine_pressure.std():.4f}")
    print(f"  Coarse: mean={coarse_pressure.mean():.4f}, std={coarse_pressure.std():.4f}")
    print(f"  Correlation: {np.corrcoef(fine_pressure, coarse_pressure)[0,1]:.4f}")
    
    return True

def main():
    """Run all verification steps."""
    
    print("\n🔍 SYSTEMATIC INVESTIGATION OF ENHANCED DOMINO TRAINING\n")
    
    # Step 1: Verify data structure
    data_ok = verify_training_data_structure()
    
    # Step 2: Check model forward pass
    model_ok = check_enhanced_model_forward()
    
    # Step 3: Analyze loss calculation
    loss_ok = analyze_loss_calculation()
    
    # Step 4: Visualize distributions
    viz_ok = visualize_feature_distributions()
    
    # Summary
    print("\n" + "="*80)
    print("INVESTIGATION SUMMARY")
    print("="*80)
    
    issues = []
    
    if not data_ok:
        issues.append("Training data structure problems detected")
    
    if not model_ok:
        issues.append("Model forward pass issues")
        
    if not loss_ok:
        issues.append("Loss calculation may be incorrect")
    
    if issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Basic structure appears correct")
        
    print("\n🎯 NEXT STEPS:")
    print("1. Check training convergence - does loss actually decrease?")
    print("2. Test if model can overfit to a single sample")
    print("3. Verify the CoarseToFineModel architecture")
    print("4. Check if residual connections are causing issues")
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
