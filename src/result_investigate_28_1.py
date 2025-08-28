# Fixed analysis script
import numpy as np
import pyvista as pv
from pathlib import Path

def analyze_test_results():
    print("="*80)
    print("ANALYZING TEST RESULTS")
    print("="*80)
    
    # Load one of the output VTP files to examine
    test_case = 451
    vtp_path = f"/data/ahmed_data/predictions_v2/boundary_{test_case}_comprehensive_comparison.vtp"
    
    if not Path(vtp_path).exists():
        print(f"❌ Output file not found: {vtp_path}")
        return
    
    # Load the mesh
    mesh = pv.read(vtp_path)
    
    print(f"\nAnalyzing test case {test_case}:")
    print("-"*40)
    
    # Extract fields
    coarse_pressure = np.array(mesh['Coarse_Pressure'])
    fine_pressure = np.array(mesh['Fine_Pressure_GroundTruth_Interpolated'])
    pred_pressure = np.array(mesh['Predicted_Pressure'])
    
    print("\nPressure Statistics:")
    print(f"  Coarse:    mean={coarse_pressure.mean():.6f}, std={coarse_pressure.std():.6f}")
    print(f"             min={coarse_pressure.min():.6f}, max={coarse_pressure.max():.6f}")
    print(f"  Fine:      mean={fine_pressure.mean():.6f}, std={fine_pressure.std():.6f}")
    print(f"             min={fine_pressure.min():.6f}, max={fine_pressure.max():.6f}")
    print(f"  Predicted: mean={pred_pressure.mean():.6f}, std={pred_pressure.std():.6f}")
    print(f"             min={pred_pressure.min():.6f}, max={pred_pressure.max():.6f}")
    
    # Check if predictions are reasonable
    print("\n🔍 Diagnostic Checks:")
    
    # 1. Check if predictions have proper variation
    pred_std_ratio = pred_pressure.std() / fine_pressure.std()
    print(f"  Prediction std / Fine std: {pred_std_ratio:.3f}")
    if pred_std_ratio < 0.1:
        print("    ⚠️ Predictions have very low variation - might be nearly constant!")
    elif pred_std_ratio > 3.0:
        print("    ⚠️ Predictions have excessive variation!")
    else:
        print("    ✅ Prediction variation seems reasonable")
    
    # 2. Check mean shift
    mean_diff_coarse = abs(coarse_pressure.mean() - fine_pressure.mean())
    mean_diff_pred = abs(pred_pressure.mean() - fine_pressure.mean())
    print(f"  Mean difference from fine:")
    print(f"    Coarse: {mean_diff_coarse:.6f}")
    print(f"    Pred:   {mean_diff_pred:.6f}")
    if mean_diff_pred > mean_diff_coarse * 2:
        print("    ⚠️ Predictions have larger mean error than coarse baseline!")
    
    # 3. Check range
    print(f"\n  Range analysis:")
    coarse_range = coarse_pressure.max() - coarse_pressure.min()
    fine_range = fine_pressure.max() - fine_pressure.min()
    pred_range = pred_pressure.max() - pred_pressure.min()
    print(f"    Coarse range: {coarse_range:.6f}")
    print(f"    Fine range:   {fine_range:.6f}")
    print(f"    Pred range:   {pred_range:.6f}")
    
    # 4. Check correlation
    from scipy.stats import pearsonr
    corr_coarse_fine, _ = pearsonr(coarse_pressure.flatten(), fine_pressure.flatten())
    corr_pred_fine, _ = pearsonr(pred_pressure.flatten(), fine_pressure.flatten())
    print(f"\n  Correlation with fine:")
    print(f"    Coarse: {corr_coarse_fine:.3f}")
    print(f"    Pred:   {corr_pred_fine:.3f}")
    if corr_pred_fine < corr_coarse_fine:
        print("    ⚠️ Predictions have worse correlation than baseline!")
    
    # 5. Check model weights from checkpoint
    print("\n" + "="*60)
    print("CHECKING MODEL WEIGHTS")
    print("="*60)
    
    import torch
    checkpoint_path = "outputs/Ahmed_Dataset/enhanced_v2_physics/models/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Look for coarse-to-fine model weights
    for key in checkpoint.keys():
        if 'residual_weight' in key:
            print(f"  {key}: {checkpoint[key].item():.6f}")
        if 'correction_weight' in key:
            print(f"  {key}: {checkpoint[key].item():.6f}")
        if 'output_bias' in key:
            bias = checkpoint[key].numpy()
            print(f"  {key}: {bias}")
    
    # Check output projection weights
    output_weight_key = 'coarse_to_fine_model.output_projection.weight'
    if output_weight_key in checkpoint:
        weights = checkpoint[output_weight_key]
        print(f"\n  Output projection weight stats:")
        print(f"    Shape: {weights.shape}")
        print(f"    Mean: {weights.mean():.6f}")
        print(f"    Std:  {weights.std():.6f}")
        print(f"    Max:  {weights.max():.6f}")  # Fixed the typo here
        print(f"    Min:  {weights.min():.6f}")
    
    # Let's also check what happens if we look at errors spatially
    print("\n" + "="*60)
    print("SPATIAL ERROR ANALYSIS")
    print("="*60)
    
    # Calculate errors
    coarse_error = np.abs(coarse_pressure - fine_pressure)
    pred_error = np.abs(pred_pressure - fine_pressure)
    
    print(f"  Mean Absolute Error:")
    print(f"    Coarse: {coarse_error.mean():.6f}")
    print(f"    Pred:   {pred_error.mean():.6f}")
    
    print(f"\n  Max Absolute Error:")
    print(f"    Coarse: {coarse_error.max():.6f}")
    print(f"    Pred:   {pred_error.max():.6f}")
    
    # Check where the largest errors are
    worst_pred_idx = np.argmax(pred_error)
    print(f"\n  Worst prediction error at index {worst_pred_idx}:")
    print(f"    Coarse value: {coarse_pressure[worst_pred_idx]:.6f}")
    print(f"    Fine value:   {fine_pressure[worst_pred_idx]:.6f}")
    print(f"    Pred value:   {pred_pressure[worst_pred_idx]:.6f}")
    
    return mesh

# Run analysis
mesh = analyze_test_results()

# Let's also check what the training loss was
print("\n" + "="*60)
print("CHECKING TRAINING HISTORY")
print("="*60)

import glob
log_files = glob.glob("outputs/Ahmed_Dataset/enhanced_v2_physics/*.log")
if log_files:
    print(f"Found {len(log_files)} log files")
    # Check the training log specifically
    train_log = "outputs/Ahmed_Dataset/enhanced_v2_physics/train.log"
    if Path(train_log).exists():
        with open(train_log, 'r') as f:
            lines = f.readlines()
            # Look for the last epoch's performance
            for i, line in enumerate(lines[-100:]):
                if 'EPOCH' in line and 'SUMMARY' in line:
                    # Print this and next few lines
                    for j in range(min(10, len(lines) - i)):
                        print(lines[-100+i+j].strip())
                    break