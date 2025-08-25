#!/usr/bin/env python3
"""
Diagnostic script to investigate why drag predictions are worse than baseline
while lift predictions are better.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_force_predictions(case_name="run_451"):
    """Analyze the force predictions in detail."""
    
    print("="*80)
    print(f"FORCE PREDICTION DIAGNOSTIC FOR {case_name}")
    print("="*80)
    
    # Load the comprehensive VTP file
    vtp_path = Path(f"/data/ahmed_data/predictions_fixed/boundary_{case_name.split('_')[1]}_comprehensive_comparison.vtp")
    
    if not vtp_path.exists():
        print(f"File not found: {vtp_path}")
        return
    
    mesh = pv.read(str(vtp_path))
    print(f"\nLoaded mesh with {mesh.n_cells} cells")
    
    # Extract fields
    coarse_pressure = mesh['Coarse_Pressure']
    fine_pressure = mesh['Fine_Pressure_GroundTruth_Interpolated']
    pred_pressure = mesh['Predicted_Pressure']
    
    coarse_shear = mesh['Coarse_WallShearStress']
    fine_shear = mesh['Fine_WallShearStress_GroundTruth_Interpolated']
    pred_shear = mesh['Predicted_WallShearStress']
    
    # Get geometry data
    normals = np.array(mesh.cell_normals)
    areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)['Area']
    
    print("\n" + "="*60)
    print("FIELD STATISTICS")
    print("="*60)
    
    # Pressure statistics
    print("\nPressure Statistics:")
    print(f"  Coarse: mean={coarse_pressure.mean():.6f}, std={coarse_pressure.std():.6f}")
    print(f"          min={coarse_pressure.min():.6f}, max={coarse_pressure.max():.6f}")
    print(f"  Fine:   mean={fine_pressure.mean():.6f}, std={fine_pressure.std():.6f}")
    print(f"          min={fine_pressure.min():.6f}, max={fine_pressure.max():.6f}")
    print(f"  Pred:   mean={pred_pressure.mean():.6f}, std={pred_pressure.std():.6f}")
    print(f"          min={pred_pressure.min():.6f}, max={pred_pressure.max():.6f}")
    
    # Wall shear stress statistics (X-component for drag)
    print("\nWall Shear Stress X (Drag component):")
    print(f"  Coarse: mean={coarse_shear[:, 0].mean():.6f}, std={coarse_shear[:, 0].std():.6f}")
    print(f"          min={coarse_shear[:, 0].min():.6f}, max={coarse_shear[:, 0].max():.6f}")
    print(f"  Fine:   mean={fine_shear[:, 0].mean():.6f}, std={fine_shear[:, 0].std():.6f}")
    print(f"          min={fine_shear[:, 0].min():.6f}, max={fine_shear[:, 0].max():.6f}")
    print(f"  Pred:   mean={pred_shear[:, 0].mean():.6f}, std={pred_shear[:, 0].std():.6f}")
    print(f"          min={pred_shear[:, 0].min():.6f}, max={pred_shear[:, 0].max():.6f}")
    
    # Check correlations
    print("\n" + "="*60)
    print("CORRELATIONS")
    print("="*60)
    
    print("\nPressure Correlations:")
    print(f"  Coarse vs Fine: {np.corrcoef(coarse_pressure, fine_pressure)[0,1]:.4f}")
    print(f"  Pred vs Fine:   {np.corrcoef(pred_pressure, fine_pressure)[0,1]:.4f}")
    print(f"  Pred vs Coarse: {np.corrcoef(pred_pressure, coarse_pressure)[0,1]:.4f}")
    
    print("\nWall Shear X Correlations:")
    print(f"  Coarse vs Fine: {np.corrcoef(coarse_shear[:,0], fine_shear[:,0])[0,1]:.4f}")
    print(f"  Pred vs Fine:   {np.corrcoef(pred_shear[:,0], fine_shear[:,0])[0,1]:.4f}")
    print(f"  Pred vs Coarse: {np.corrcoef(pred_shear[:,0], coarse_shear[:,0])[0,1]:.4f}")
    
    # Detailed force calculation
    print("\n" + "="*60)
    print("FORCE CALCULATION BREAKDOWN")
    print("="*60)
    
    # Drag force: Fx = ∫(p·nx - τx)dA
    print("\nDrag Force Components:")
    
    # Pressure contribution to drag
    pressure_drag_coarse = np.sum(coarse_pressure * normals[:, 0] * areas)
    pressure_drag_fine = np.sum(fine_pressure * normals[:, 0] * areas)
    pressure_drag_pred = np.sum(pred_pressure * normals[:, 0] * areas)
    
    print(f"  Pressure contribution:")
    print(f"    Coarse: {pressure_drag_coarse:.6f}")
    print(f"    Fine:   {pressure_drag_fine:.6f}")
    print(f"    Pred:   {pressure_drag_pred:.6f}")
    
    # Shear contribution to drag
    shear_drag_coarse = np.sum(coarse_shear[:, 0] * areas)
    shear_drag_fine = np.sum(fine_shear[:, 0] * areas)
    shear_drag_pred = np.sum(pred_shear[:, 0] * areas)
    
    print(f"  Shear contribution:")
    print(f"    Coarse: {shear_drag_coarse:.6f}")
    print(f"    Fine:   {shear_drag_fine:.6f}")
    print(f"    Pred:   {shear_drag_pred:.6f}")
    
    # Total drag
    drag_coarse = pressure_drag_coarse - shear_drag_coarse
    drag_fine = pressure_drag_fine - shear_drag_fine
    drag_pred = pressure_drag_pred - shear_drag_pred
    
    print(f"  Total Drag (pressure - shear):")
    print(f"    Coarse: {drag_coarse:.6f}")
    print(f"    Fine:   {drag_fine:.6f}")
    print(f"    Pred:   {drag_pred:.6f}")
    
    # Calculate improvements
    baseline_error = abs(drag_coarse - drag_fine)
    pred_error = abs(drag_pred - drag_fine)
    improvement = (baseline_error - pred_error) / baseline_error * 100
    
    print(f"\n  Drag Improvement: {improvement:.1f}%")
    if improvement < 0:
        print("  ⚠️ PREDICTION WORSE THAN BASELINE")
    
    # Check if the issue is in pressure or shear
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    # Check component-wise errors
    pressure_baseline_error = abs(pressure_drag_coarse - pressure_drag_fine)
    pressure_pred_error = abs(pressure_drag_pred - pressure_drag_fine)
    pressure_improvement = (pressure_baseline_error - pressure_pred_error) / pressure_baseline_error * 100
    
    shear_baseline_error = abs(shear_drag_coarse - shear_drag_fine)
    shear_pred_error = abs(shear_drag_pred - shear_drag_fine)
    shear_improvement = (shear_baseline_error - shear_pred_error) / shear_baseline_error * 100
    
    print(f"\nComponent-wise improvements:")
    print(f"  Pressure component: {pressure_improvement:.1f}%")
    print(f"  Shear component:    {shear_improvement:.1f}%")
    
    if pressure_improvement < -50:
        print("\n⚠️ ISSUE: Pressure prediction is much worse than baseline")
    if shear_improvement < -50:
        print("\n⚠️ ISSUE: Shear stress prediction is much worse than baseline")
    
    # Check for sign flips or magnitude issues
    print("\n" + "="*60)
    print("POTENTIAL ISSUES")
    print("="*60)
    
    # Check if predictions have wrong sign
    pressure_sign_flips = np.sum(np.sign(pred_pressure) != np.sign(fine_pressure))
    shear_sign_flips = np.sum(np.sign(pred_shear[:,0]) != np.sign(fine_shear[:,0]))
    
    print(f"\nSign flips (prediction vs fine):")
    print(f"  Pressure: {pressure_sign_flips}/{len(pred_pressure)} cells")
    print(f"  Shear X:  {shear_sign_flips}/{len(pred_shear)} cells")
    
    if pressure_sign_flips > len(pred_pressure) * 0.1:
        print("  ⚠️ Many pressure values have wrong sign!")
    if shear_sign_flips > len(pred_shear) * 0.1:
        print("  ⚠️ Many shear stress values have wrong sign!")
    
    # Check if predictions are scaled wrong
    pressure_scale_ratio = np.std(pred_pressure) / np.std(fine_pressure)
    shear_scale_ratio = np.std(pred_shear[:,0]) / np.std(fine_shear[:,0])
    
    print(f"\nScale ratios (pred_std/fine_std):")
    print(f"  Pressure: {pressure_scale_ratio:.3f}")
    print(f"  Shear X:  {shear_scale_ratio:.3f}")
    
    if pressure_scale_ratio < 0.5 or pressure_scale_ratio > 2.0:
        print("  ⚠️ Pressure predictions have wrong scale!")
    if shear_scale_ratio < 0.5 or shear_scale_ratio > 2.0:
        print("  ⚠️ Shear predictions have wrong scale!")
    
    # Create diagnostic plots
    create_diagnostic_plots(
        coarse_pressure, fine_pressure, pred_pressure,
        coarse_shear[:,0], fine_shear[:,0], pred_shear[:,0],
        case_name
    )
    
    return {
        'drag_improvement': improvement,
        'pressure_improvement': pressure_improvement,
        'shear_improvement': shear_improvement,
        'pressure_scale_ratio': pressure_scale_ratio,
        'shear_scale_ratio': shear_scale_ratio
    }

def create_diagnostic_plots(coarse_p, fine_p, pred_p, coarse_s, fine_s, pred_s, case_name):
    """Create diagnostic scatter plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sample points for visualization
    n_points = min(1000, len(coarse_p))
    idx = np.random.choice(len(coarse_p), n_points, replace=False)
    
    # Pressure plots
    axes[0,0].scatter(fine_p[idx], coarse_p[idx], alpha=0.5, s=1)
    axes[0,0].plot([fine_p.min(), fine_p.max()], [fine_p.min(), fine_p.max()], 'r--')
    axes[0,0].set_xlabel('Fine Pressure')
    axes[0,0].set_ylabel('Coarse Pressure')
    axes[0,0].set_title('Coarse vs Fine (Baseline)')
    
    axes[0,1].scatter(fine_p[idx], pred_p[idx], alpha=0.5, s=1)
    axes[0,1].plot([fine_p.min(), fine_p.max()], [fine_p.min(), fine_p.max()], 'r--')
    axes[0,1].set_xlabel('Fine Pressure')
    axes[0,1].set_ylabel('Predicted Pressure')
    axes[0,1].set_title('Predicted vs Fine')
    
    axes[0,2].scatter(coarse_p[idx], pred_p[idx], alpha=0.5, s=1)
    axes[0,2].plot([coarse_p.min(), coarse_p.max()], [coarse_p.min(), coarse_p.max()], 'r--')
    axes[0,2].set_xlabel('Coarse Pressure')
    axes[0,2].set_ylabel('Predicted Pressure')
    axes[0,2].set_title('Predicted vs Coarse (Input)')
    
    # Shear stress plots
    axes[1,0].scatter(fine_s[idx], coarse_s[idx], alpha=0.5, s=1)
    axes[1,0].plot([fine_s.min(), fine_s.max()], [fine_s.min(), fine_s.max()], 'r--')
    axes[1,0].set_xlabel('Fine Shear X')
    axes[1,0].set_ylabel('Coarse Shear X')
    axes[1,0].set_title('Coarse vs Fine (Baseline)')
    
    axes[1,1].scatter(fine_s[idx], pred_s[idx], alpha=0.5, s=1)
    axes[1,1].plot([fine_s.min(), fine_s.max()], [fine_s.min(), fine_s.max()], 'r--')
    axes[1,1].set_xlabel('Fine Shear X')
    axes[1,1].set_ylabel('Predicted Shear X')
    axes[1,1].set_title('Predicted vs Fine')
    
    axes[1,2].scatter(coarse_s[idx], pred_s[idx], alpha=0.5, s=1)
    axes[1,2].plot([coarse_s.min(), coarse_s.max()], [coarse_s.min(), coarse_s.max()], 'r--')
    axes[1,2].set_xlabel('Coarse Shear X')
    axes[1,2].set_ylabel('Predicted Shear X')
    axes[1,2].set_title('Predicted vs Coarse (Input)')
    
    plt.suptitle(f'Diagnostic Plots for {case_name}')
    plt.tight_layout()
    plt.savefig(f'diagnostic_{case_name}.png', dpi=150)
    print(f"\n✅ Diagnostic plots saved to diagnostic_{case_name}.png")

def check_scaling_factors():
    """Check if scaling factors are consistent."""
    
    print("\n" + "="*80)
    print("CHECKING SCALING FACTORS")
    print("="*80)
    
    # Check training scaling factors
    train_scale_path = Path("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors.npy")
    if train_scale_path.exists():
        train_scales = np.load(train_scale_path)
        print("\nTraining scaling factors:")
        print(f"  Shape: {train_scales.shape}")
        print(f"  Max values: {train_scales[0]}")
        print(f"  Min values: {train_scales[1]}")
    else:
        print("⚠️ Training scaling factors not found!")
    
    # Check inference scaling factors
    test_scale_path = Path("outputs/Ahmed_Dataset/enhanced_fixed/surface_scaling_factors_inference.npy")
    if test_scale_path.exists():
        test_scales = np.load(test_scale_path)
        print("\nInference scaling factors:")
        print(f"  Shape: {test_scales.shape}")
        print(f"  Max values: {test_scales[0]}")
        print(f"  Min values: {test_scales[1]}")
        
        if train_scale_path.exists():
            diff = np.abs(train_scales - test_scales)
            if np.max(diff) > 1e-3:
                print("\n⚠️ SCALING FACTORS MISMATCH!")
                print(f"  Maximum difference: {np.max(diff)}")
    else:
        print("⚠️ Inference scaling factors not found!")

if __name__ == "__main__":
    # Analyze first test case in detail
    results = analyze_force_predictions("run_451")
    
    # Check scaling factors
    check_scaling_factors()
    
    # Analyze all test cases
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL TEST CASES")
    print("="*80)
    
    all_results = []
    for i in range(451, 456):
        case = f"run_{i}"
        print(f"\nAnalyzing {case}...")
        result = analyze_force_predictions(case)
        if result:
            all_results.append(result)
    
    if all_results:
        avg_drag_imp = np.mean([r['drag_improvement'] for r in all_results])
        avg_pressure_imp = np.mean([r['pressure_improvement'] for r in all_results])
        avg_shear_imp = np.mean([r['shear_improvement'] for r in all_results])
        
        print("\n" + "="*60)
        print("OVERALL DIAGNOSIS")
        print("="*60)
        print(f"Average drag improvement: {avg_drag_imp:.1f}%")
        print(f"Average pressure component improvement: {avg_pressure_imp:.1f}%")
        print(f"Average shear component improvement: {avg_shear_imp:.1f}%")
        
        if avg_pressure_imp < -50:
            print("\n🔴 MAIN ISSUE: Pressure predictions are problematic")
        if avg_shear_imp < -50:
            print("\n🔴 MAIN ISSUE: Shear stress predictions are problematic")
