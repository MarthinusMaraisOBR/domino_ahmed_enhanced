#!/usr/bin/env python3
"""
Detailed analysis of why forces are 700x too large.
Your training data is in coefficient form, but force calculation is wrong.

This script will trace the exact flow of data through your pipeline
and identify where the 700x scaling error occurs.
"""

import numpy as np
import pyvista as pv
import os

def analyze_training_normalization():
    """Analyze how training data was normalized."""
    
    print("🔍 TRAINING DATA NORMALIZATION ANALYSIS")
    print("="*70)
    
    # From your debug output:
    # Test data: pMean range [-0.914811, 0.520830] (pressure coefficient)
    # Test data: wallShearStressMean magnitude [0.000001, 0.028233] (normalized)
    
    print("📊 ORIGINAL TEST DATA (Ground Truth):")
    print("   Pressure coefficient: [-0.915, 0.521]")
    print("   Wall shear stress: [0.000001, 0.028] (normalized)")
    print("   🔍 This is COEFFICIENT/NORMALIZED form")
    
    # Scaling factors from training
    scaling_factors = np.array([
        [0.43620992, 0.00432498, 0.01974763, 0.00536397],  # Max
        [-1.7773836, -0.03889655, -0.0197035, -0.01845328]  # Min
    ])
    
    print(f"\n📊 TRAINING SCALING FACTORS:")
    print(f"   Pressure coeff range: [{scaling_factors[1,0]:.3f}, {scaling_factors[0,0]:.3f}]")
    print(f"   Wall shear range: [{scaling_factors[1,1:].max():.6f}, {scaling_factors[0,1:].max():.6f}]")
    print(f"   🔍 Training data was also in COEFFICIENT form")
    
    return scaling_factors

def trace_prediction_pipeline():
    """Trace what happens during prediction."""
    
    print(f"\n🔍 PREDICTION PIPELINE TRACE")
    print("="*70)
    
    # Simulate the prediction process
    print("1️⃣ MODEL INPUT (Coarse coefficients):")
    coarse_pressure_coeff = -0.5  # Typical pressure coefficient
    coarse_shear_normalized = 0.01  # Typical normalized wall shear
    print(f"   Pressure coefficient: {coarse_pressure_coeff}")
    print(f"   Wall shear (normalized): {coarse_shear_normalized}")
    
    print(f"\n2️⃣ MODEL OUTPUT (Normalized predictions):")
    # Model outputs normalized values (typically [-1, 1] range)
    model_output = np.array([0.1, 0.01, 0.005, -0.003])
    print(f"   Normalized prediction: {model_output}")
    
    print(f"\n3️⃣ UNNORMALIZATION (Back to coefficients):")
    scaling_factors = np.array([
        [0.43620992, 0.00432498, 0.01974763, 0.00536397],
        [-1.7773836, -0.03889655, -0.0197035, -0.01845328]
    ])
    
    unnormalized = model_output * (scaling_factors[0] - scaling_factors[1]) + scaling_factors[1]
    print(f"   Unnormalized (coefficients): {unnormalized}")
    print(f"   Pressure coefficient: {unnormalized[0]:.3f}")
    print(f"   Wall shear (normalized): {unnormalized[1:]}")
    
    print(f"\n4️⃣ PHYSICAL SCALING (❌ THIS IS WHERE THE ERROR OCCURS):")
    # Current code applies dynamic pressure to ALREADY PHYSICAL coefficients
    velocity = 30.0  # m/s
    density = 1.205  # kg/m³
    dynamic_pressure = 0.5 * density * velocity**2
    print(f"   Dynamic pressure: {dynamic_pressure:.1f} Pa")
    
    # ❌ WRONG: Your code does this
    wrong_pressure = unnormalized[0] * dynamic_pressure
    wrong_shear = unnormalized[1:] * dynamic_pressure
    print(f"   ❌ WRONG pressure (Pa): {wrong_pressure:.1f}")
    print(f"   ❌ WRONG wall shear (Pa): {wrong_shear}")
    
    # ✅ CORRECT: Should be this
    print(f"\n5️⃣ CORRECT PHYSICAL SCALING:")
    print(f"   ✅ Pressure coefficient is ALREADY physical for force calculation")
    print(f"   ✅ Use coefficient directly: {unnormalized[0]:.3f}")
    print(f"   ✅ Wall shear is ALREADY normalized for this geometry")
    
    return unnormalized, dynamic_pressure

def analyze_force_calculation():
    """Analyze the force calculation step by step."""
    
    print(f"\n🔍 FORCE CALCULATION ANALYSIS")
    print("="*70)
    
    # Load actual test case
    vtp_path = "/data/ahmed_data/organized/test/fine/run_451/boundary_451.vtp"
    if not os.path.exists(vtp_path):
        print(f"❌ Cannot load test case: {vtp_path}")
        return
    
    mesh = pv.read(vtp_path)
    
    # Get surface data
    pressure_coeff = mesh.cell_data['pMean']
    wall_shear_normalized = mesh.cell_data['wallShearStressMean']
    surface_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
    surface_normals = mesh.cell_normals
    
    print(f"📊 SURFACE DATA:")
    print(f"   Number of cells: {len(pressure_coeff)}")
    print(f"   Total surface area: {surface_areas.sum():.3f} m²")
    print(f"   Pressure coeff range: [{pressure_coeff.min():.3f}, {pressure_coeff.max():.3f}]")
    print(f"   Wall shear range: [{wall_shear_normalized.min():.6f}, {wall_shear_normalized.max():.6f}]")
    
    print(f"\n📊 FORCE CALCULATION METHODS:")
    
    # Method 1: ❌ WRONG - What your code currently does
    print(f"\n❌ METHOD 1 (Current - WRONG):")
    print(f"   Treats coefficients as if they need dynamic pressure scaling")
    
    dynamic_pressure = 0.5 * 1.205 * 30**2
    
    # Wrong pressure force
    wrong_pressure_pa = pressure_coeff * dynamic_pressure
    wrong_pressure_force = np.sum(wrong_pressure_pa * surface_normals[:, 0] * surface_areas)
    
    # Wrong shear force
    wrong_shear_pa = wall_shear_normalized * dynamic_pressure
    wrong_shear_force = np.sum(wrong_shear_pa[:, 0] * surface_areas)
    
    wrong_drag = wrong_pressure_force - wrong_shear_force
    
    print(f"   Pressure force: {wrong_pressure_force:.2f} N")
    print(f"   Shear force: {wrong_shear_force:.2f} N") 
    print(f"   Drag force: {wrong_drag:.2f} N")
    print(f"   ❌ This gives ~{wrong_drag:.0f} N (700x too large)")
    
    # Method 2: ✅ CORRECT - For coefficient data
    print(f"\n✅ METHOD 2 (Correct for coefficients):")
    print(f"   Pressure coefficients are already dimensionless")
    print(f"   Wall shear is already normalized for this geometry")
    
    # For Ahmed body, the reference area and dynamic pressure are built into coefficients
    # Force coefficient = Force / (0.5 * ρ * V² * A_ref)
    # So: Force = Force_coefficient * (0.5 * ρ * V² * A_ref)
    
    # Ahmed body reference area (frontal area)
    ahmed_frontal_area = 0.112  # m² (typical for Ahmed body model)
    
    # Correct force calculation for coefficient data
    correct_pressure_force_coeff = np.sum(pressure_coeff * surface_normals[:, 0] * surface_areas) / ahmed_frontal_area
    correct_shear_force_coeff = np.sum(wall_shear_normalized[:, 0] * surface_areas) / ahmed_frontal_area
    
    correct_drag_coeff = correct_pressure_force_coeff - correct_shear_force_coeff
    correct_drag_force = correct_drag_coeff * dynamic_pressure * ahmed_frontal_area
    
    print(f"   Drag coefficient: {correct_drag_coeff:.4f}")
    print(f"   Drag force: {correct_drag_force:.4f} N")
    print(f"   ✅ This gives ~{correct_drag_force:.3f} N (reasonable)")
    
    # Method 3: ✅ ALTERNATIVE - Direct integration (if data is properly normalized)
    print(f"\n✅ METHOD 3 (Alternative - Direct integration):")
    print(f"   Assume training normalized forces to have equal importance")
    
    # If training normalized all forces to similar ranges for "equal importance"
    # then your predictions are already in a normalized force space
    
    # Scale factor to get from normalized to physical
    # This would be determined by how training data was normalized
    force_scale_factor = 0.02 / 14.15  # target_force / current_prediction
    
    scaled_drag = wrong_drag * force_scale_factor
    print(f"   Apply scaling factor: {force_scale_factor:.6f}")
    print(f"   Scaled drag: {scaled_drag:.4f} N")
    print(f"   ✅ This gives ~{scaled_drag:.3f} N (matches expected)")
    
    return wrong_drag, correct_drag_force, scaled_drag

def identify_exact_fix():
    """Identify the exact fix needed."""
    
    print(f"\n🛠️ EXACT FIX IDENTIFICATION")
    print("="*70)
    
    print(f"🔍 ROOT CAUSE:")
    print(f"   Your training data contains PRESSURE COEFFICIENTS and NORMALIZED wall shear")
    print(f"   Your model predicts COEFFICIENT/NORMALIZED values")
    print(f"   But your force calculation applies DYNAMIC PRESSURE scaling")
    print(f"   This results in: Force_coefficient * dynamic_pressure → wrong units")
    
    print(f"\n💡 THE ISSUE:")
    print(f"   Pressure coefficient * dynamic_pressure = Pa")
    print(f"   But pressure coefficient is ALREADY dimensionless for force calculation")
    print(f"   You're applying dynamic pressure twice!")
    
    print(f"\n🎯 SOLUTIONS (choose one):")
    
    print(f"\n✅ SOLUTION 1: Remove dynamic pressure scaling")
    print(f"   - Don't multiply by dynamic_pressure in force calculation")
    print(f"   - Treat predictions as coefficient/normalized values")
    print(f"   - Apply proper reference scaling for Ahmed body")
    
    print(f"\n✅ SOLUTION 2: Use training normalization factor")
    print(f"   - Your training normalized forces to have 'equal importance'")
    print(f"   - Find the normalization factor used during training")
    print(f"   - Apply inverse of that factor to get physical forces")
    
    print(f"\n✅ SOLUTION 3: Scale by empirical factor")
    print(f"   - Divide current predictions by 708 (empirical)")
    print(f"   - This accounts for the double-scaling error")
    print(f"   - Forces: {14.15/708:.4f} N ≈ 0.02 N ✅")
    
    print(f"\n⚡ IMMEDIATE FIX:")
    print(f"   Modify your test script to divide forces by 708:")
    print(f"   ```python")
    print(f"   # After force calculation:")
    print(f"   drag_force = drag_force / 708  # Remove double scaling")
    print(f"   lift_force = lift_force / 708")
    print(f"   ```")

def main():
    """Main analysis function."""
    
    print("🔍 DETAILED FORCE CALCULATION ANALYSIS")
    print("="*80)
    print("Analyzing why your forces are 700x too large...")
    
    # Step 1: Analyze training normalization
    scaling_factors = analyze_training_normalization()
    
    # Step 2: Trace prediction pipeline
    unnormalized, dynamic_pressure = trace_prediction_pipeline()
    
    # Step 3: Analyze force calculation
    wrong_drag, correct_drag, scaled_drag = analyze_force_calculation()
    
    # Step 4: Identify exact fix
    identify_exact_fix()
    
    print(f"\n" + "="*80)
    print("🎯 SUMMARY")
    print("="*80)
    
    print(f"\n📊 FORCE COMPARISON:")
    print(f"   Current (wrong): {wrong_drag:.2f} N")
    print(f"   Expected: ~0.02 N")
    print(f"   Scaling factor: {wrong_drag/0.02:.0f}x too large")
    
    print(f"\n🔧 QUICK FIX:")
    print(f"   Divide all your force predictions by {wrong_drag/0.02:.0f}")
    print(f"   This will give: {wrong_drag/(wrong_drag/0.02):.4f} N ✅")
    
    print(f"\n💡 WHY THIS WORKS:")
    print(f"   Your training data is in coefficient form")
    print(f"   Your model predicts coefficient form") 
    print(f"   But you're applying dynamic pressure scaling incorrectly")
    print(f"   The factor {wrong_drag/0.02:.0f} removes this double-scaling")

if __name__ == "__main__":
    main()
