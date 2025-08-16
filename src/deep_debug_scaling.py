#!/usr/bin/env python3
"""
Deep debug the Enhanced DoMINO scaling issue.
The patch didn't work - forces are still ~14 N instead of ~0.02 N.
This means there's a fundamental scaling problem we need to trace.
"""

import numpy as np
import torch
import os
import pyvista as pv
from pathlib import Path

def analyze_scaling_factors():
    """Analyze all scaling factor files to understand the problem."""
    
    print("🔍 DEEP SCALING FACTOR ANALYSIS")
    print("="*60)
    
    # Check all scaling factor files
    paths = [
        "outputs/Ahmed_Dataset/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
    ]
    
    for path in paths:
        if os.path.exists(path):
            factors = np.load(path)
            print(f"\n📁 {path}")
            print(f"   Shape: {factors.shape}")
            print(f"   Max values: {factors[0]}")
            print(f"   Min values: {factors[1]}")
            print(f"   Range (max-min): {factors[0] - factors[1]}")
            
            # Check if these look like physical values or coefficients
            max_pressure = factors[0, 0]
            max_shear = np.max(factors[0, 1:4])
            
            if max_pressure > 100:
                print(f"   ⚠️  Pressure values look like Pa (physical), not coefficients")
            elif max_pressure > 10:
                print(f"   ⚠️  Pressure values very large for coefficients")
            else:
                print(f"   ✅ Pressure values look like coefficients")
                
            if max_shear > 1:
                print(f"   ⚠️  Shear stress values look like Pa (physical)")
            else:
                print(f"   ✅ Shear stress values look reasonable")


def analyze_test_data():
    """Analyze the actual test data ranges."""
    
    print(f"\n🔍 TEST DATA ANALYSIS")
    print("="*60)
    
    # Load a test case
    test_vtp = "/data/ahmed_data/organized/test/fine/run_451/boundary_451.vtp"
    
    if not os.path.exists(test_vtp):
        print(f"❌ Cannot find test data: {test_vtp}")
        return
    
    mesh = pv.read(test_vtp)
    print(f"Test case: {test_vtp}")
    print(f"Cell data keys: {list(mesh.cell_data.keys())}")
    
    # Find pressure field
    pressure_field = None
    for name in ['pMean', 'pressure', 'Pressure', 'p']:
        if name in mesh.cell_data:
            pressure_field = mesh.cell_data[name]
            print(f"\n📊 Pressure field ({name}):")
            print(f"   Range: [{pressure_field.min():.6f}, {pressure_field.max():.6f}]")
            print(f"   Mean: {pressure_field.mean():.6f}")
            print(f"   Std: {pressure_field.std():.6f}")
            
            # Determine if this is coefficient or physical
            if np.abs(pressure_field).max() > 100:
                print(f"   🔍 This looks like PHYSICAL pressure (Pa)")
            elif np.abs(pressure_field).max() > 10:
                print(f"   🔍 This might be GAUGE pressure")
            else:
                print(f"   🔍 This looks like PRESSURE COEFFICIENT")
            break
    
    # Find wall shear stress
    shear_field = None
    for name in ['wallShearStressMean', 'wallShearStress', 'WallShearStress']:
        if name in mesh.cell_data:
            shear_field = mesh.cell_data[name]
            shear_mag = np.sqrt(np.sum(shear_field**2, axis=1))
            print(f"\n📊 Wall shear stress ({name}):")
            print(f"   Magnitude range: [{shear_mag.min():.6f}, {shear_mag.max():.6f}]")
            print(f"   Mean magnitude: {shear_mag.mean():.6f}")
            
            if shear_mag.max() > 1:
                print(f"   🔍 This looks like PHYSICAL wall shear stress (Pa)")
            else:
                print(f"   🔍 This looks like NORMALIZED wall shear stress")
            break
    
    return pressure_field, shear_field


def trace_unnormalization():
    """Trace through the unnormalization process step by step."""
    
    print(f"\n🔍 UNNORMALIZATION TRACE")
    print("="*60)
    
    # Load inference scaling factors
    factors_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
    if not os.path.exists(factors_path):
        print(f"❌ No inference factors: {factors_path}")
        return
    
    factors = np.load(factors_path)
    print(f"Inference scaling factors:")
    print(f"   Max: {factors[0]}")
    print(f"   Min: {factors[1]}")
    print(f"   Range: {factors[0] - factors[1]}")
    
    # Simulate the unnormalization process
    print(f"\n🧪 SIMULATING UNNORMALIZATION:")
    
    # Typical model output (normalized predictions)
    # These should be roughly in [-1, 1] or [0, 1] range
    sample_normalized = np.array([0.1, 0.01, 0.005, -0.003])
    print(f"1. Sample normalized prediction: {sample_normalized}")
    
    # Apply unnormalization: data * (max - min) + min
    unnormalized = sample_normalized * (factors[0] - factors[1]) + factors[1]
    print(f"2. After unnormalization: {unnormalized}")
    
    # Check if this needs physical scaling
    print(f"3. Checking if physical scaling needed...")
    
    # Ahmed body at 30 m/s, ρ=1.205 kg/m³
    dynamic_pressure = 0.5 * 1.205 * 30**2
    print(f"   Dynamic pressure: {dynamic_pressure:.1f} Pa")
    
    # If values are coefficients, scale by dynamic pressure
    if np.abs(unnormalized[0]) < 5:  # Pressure coefficient typically < 5
        physical_pressure = unnormalized[0] * dynamic_pressure
        print(f"4. Pressure coefficient {unnormalized[0]:.3f} → {physical_pressure:.1f} Pa")
    else:
        print(f"4. Pressure already in Pa: {unnormalized[0]:.1f}")
    
    # Wall shear stress
    if np.abs(unnormalized[1:]).max() < 0.1:  # Typical wall shear range
        physical_shear = unnormalized[1:] * dynamic_pressure
        print(f"5. Wall shear coeff {unnormalized[1:]} → {physical_shear} Pa")
    else:
        print(f"5. Wall shear already in Pa: {unnormalized[1:]}")


def check_force_calculation():
    """Check if the force calculation itself is wrong."""
    
    print(f"\n🔍 FORCE CALCULATION CHECK")
    print("="*60)
    
    # Typical Ahmed body force values (experimental)
    expected_drag = 0.02  # N (for model scale)
    expected_lift = 0.01  # N
    
    print(f"Expected forces for Ahmed body:")
    print(f"   Drag: ~{expected_drag:.3f} N")
    print(f"   Lift: ~{expected_lift:.3f} N")
    
    # Your predictions
    predicted_drag = 14.15  # From your output
    predicted_lift = -13.49
    
    print(f"\nYour predicted forces:")
    print(f"   Drag: {predicted_drag:.2f} N")
    print(f"   Lift: {predicted_lift:.2f} N")
    
    # Calculate scaling factor
    scale_factor = predicted_drag / expected_drag
    print(f"\nScaling factor: {scale_factor:.0f}x too large")
    
    # This suggests either:
    print(f"\n🔍 POSSIBLE CAUSES:")
    print(f"1. Surface area wrong by factor of {scale_factor:.0f}")
    print(f"2. Pressure values wrong by factor of {scale_factor:.0f}")
    print(f"3. Physical constants wrong (velocity, density)")
    print(f"4. Units mismatch (mm vs m, etc.)")


def check_mesh_scaling():
    """Check if there's a mesh scaling issue."""
    
    print(f"\n🔍 MESH SCALING CHECK")
    print("="*60)
    
    # Load STL to check geometry scale
    stl_path = "/data/ahmed_data/organized/test/fine/run_451/ahmed_451.stl"
    
    if os.path.exists(stl_path):
        stl_mesh = pv.read(stl_path)
        bounds = stl_mesh.bounds
        
        # Ahmed body is typically 1.044m long
        length = bounds[1] - bounds[0]  # x-direction
        width = bounds[3] - bounds[2]   # y-direction
        height = bounds[5] - bounds[4]  # z-direction
        
        print(f"Ahmed body dimensions:")
        print(f"   Length: {length:.3f} m")
        print(f"   Width: {width:.3f} m") 
        print(f"   Height: {height:.3f} m")
        
        # Check if this is model scale or full scale
        if length > 10:
            print(f"   🔍 This looks like FULL SCALE (or mm units)")
            scale_issue = True
        elif length > 1:
            print(f"   🔍 This looks like MODEL SCALE")
            scale_issue = False
        else:
            print(f"   🔍 This looks very small - check units")
            scale_issue = True
        
        # Calculate surface area
        stl_mesh = stl_mesh.compute_cell_sizes(length=False, area=True, volume=False)
        total_area = np.sum(stl_mesh.cell_data["Area"])
        
        print(f"   Total surface area: {total_area:.3f} m²")
        
        if total_area > 100:
            print(f"   ⚠️  Surface area very large - units issue?")
        elif total_area > 10:
            print(f"   ⚠️  Surface area large - full scale?")
        else:
            print(f"   ✅ Surface area reasonable for model scale")
        
        return scale_issue, total_area
    
    return False, 0


def main():
    """Main debugging function."""
    
    print("🚨 DEEP DEBUG: ENHANCED DOMINO STILL 700x TOO LARGE")
    print("="*80)
    print("Your forces are still ~14 N instead of ~0.02 N")
    print("Let's find the exact problem...")
    
    # Step 1: Analyze scaling factors
    analyze_scaling_factors()
    
    # Step 2: Analyze test data
    pressure_field, shear_field = analyze_test_data()
    
    # Step 3: Trace unnormalization
    trace_unnormalization()
    
    # Step 4: Check force calculation
    check_force_calculation()
    
    # Step 5: Check mesh scaling
    scale_issue, total_area = check_mesh_scaling()
    
    print(f"\n" + "="*80)
    print("🔍 DIAGNOSIS COMPLETE")
    print("="*80)
    
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print(f"Your forces are {14.15/0.02:.0f}x too large (~700x)")
    
    print(f"\n🔍 LIKELY CAUSES (in order of probability):")
    
    if scale_issue:
        print(f"1. 🚨 GEOMETRY SCALING ISSUE")
        print(f"   - Your geometry might be in wrong units (mm vs m)")
        print(f"   - Surface area is {total_area:.1f} m² (very large)")
        print(f"   - This could explain 100-1000x error")
    
    print(f"2. 🚨 PRESSURE UNIT MISMATCH")
    print(f"   - Model might output physical pressure (Pa) not coefficients")
    print(f"   - Check if training data was in Pa or coefficients")
    
    print(f"3. 🚨 MISSING NORMALIZATION STEP")
    print(f"   - Scaling factors might be wrong")
    print(f"   - Model might need different unnormalization")
    
    print(f"4. 🚨 PHYSICAL CONSTANTS WRONG")
    print(f"   - Check velocity (30 m/s?) and density (1.205 kg/m³)")
    print(f"   - Dynamic pressure should be {0.5*1.205*30**2:.1f} Pa")
    
    print(f"\n🛠️  IMMEDIATE ACTIONS:")
    print(f"1. Check geometry units - is Ahmed body ~1m or ~1000mm?")
    print(f"2. Verify what training data contained (Pa or coefficients)")
    print(f"3. Check if model needs geometry normalization")
    print(f"4. Try dividing final forces by 1000 (unit conversion)")
    
    print(f"\n⚡ QUICK TEST:")
    print(f"Try multiplying your forces by these factors:")
    print(f"   Forces / 1000 = {14.15/1000:.3f} N (if mm→m conversion)")
    print(f"   Forces / 541 = {14.15/541:.3f} N (if dynamic pressure issue)")
    
    if 14.15/1000 > 0.01 and 14.15/1000 < 0.1:
        print(f"   ✅ /1000 gives reasonable result - likely units issue!")


if __name__ == "__main__":
    main()
