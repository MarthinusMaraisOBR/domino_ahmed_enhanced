#!/usr/bin/env python3
"""
Create proper scaling factors for Enhanced DoMINO inference
This script computes scaling factors from training data if they don't exist
"""

import numpy as np
import torch
import h5py
from pathlib import Path
import os

def compute_scaling_factors_from_data():
    """
    Compute scaling factors from the actual training data
    """
    print("=" * 80)
    print("COMPUTING SCALING FACTORS FROM TRAINING DATA")
    print("=" * 80)
    
    # Path to training data
    data_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data")
    
    # Collect statistics from training data (cases 1-400)
    all_fine = []
    all_coarse = []
    
    print("\nAnalyzing training data...")
    for i in range(1, 401, 20):  # Sample every 20th file for efficiency
        h5_file = data_dir / f"case_{i}.h5"
        
        if h5_file.exists():
            with h5py.File(h5_file, 'r') as f:
                if 'enhanced_data' in f:
                    # Enhanced format: [fine_features, coarse_features]
                    data = f['enhanced_data'][:]
                    fine = data[:, :4]    # First 4 are fine
                    coarse = data[:, 4:8]  # Last 4 are coarse
                    all_fine.append(fine)
                    all_coarse.append(coarse)
                    print(f"  Case {i}: fine shape {fine.shape}, coarse shape {coarse.shape}")
                elif 'coarse_input' in f and 'fine_output' in f:
                    # Separate format
                    coarse = f['coarse_input'][:]
                    fine = f['fine_output'][:]
                    all_coarse.append(coarse)
                    all_fine.append(fine)
                    print(f"  Case {i}: separate format")
                elif 'coarse_input' in f:
                    # Only coarse available
                    coarse = f['coarse_input'][:]
                    all_coarse.append(coarse)
                    print(f"  Case {i}: coarse only")
    
    if not all_coarse:
        print("ERROR: No training data found!")
        return None
    
    # Concatenate all data
    if all_fine:
        fine_data = np.concatenate(all_fine, axis=0)
        print(f"\nFine data shape: {fine_data.shape}")
        fine_mean = np.mean(fine_data, axis=0)
        fine_std = np.std(fine_data, axis=0)
        print(f"Fine statistics:")
        print(f"  Pressure: mean={fine_mean[0]:.6f}, std={fine_std[0]:.6f}")
        print(f"  WSS_x:    mean={fine_mean[1]:.6f}, std={fine_std[1]:.6f}")
        print(f"  WSS_y:    mean={fine_mean[2]:.6f}, std={fine_std[2]:.6f}")
        print(f"  WSS_z:    mean={fine_mean[3]:.6f}, std={fine_std[3]:.6f}")
    else:
        fine_mean = None
        fine_std = None
    
    coarse_data = np.concatenate(all_coarse, axis=0)
    print(f"\nCoarse data shape: {coarse_data.shape}")
    coarse_mean = np.mean(coarse_data, axis=0)
    coarse_std = np.std(coarse_data, axis=0)
    print(f"Coarse statistics:")
    print(f"  Pressure: mean={coarse_mean[0]:.6f}, std={coarse_std[0]:.6f}")
    print(f"  WSS_x:    mean={coarse_mean[1]:.6f}, std={coarse_std[1]:.6f}")
    print(f"  WSS_y:    mean={coarse_mean[2]:.6f}, std={coarse_std[2]:.6f}")
    print(f"  WSS_z:    mean={coarse_mean[3]:.6f}, std={coarse_std[3]:.6f}")
    
    # Create scaling factors for enhanced training (8 features)
    if fine_mean is not None:
        # Enhanced format: [fine_features, coarse_features]
        scaling_enhanced = np.zeros((2, 8), dtype=np.float32)
        scaling_enhanced[0, :4] = fine_mean
        scaling_enhanced[0, 4:] = coarse_mean
        scaling_enhanced[1, :4] = fine_std
        scaling_enhanced[1, 4:] = coarse_std
        
        print(f"\nEnhanced scaling factors (8 features):")
        print(f"  Mean: {scaling_enhanced[0]}")
        print(f"  Std:  {scaling_enhanced[1]}")
    else:
        scaling_enhanced = None
    
    # Create scaling factors for inference (4 features - coarse only)
    scaling_inference = np.zeros((2, 4), dtype=np.float32)
    scaling_inference[0] = coarse_mean
    scaling_inference[1] = coarse_std
    
    print(f"\nInference scaling factors (4 features):")
    print(f"  Mean: {scaling_inference[0]}")
    print(f"  Std:  {scaling_inference[1]}")
    
    # Also create target scaling (for unnormalizing predictions)
    if fine_mean is not None:
        scaling_target = np.zeros((2, 4), dtype=np.float32)
        scaling_target[0] = fine_mean
        scaling_target[1] = fine_std
        
        print(f"\nTarget scaling factors (4 features):")
        print(f"  Mean: {scaling_target[0]}")
        print(f"  Std:  {scaling_target[1]}")
    else:
        scaling_target = scaling_inference  # Use coarse as fallback
    
    return {
        'enhanced': scaling_enhanced,
        'inference': scaling_inference,
        'target': scaling_target,
        'coarse_mean': coarse_mean,
        'coarse_std': coarse_std,
        'fine_mean': fine_mean,
        'fine_std': fine_std
    }

def save_scaling_factors(scaling_data):
    """
    Save scaling factors to appropriate locations
    """
    print("\n" + "=" * 80)
    print("SAVING SCALING FACTORS")
    print("=" * 80)
    
    # Create directories
    output_base = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced scaling (8 features) for training
    if scaling_data['enhanced'] is not None:
        enhanced_path = output_base / "scaling_factors_enhanced.pt"
        torch.save(torch.tensor(scaling_data['enhanced']), enhanced_path)
        print(f"✓ Saved enhanced scaling (8 features): {enhanced_path}")
        
        # Also save as numpy
        np_path = output_base / "scaling_factors_enhanced.npy"
        np.save(np_path, scaling_data['enhanced'])
        print(f"✓ Saved as numpy: {np_path}")
    
    # Save inference scaling (4 features)
    inference_path = output_base / "scaling_factors_inference.pt"
    torch.save(torch.tensor(scaling_data['inference']), inference_path)
    print(f"✓ Saved inference scaling (4 features): {inference_path}")
    
    # Save target scaling for predictions
    target_path = output_base / "scaling_factors_target.pt"
    torch.save(torch.tensor(scaling_data['target']), target_path)
    print(f"✓ Saved target scaling (4 features): {target_path}")
    
    # Save in format expected by test script
    surface_path = output_base / "surface_scaling_factors_inference.npy"
    np.save(surface_path, scaling_data['target'])  # Use target (fine) for unnormalizing
    print(f"✓ Saved surface scaling: {surface_path}")
    
    # Create a reference file with all info
    info_path = output_base / "scaling_info.txt"
    with open(info_path, 'w') as f:
        f.write("SCALING FACTORS INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Coarse Input Statistics:\n")
        f.write(f"  Mean: {scaling_data['coarse_mean']}\n")
        f.write(f"  Std:  {scaling_data['coarse_std']}\n\n")
        
        if scaling_data['fine_mean'] is not None:
            f.write("Fine Target Statistics:\n")
            f.write(f"  Mean: {scaling_data['fine_mean']}\n")
            f.write(f"  Std:  {scaling_data['fine_std']}\n\n")
        
        f.write("File Usage:\n")
        f.write("  scaling_factors_enhanced.pt - For training (8 features)\n")
        f.write("  scaling_factors_inference.pt - For normalizing input (4 features)\n")
        f.write("  scaling_factors_target.pt - For unnormalizing output (4 features)\n")
        f.write("  surface_scaling_factors_inference.npy - Used by test script\n")
    
    print(f"✓ Saved info file: {info_path}")

def verify_scaling_with_test_data():
    """
    Verify scaling factors work correctly with test data
    """
    print("\n" + "=" * 80)
    print("VERIFYING SCALING FACTORS WITH TEST DATA")
    print("=" * 80)
    
    # Load a test case
    test_file = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data/case_451.h5")
    
    if not test_file.exists():
        print("Test file not found")
        return
    
    # Load scaling factors
    scaling_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/scaling_factors_inference.pt")
    target_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/scaling_factors_target.pt")
    
    if not scaling_path.exists():
        print("Scaling factors not found")
        return
    
    scaling_input = torch.load(scaling_path).numpy()
    scaling_target = torch.load(target_path).numpy()
    
    with h5py.File(test_file, 'r') as f:
        if 'coarse_input' in f:
            coarse = f['coarse_input'][:]
            
            print(f"\nTest case 451:")
            print(f"  Raw coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
            print(f"  Raw coarse range: [{coarse.min():.4f}, {coarse.max():.4f}]")
            
            # Normalize input
            coarse_norm = (coarse - scaling_input[0]) / scaling_input[1]
            print(f"\n  Normalized coarse pressure: mean={coarse_norm[:, 0].mean():.4f}, std={coarse_norm[:, 0].std():.4f}")
            print(f"  Normalized range: [{coarse_norm.min():.4f}, {coarse_norm.max():.4f}]")
            
            if np.abs(coarse_norm).max() > 10:
                print("  ⚠️ WARNING: Normalized values exceed ±10 std!")
                print("     This might cause prediction issues")
            else:
                print("  ✓ Normalization looks reasonable")
            
            # Show what unnormalized predictions would look like
            print(f"\n  Target scaling (for predictions):")
            print(f"    Mean: {scaling_target[0]}")
            print(f"    Std:  {scaling_target[1]}")

def main():
    """
    Main function to create and verify scaling factors
    """
    print("=" * 80)
    print("ENHANCED DoMINO SCALING FACTOR CREATION")
    print("=" * 80)
    
    # Check if scaling factors already exist
    existing_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/scaling_factors_enhanced.pt")
    
    if existing_path.exists():
        print(f"\n⚠️ Scaling factors already exist at: {existing_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Compute scaling factors
    scaling_data = compute_scaling_factors_from_data()
    
    if scaling_data is None:
        print("ERROR: Could not compute scaling factors")
        return
    
    # Save scaling factors
    save_scaling_factors(scaling_data)
    
    # Verify with test data
    verify_scaling_with_test_data()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Re-run test_enhanced.py with the new scaling factors")
    print("2. Check if predictions are now in the correct range")
    print("3. If still inverted, the model weights may need adjustment")

if __name__ == "__main__":
    main()
