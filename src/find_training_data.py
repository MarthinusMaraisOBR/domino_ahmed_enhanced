#!/usr/bin/env python3
"""
Find where the actual training data is stored and compute scaling factors
"""

import numpy as np
import torch
import h5py
from pathlib import Path
import os
import pyvista as pv

def find_training_data():
    """
    Search for training data in various formats and locations
    """
    print("=" * 80)
    print("SEARCHING FOR TRAINING DATA")
    print("=" * 80)
    
    # Possible data locations
    search_paths = [
        "/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data",
        "/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_vtp",
        "/data/ahmed_data",
        "/data/ahmed_data/organized/train",
        "/data/ahmed_data/organized/train/coarse",
        "/data/ahmed_data/organized/train/fine",
        "./dataset",
        "../dataset"
    ]
    
    found_data = {}
    
    for base_path in search_paths:
        base = Path(base_path)
        if base.exists():
            print(f"\n📁 Checking: {base_path}")
            
            # Look for H5 files
            h5_files = list(base.glob("*.h5")) + list(base.glob("*/*.h5"))
            if h5_files:
                print(f"  Found {len(h5_files)} H5 files")
                found_data['h5'] = h5_files[:5]  # Sample first 5
                
                # Check what's in them
                with h5py.File(h5_files[0], 'r') as f:
                    print(f"  H5 keys: {list(f.keys())}")
                    for key in f.keys():
                        print(f"    {key}: shape {f[key].shape}")
            
            # Look for VTP files
            vtp_files = list(base.glob("*.vtp")) + list(base.glob("*/*.vtp"))
            if vtp_files:
                print(f"  Found {len(vtp_files)} VTP files")
                found_data['vtp'] = vtp_files[:5]
            
            # Look for organized directories
            if (base / "run_1").exists():
                print(f"  Found organized run directories")
                found_data['runs'] = base
    
    return found_data

def analyze_vtp_data_for_scaling():
    """
    Compute scaling factors from VTP files
    """
    print("\n" + "=" * 80)
    print("COMPUTING SCALING FROM VTP FILES")
    print("=" * 80)
    
    # Paths to coarse and fine training data
    coarse_path = Path("/data/ahmed_data/organized/train/coarse")
    fine_path = Path("/data/ahmed_data/organized/train/fine")
    
    if not coarse_path.exists():
        print(f"ERROR: Coarse path doesn't exist: {coarse_path}")
        return None
    
    # Collect statistics from VTP files
    coarse_pressure_all = []
    coarse_wss_all = []
    fine_pressure_all = []
    fine_wss_all = []
    
    print("\nAnalyzing VTP training data...")
    
    # Sample every 10th case for efficiency
    for i in range(1, 401, 10):
        run_dir = f"run_{i}"
        
        # Load coarse VTP
        coarse_vtp = coarse_path / run_dir / f"boundary_{i}.vtp"
        if coarse_vtp.exists():
            mesh = pv.read(str(coarse_vtp))
            
            # Extract pressure (try different names)
            for p_name in ['p', 'pressure', 'Pressure', 'pMean']:
                if p_name in mesh.cell_data:
                    pressure = mesh.cell_data[p_name]
                    coarse_pressure_all.append(pressure)
                    break
                elif p_name in mesh.point_data:
                    pressure = mesh.point_data[p_name]
                    # Convert to cell data
                    mesh_cells = mesh.point_data_to_cell_data()
                    if p_name in mesh_cells.cell_data:
                        pressure = mesh_cells.cell_data[p_name]
                        coarse_pressure_all.append(pressure)
                        break
            
            # Extract wall shear stress
            for w_name in ['wallShearStress', 'WallShearStress', 'wallShearStressMean']:
                if w_name in mesh.cell_data:
                    wss = mesh.cell_data[w_name]
                    coarse_wss_all.append(wss)
                    break
                elif w_name in mesh.point_data:
                    wss = mesh.point_data[w_name]
                    mesh_cells = mesh.point_data_to_cell_data()
                    if w_name in mesh_cells.cell_data:
                        wss = mesh_cells.cell_data[w_name]
                        coarse_wss_all.append(wss)
                        break
            
            if i % 100 == 1:
                print(f"  Processed coarse case {i}")
        
        # Load fine VTP if it exists
        fine_vtp = fine_path / run_dir / f"boundary_{i}.vtp"
        if fine_vtp.exists():
            mesh = pv.read(str(fine_vtp))
            
            # Extract pressure
            for p_name in ['pMean', 'p', 'pressure', 'Pressure']:
                if p_name in mesh.cell_data:
                    pressure = mesh.cell_data[p_name]
                    fine_pressure_all.append(pressure)
                    break
                elif p_name in mesh.point_data:
                    pressure = mesh.point_data[p_name]
                    mesh_cells = mesh.point_data_to_cell_data()
                    if p_name in mesh_cells.cell_data:
                        pressure = mesh_cells.cell_data[p_name]
                        fine_pressure_all.append(pressure)
                        break
            
            # Extract wall shear stress
            for w_name in ['wallShearStressMean', 'wallShearStress', 'WallShearStress']:
                if w_name in mesh.cell_data:
                    wss = mesh.cell_data[w_name]
                    fine_wss_all.append(wss)
                    break
                elif w_name in mesh.point_data:
                    wss = mesh.point_data[w_name]
                    mesh_cells = mesh.point_data_to_cell_data()
                    if w_name in mesh_cells.cell_data:
                        wss = mesh_cells.cell_data[w_name]
                        fine_wss_all.append(wss)
                        break
    
    if not coarse_pressure_all:
        print("ERROR: No coarse pressure data found")
        return None
    
    # Compute statistics
    coarse_pressure = np.concatenate(coarse_pressure_all)
    coarse_wss = np.concatenate(coarse_wss_all, axis=0)
    
    print(f"\nCoarse data statistics:")
    print(f"  Pressure: mean={coarse_pressure.mean():.6f}, std={coarse_pressure.std():.6f}")
    print(f"  Pressure range: [{coarse_pressure.min():.4f}, {coarse_pressure.max():.4f}]")
    print(f"  WSS magnitude: mean={np.linalg.norm(coarse_wss, axis=1).mean():.6f}")
    
    # Create coarse statistics (4 features: pressure + 3 WSS components)
    coarse_mean = np.zeros(4, dtype=np.float32)
    coarse_std = np.zeros(4, dtype=np.float32)
    
    coarse_mean[0] = coarse_pressure.mean()
    coarse_std[0] = coarse_pressure.std()
    
    for i in range(3):
        coarse_mean[i+1] = coarse_wss[:, i].mean()
        coarse_std[i+1] = coarse_wss[:, i].std()
    
    print(f"\nCoarse scaling (4 features):")
    print(f"  Mean: {coarse_mean}")
    print(f"  Std:  {coarse_std}")
    
    # Process fine data if available
    if fine_pressure_all:
        fine_pressure = np.concatenate(fine_pressure_all)
        fine_wss = np.concatenate(fine_wss_all, axis=0)
        
        print(f"\nFine data statistics:")
        print(f"  Pressure: mean={fine_pressure.mean():.6f}, std={fine_pressure.std():.6f}")
        print(f"  Pressure range: [{fine_pressure.min():.4f}, {fine_pressure.max():.4f}]")
        print(f"  WSS magnitude: mean={np.linalg.norm(fine_wss, axis=1).mean():.6f}")
        
        fine_mean = np.zeros(4, dtype=np.float32)
        fine_std = np.zeros(4, dtype=np.float32)
        
        fine_mean[0] = fine_pressure.mean()
        fine_std[0] = fine_pressure.std()
        
        for i in range(3):
            fine_mean[i+1] = fine_wss[:, i].mean()
            fine_std[i+1] = fine_wss[:, i].std()
        
        print(f"\nFine scaling (4 features):")
        print(f"  Mean: {fine_mean}")
        print(f"  Std:  {fine_std}")
    else:
        fine_mean = coarse_mean  # Use coarse as fallback
        fine_std = coarse_std
    
    return {
        'coarse_mean': coarse_mean,
        'coarse_std': coarse_std,
        'fine_mean': fine_mean,
        'fine_std': fine_std
    }

def save_computed_scaling(stats):
    """
    Save the computed scaling factors
    """
    print("\n" + "=" * 80)
    print("SAVING SCALING FACTORS")
    print("=" * 80)
    
    output_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scaling tensors
    # For enhanced training (8 features: fine + coarse)
    scaling_enhanced = np.zeros((2, 8), dtype=np.float32)
    scaling_enhanced[0, :4] = stats['fine_mean']
    scaling_enhanced[0, 4:] = stats['coarse_mean']
    scaling_enhanced[1, :4] = stats['fine_std']
    scaling_enhanced[1, 4:] = stats['coarse_std']
    
    # Save enhanced (8 features)
    torch.save(torch.tensor(scaling_enhanced), output_dir / "scaling_factors_enhanced.pt")
    np.save(output_dir / "scaling_factors_enhanced.npy", scaling_enhanced)
    print(f"✓ Saved enhanced scaling (8 features)")
    
    # For inference input (4 features: coarse only)
    scaling_input = np.zeros((2, 4), dtype=np.float32)
    scaling_input[0] = stats['coarse_mean']
    scaling_input[1] = stats['coarse_std']
    
    torch.save(torch.tensor(scaling_input), output_dir / "scaling_factors_inference.pt")
    print(f"✓ Saved input scaling (4 features)")
    
    # For output unnormalization (4 features: fine)
    scaling_output = np.zeros((2, 4), dtype=np.float32)
    scaling_output[0] = stats['fine_mean']
    scaling_output[1] = stats['fine_std']
    
    torch.save(torch.tensor(scaling_output), output_dir / "scaling_factors_target.pt")
    np.save(output_dir / "surface_scaling_factors_inference.npy", scaling_output)
    print(f"✓ Saved output scaling (4 features)")
    
    # Save readable info
    info_path = output_dir / "scaling_info.txt"
    with open(info_path, 'w') as f:
        f.write("SCALING FACTORS FROM VTP DATA\n")
        f.write("=" * 60 + "\n\n")
        f.write("Coarse (Input) Statistics:\n")
        f.write(f"  Mean: {stats['coarse_mean']}\n")
        f.write(f"  Std:  {stats['coarse_std']}\n\n")
        f.write("Fine (Target) Statistics:\n")
        f.write(f"  Mean: {stats['fine_mean']}\n")
        f.write(f"  Std:  {stats['fine_std']}\n\n")
        f.write("Usage:\n")
        f.write("  - Input normalization: (x - coarse_mean) / coarse_std\n")
        f.write("  - Output unnormalization: y * fine_std + fine_mean\n")
    
    print(f"✓ Saved info file: {info_path}")

def main():
    # First, find where data is stored
    found_data = find_training_data()
    
    # Try to compute from VTP files
    stats = analyze_vtp_data_for_scaling()
    
    if stats is None:
        print("\nERROR: Could not compute scaling factors from data")
        print("\nPossible solutions:")
        print("1. Check if training data exists in /data/ahmed_data/organized/train/")
        print("2. Verify VTP files contain pressure and wallShearStress fields")
        print("3. Use scaling factors from original training if available")
        return
    
    # Save the scaling factors
    save_computed_scaling(stats)
    
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nScaling factors created successfully!")
    print("Now re-run: python test_enhanced.py")

if __name__ == "__main__":
    main()
