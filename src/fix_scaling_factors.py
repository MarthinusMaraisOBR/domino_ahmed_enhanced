#!/usr/bin/env python3
"""
fix_scaling_factors.py - Compute proper scaling factors for 8-feature enhanced data
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def compute_enhanced_scaling_factors():
    """Compute scaling factors for enhanced 8-feature surface fields"""
    
    print("="*80)
    print("COMPUTING ENHANCED SCALING FACTORS (8 FEATURES)")
    print("="*80)
    
    # Input and output paths
    data_dir = "/data/ahmed_data/processed/train/"
    output_base = "outputs/Ahmed_Dataset/"
    
    # Create output directories
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(os.path.join(output_base, "enhanced_fixed_v2"), exist_ok=True)
    
    print(f"\nProcessing data from: {data_dir}")
    
    # Get all NPY files
    npy_files = list(Path(data_dir).glob("*.npy"))
    print(f"Found {len(npy_files)} NPY files")
    
    if not npy_files:
        print("❌ No NPY files found! Run process_data.py first.")
        return False
    
    # Initialize min/max trackers for 8 features
    global_min = np.full(8, np.inf)
    global_max = np.full(8, -np.inf)
    
    # Track statistics
    feature_stats = {i: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for i in range(8)}
    
    # Process each file
    print("\nProcessing files...")
    for npy_file in tqdm(npy_files[:50], desc="Computing statistics"):  # Sample first 50 files
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            
            if 'surface_fields' in data:
                surface_fields = data['surface_fields']
                
                # Check shape
                if surface_fields.shape[1] == 8:
                    # Update min/max for each feature
                    for i in range(8):
                        field_data = surface_fields[:, i]
                        global_min[i] = min(global_min[i], field_data.min())
                        global_max[i] = max(global_max[i], field_data.max())
                        
                        # Track statistics
                        feature_stats[i]['sum'] += field_data.sum()
                        feature_stats[i]['sum_sq'] += (field_data**2).sum()
                        feature_stats[i]['count'] += len(field_data)
                        
                elif surface_fields.shape[1] == 4:
                    print(f"⚠️ Warning: {npy_file.name} has only 4 features - not enhanced data!")
                else:
                    print(f"⚠️ Warning: {npy_file.name} has {surface_fields.shape[1]} features")
                    
        except Exception as e:
            print(f"⚠️ Error processing {npy_file.name}: {e}")
            continue
    
    # Check if we found 8-feature data
    if np.any(np.isinf(global_min[:4])) or np.any(np.isinf(global_max[:4])):
        print("\n❌ ERROR: No valid 8-feature enhanced data found!")
        print("Make sure you've run process_data.py with enhanced features enabled.")
        return False
    
    # Create scaling factors array [2, 8]
    # Row 0: max values, Row 1: min values
    scaling_factors = np.array([global_max, global_min], dtype=np.float32)
    
    print("\n" + "="*60)
    print("ENHANCED SCALING FACTORS COMPUTED")
    print("="*60)
    print("\nFeature Statistics:")
    print(f"{'Feature':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 68)
    
    feature_names = [
        "Fine Pressure", "Fine Shear X", "Fine Shear Y", "Fine Shear Z",
        "Coarse Pressure", "Coarse Shear X", "Coarse Shear Y", "Coarse Shear Z"
    ]
    
    for i in range(8):
        if feature_stats[i]['count'] > 0:
            mean = feature_stats[i]['sum'] / feature_stats[i]['count']
            var = (feature_stats[i]['sum_sq'] / feature_stats[i]['count']) - mean**2
            std = np.sqrt(max(0, var))
        else:
            mean = std = 0.0
            
        print(f"{feature_names[i]:<20} {global_min[i]:>12.6f} {global_max[i]:>12.6f} {mean:>12.6f} {std:>12.6f}")
    
    # Validate the scaling factors
    print("\n📊 Validation:")
    
    # Check if fine and coarse have similar ranges
    for i in range(4):
        fine_range = global_max[i] - global_min[i]
        coarse_range = global_max[i+4] - global_min[i+4]
        ratio = coarse_range / fine_range if fine_range > 0 else 0
        
        print(f"  Feature {i}: Fine range={fine_range:.6f}, Coarse range={coarse_range:.6f}, Ratio={ratio:.2f}")
        
        if ratio < 0.5 or ratio > 2.0:
            print(f"    ⚠️ Large range difference between fine and coarse!")
    
    # Save scaling factors in multiple locations
    save_paths = [
        os.path.join(output_base, "surface_scaling_factors.npy"),
        os.path.join(output_base, "enhanced_fixed_v2/surface_scaling_factors.npy"),
        os.path.join(output_base, "surface_scaling_factors_enhanced.npy"),
    ]
    
    print("\n💾 Saving scaling factors:")
    for save_path in save_paths:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, scaling_factors)
        print(f"  ✅ Saved to: {save_path}")
    
    # Also save a human-readable version
    info_path = os.path.join(output_base, "enhanced_fixed_v2/scaling_info.txt")
    with open(info_path, 'w') as f:
        f.write("Enhanced Scaling Factors (8 features)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Shape: {scaling_factors.shape}\n\n")
        f.write(f"{'Feature':<20} {'Min':>12} {'Max':>12}\n")
        f.write("-" * 44 + "\n")
        for i in range(8):
            f.write(f"{feature_names[i]:<20} {global_min[i]:>12.6f} {global_max[i]:>12.6f}\n")
    print(f"  ✅ Info saved to: {info_path}")
    
    print("\n✅ SUCCESS: Enhanced scaling factors computed and saved!")
    print("You can now run training with these proper 8-feature scaling factors.")
    
    return True


def verify_data_format():
    """Quick verification that data has 8 features"""
    
    print("\n" + "="*60)
    print("VERIFYING DATA FORMAT")
    print("="*60)
    
    # Check a sample file
    sample_file = Path("/data/ahmed_data/processed/train/run_1.npy")
    
    if not sample_file.exists():
        print(f"❌ Sample file not found: {sample_file}")
        return False
    
    data = np.load(sample_file, allow_pickle=True).item()
    
    print(f"\nSample file: {sample_file.name}")
    print(f"Keys in data: {list(data.keys())}")
    
    if 'surface_fields' in data:
        surface_fields = data['surface_fields']
        print(f"Surface fields shape: {surface_fields.shape}")
        
        if surface_fields.shape[1] == 8:
            print("✅ Data has 8 features - enhanced format confirmed!")
            
            # Show sample statistics
            print("\nSample statistics (first 100 points):")
            sample = surface_fields[:100]
            
            print(f"{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print("-" * 55)
            for i in range(8):
                feature_type = "Fine" if i < 4 else "Coarse"
                feature_name = ["Pressure", "Shear X", "Shear Y", "Shear Z"][i % 4]
                label = f"{feature_type} {feature_name}"
                
                print(f"{label:<15} {sample[:, i].mean():>10.6f} {sample[:, i].std():>10.6f} "
                      f"{sample[:, i].min():>10.6f} {sample[:, i].max():>10.6f}")
                      
            return True
        else:
            print(f"❌ Data has {surface_fields.shape[1]} features - not enhanced format!")
            print("You need to re-run process_data.py with enhanced features enabled.")
            return False
    else:
        print("❌ No surface_fields in data!")
        return False


if __name__ == "__main__":
    print("🔧 ENHANCED SCALING FACTORS FIX")
    print("="*80)
    print("This will compute proper 8-feature scaling factors for enhanced training.\n")
    
    # First verify data format
    if not verify_data_format():
        print("\n❌ Data verification failed!")
        print("Please ensure you've run process_data.py with:")
        print("  use_enhanced_features: true")
        exit(1)
    
    # Compute scaling factors
    success = compute_enhanced_scaling_factors()
    
    if success:
        print("\n" + "="*80)
        print("✅ SCALING FACTORS FIXED!")
        print("="*80)
        print("\nYou can now run training with:")
        print("  python train_fixed.py --config-name=config_fixed")
        print("\nOr use the original training script:")
        print("  python train.py")
    else:
        print("\n❌ Failed to compute scaling factors!")
        print("Check the error messages above and fix any issues.")
        exit(1)
