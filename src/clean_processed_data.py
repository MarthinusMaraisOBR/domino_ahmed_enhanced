# clean_processed_data.py
#!/usr/bin/env python3
"""
Safely clean previously processed data and scaling factors
"""

import os
import shutil
from pathlib import Path

def clean_processed_data():
    """Clean old processed data and scaling factors"""
    
    print("="*60)
    print("CLEANING PREVIOUS PROCESSED DATA")
    print("="*60)
    
    # Directories to clean
    dirs_to_clean = [
        "/data/ahmed_data/processed/train/",
        "/data/ahmed_data/processed/val/",
        "/data/ahmed_data/processed/test/",
        "/data/ahmed_data/cached/",
    ]
    
    # Scaling factor files to remove
    scaling_files = [
        "outputs/Ahmed_Dataset/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/volume_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_enhanced.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy",
    ]
    
    # Clean directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            print(f"\nCleaning: {dir_path}")
            
            # Count files before cleaning
            npy_files = list(Path(dir_path).glob("*.npy"))
            print(f"  Found {len(npy_files)} NPY files")
            
            # Ask for confirmation
            response = input(f"  Delete all NPY files in {dir_path}? (y/n): ")
            if response.lower() == 'y':
                for npy_file in npy_files:
                    os.remove(npy_file)
                print(f"  ✅ Deleted {len(npy_files)} files")
            else:
                print(f"  ⏭️  Skipped")
    
    # Clean scaling factors
    print("\nCleaning scaling factors:")
    for file_path in scaling_files:
        if os.path.exists(file_path):
            print(f"  Removing: {file_path}")
            os.remove(file_path)
            print(f"  ✅ Deleted")
    
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    clean_processed_data()
