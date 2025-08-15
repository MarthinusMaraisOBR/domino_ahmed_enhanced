#!/usr/bin/env python3
"""
Reprocess the Ahmed dataset with proper train/val/test split
"""

import os
import shutil
from pathlib import Path
from openfoam_datapipe import OpenFoamDatasetEnhanced
import numpy as np

def process_split_data():
    # Define splits
    train_cases = list(range(1, 401))
    val_cases = list(range(401, 451))
    test_cases = list(range(451, 501))
    
    # Paths
    fine_raw = Path("/data/ahmed_data/ahmed_data/raw/")
    coarse_raw = Path("/data/ahmed_data/ahmed_data_rans/raw/")
    
    # Output paths
    output_base = Path("/data/ahmed_data/ahmed_data_fixed/")
    train_out = output_base / "train"
    val_out = output_base / "validation"
    test_out = output_base / "test"
    
    # Create directories
    for dir in [train_out, val_out, test_out]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Variable names
    surface_variables = ["pMean", "wallShearStressMean"]
    coarse_variable_mapping = {
        "pMean": "p",
        "wallShearStressMean": "wallShearStress"
    }
    
    # Create dataset
    dataset = OpenFoamDatasetEnhanced(
        data_path=fine_raw,
        coarse_data_path=coarse_raw,
        kind="ahmed",
        surface_variables=surface_variables,
        model_type="surface",
        coarse_variable_mapping=coarse_variable_mapping,
    )
    
    # Process each case
    for idx, case_name in enumerate(dataset.filenames):
        case_num = int(case_name.split("_")[1])
        
        # Get data
        data_dict = dataset[idx]
        
        # Determine output directory
        if case_num in train_cases:
            output_dir = train_out
            print(f"Processing training case: {case_name}")
        elif case_num in val_cases:
            output_dir = val_out
            print(f"Processing validation case: {case_name}")
        elif case_num in test_cases:
            # Skip test cases for training
            print(f"Skipping test case: {case_name}")
            continue
        else:
            print(f"Unknown case: {case_name}")
            continue
        
        # Save
        output_file = output_dir / f"{case_name}.npy"
        np.save(output_file, data_dict)
        print(f"  Saved to: {output_file}")
    
    print("\nProcessing complete!")
    print(f"Training data: {train_out}")
    print(f"Validation data: {val_out}")
    print(f"Test cases: Not processed (keep separate)")

if __name__ == "__main__":
    process_split_data()
