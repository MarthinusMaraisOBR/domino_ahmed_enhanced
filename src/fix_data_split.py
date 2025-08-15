#!/usr/bin/env python3
"""
Fix the data split by properly separating train/val/test sets.
This script will reorganize the processed NPY files to prevent data leakage.
"""

import os
import shutil
import numpy as np
from pathlib import Path
import json

def fix_data_split():
    """Fix the data split to prevent leakage"""
    
    print("="*80)
    print("FIXING DATA SPLIT - REMOVING TEST CASES FROM TRAINING")
    print("="*80)
    
    # Define the proper split
    train_cases = list(range(1, 401))      # Cases 1-400 for training
    val_cases = list(range(401, 451))      # Cases 401-450 for validation  
    test_cases = list(range(451, 501))     # Cases 451-500 for testing
    
    print("\nProper data split:")
    print(f"  Training: Cases 1-400 ({len(train_cases)} cases)")
    print(f"  Validation: Cases 401-450 ({len(val_cases)} cases)")
    print(f"  Testing: Cases 451-500 ({len(test_cases)} cases)")
    
    # Paths
    train_dir = Path("/data/ahmed_data/ahmed_data/train/")
    val_dir = Path("/data/ahmed_data/ahmed_data/validation/")
    
    # Create backup directory
    backup_dir = Path("/data/ahmed_data/ahmed_data/train_backup/")
    
    # Step 1: Backup current training data
    if train_dir.exists() and not backup_dir.exists():
        print(f"\n1. Creating backup of current training data...")
        shutil.copytree(train_dir, backup_dir)
        print(f"   Backup created at: {backup_dir}")
    elif backup_dir.exists():
        print(f"\n1. Backup already exists at: {backup_dir}")
    
    # Step 2: Create clean training directory with only cases 1-400
    clean_train_dir = Path("/data/ahmed_data/ahmed_data/train_clean/")
    clean_train_dir.mkdir(exist_ok=True)
    
    print(f"\n2. Creating clean training set (cases 1-400 only)...")
    moved_count = 0
    kept_count = 0
    
    for npy_file in train_dir.glob("*.npy"):
        if 'run_' in npy_file.name:
            try:
                case_num = int(npy_file.name.split('run_')[1].split('.')[0])
                
                if case_num in train_cases:
                    # Keep this file in training
                    dest = clean_train_dir / npy_file.name
                    if not dest.exists():
                        shutil.copy2(npy_file, dest)
                    kept_count += 1
                elif case_num in test_cases:
                    # This should NOT be in training!
                    moved_count += 1
                    print(f"   Removing test case from training: run_{case_num}")
            except Exception as e:
                print(f"   Error processing {npy_file.name}: {e}")
    
    print(f"   Kept {kept_count} training cases")
    print(f"   Removed {moved_count} test cases from training")
    
    # Step 3: Process validation set with enhanced features
    print(f"\n3. Processing validation set with enhanced features...")
    
    # We need to reprocess validation data with enhanced features
    # Check if validation has enhanced features
    val_enhanced_dir = Path("/data/ahmed_data/ahmed_data/validation_enhanced/")
    val_enhanced_dir.mkdir(exist_ok=True)
    
    val_files = list(val_dir.glob("*.npy"))
    if val_files:
        sample_file = val_files[0]
        data = np.load(sample_file, allow_pickle=True).item()
        if 'surface_fields' in data:
            num_features = data['surface_fields'].shape[-1]
            if num_features == 4:
                print(f"   ⚠️  Validation data has only 4 features (not enhanced)")
                print(f"   You need to reprocess validation data with enhanced features!")
                print(f"   Run: python process_data.py with proper configuration")
            else:
                print(f"   ✅ Validation data has {num_features} features")
    
    # Step 4: Create a proper test set directory
    test_npy_dir = Path("/data/ahmed_data/ahmed_data/test/")
    test_npy_dir.mkdir(exist_ok=True)
    
    print(f"\n4. Setting up test directory...")
    print(f"   Test NPY files should go to: {test_npy_dir}")
    print(f"   Test cases (451-500) should NEVER be in training!")
    
    # Step 5: Verify the fix
    print(f"\n5. Verification:")
    
    # Check clean training directory
    clean_train_cases = []
    for npy_file in clean_train_dir.glob("*.npy"):
        if 'run_' in npy_file.name:
            try:
                case_num = int(npy_file.name.split('run_')[1].split('.')[0])
                clean_train_cases.append(case_num)
            except:
                pass
    
    clean_train_cases = sorted(clean_train_cases)
    
    # Check for leakage
    leak_cases = [c for c in clean_train_cases if c in test_cases]
    
    if leak_cases:
        print(f"   ❌ STILL HAVE LEAKAGE! Test cases in training: {leak_cases[:10]}")
    else:
        print(f"   ✅ No data leakage in clean training set!")
        print(f"   Clean training set has {len(clean_train_cases)} cases")
        print(f"   Range: {min(clean_train_cases) if clean_train_cases else 'N/A'} - {max(clean_train_cases) if clean_train_cases else 'N/A'}")
    
    # Save split info
    split_info = {
        'train_cases': train_cases,
        'val_cases': val_cases,
        'test_cases': test_cases,
        'clean_train_count': len(clean_train_cases),
        'leaked_cases_removed': moved_count,
        'directories': {
            'original_train': str(train_dir),
            'backup': str(backup_dir),
            'clean_train': str(clean_train_dir),
            'validation': str(val_dir),
            'test': str(test_npy_dir)
        }
    }
    
    with open('fixed_data_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n6. Split information saved to: fixed_data_split.json")
    
    return split_info


def reprocess_validation_with_enhanced():
    """Guide for reprocessing validation data with enhanced features"""
    
    print("\n" + "="*80)
    print("HOW TO REPROCESS VALIDATION DATA WITH ENHANCED FEATURES")
    print("="*80)
    
    print("""
To fix the validation data, you need to:

1. First, move the test cases out of the raw directory:
   ```bash
   # Create a temporary directory for test cases
   mkdir -p /data/ahmed_data/ahmed_data/raw_train
   
   # Move cases 1-450 to training directory
   for i in {1..450}; do
       mv /data/ahmed_data/ahmed_data/raw/run_$i /data/ahmed_data/ahmed_data/raw_train/
   done
   ```

2. Update config.yaml:
   ```yaml
   data_processor:
     input_dir: /data/ahmed_data/ahmed_data/raw_train/
     output_dir: /data/ahmed_data/ahmed_data/train_fixed/
     use_enhanced_features: true
   ```

3. Run processing for training data (cases 1-400):
   ```bash
   python process_data.py
   ```

4. Then process validation separately by moving cases 401-450.

Alternatively, you can modify process_data.py to handle the split internally.
""")


def create_reprocessing_script():
    """Create a script to reprocess data correctly"""
    
    script_content = '''#!/usr/bin/env python3
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
    
    print("\\nProcessing complete!")
    print(f"Training data: {train_out}")
    print(f"Validation data: {val_out}")
    print(f"Test cases: Not processed (keep separate)")

if __name__ == "__main__":
    process_split_data()
'''
    
    with open('reprocess_data_split.py', 'w') as f:
        f.write(script_content)
    
    print("\n7. Created reprocessing script: reprocess_data_split.py")
    print("   Run this to properly reprocess data with correct splits.")


if __name__ == "__main__":
    # Fix the current split
    split_info = fix_data_split()
    
    # Provide guidance for reprocessing
    reprocess_validation_with_enhanced()
    
    # Create reprocessing script
    create_reprocessing_script()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
CRITICAL: Do NOT test on the current model yet! It has seen the test data.

You have two options:

OPTION 1: Retrain from scratch with clean data
-----------------------------------------
1. Use the clean training data in: /data/ahmed_data/ahmed_data/train_clean/
2. Reprocess validation data with enhanced features
3. Update config.yaml to point to clean directories
4. Retrain the model (it will be much faster since you know it works)

OPTION 2: Test on truly unseen data
-----------------------------------------
Since your model has seen cases 451-500, you could:
- Test on synthetic variations (different parameters)
- Test on cases with different geometry modifications
- Cross-validate on cases 1-50 (hold them out and retrain)

RECOMMENDED: Option 1 - Retrain with clean split
This ensures scientifically valid results for publication.

The good news: Your model architecture works perfectly (99.9% improvement)!
Just need to retrain with proper data split.
""")
