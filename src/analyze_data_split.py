#!/usr/bin/env python3
"""
Analyze which Ahmed body cases were used for training, validation, and testing.
This script examines the data directories and processed NPY files to determine
the exact split of cases.
"""

import os
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def analyze_data_directories():
    """Analyze the data directory structure and case distribution"""
    
    print("="*80)
    print("AHMED BODY DATASET ANALYSIS - TRAIN/VAL/TEST SPLIT")
    print("="*80)
    
    results = {
        'raw_data': {},
        'processed_data': {},
        'summary': {}
    }
    
    # 1. Check raw data directories
    print("\n1. RAW DATA DIRECTORIES:")
    print("-" * 40)
    
    # Fine resolution data paths
    fine_paths = {
        'raw': '/data/ahmed_data/ahmed_data/raw/',
        'raw_test': '/data/ahmed_data/ahmed_data/raw_test/',
        'raw_val': '/data/ahmed_data/ahmed_data/raw_val/'  # May not exist
    }
    
    # Coarse resolution data paths
    coarse_paths = {
        'raw': '/data/ahmed_data/ahmed_data_rans/raw/',
        'raw_test': '/data/ahmed_data/ahmed_data_rans/raw_test/',
        'raw_val': '/data/ahmed_data/ahmed_data_rans/raw_val/'  # May not exist
    }
    
    # Analyze fine data
    print("\nFine Resolution Data:")
    for name, path in fine_paths.items():
        if os.path.exists(path):
            cases = sorted([d for d in os.listdir(path) if d.startswith('run_')])
            case_numbers = [int(c.split('_')[1]) for c in cases]
            results['raw_data'][f'fine_{name}'] = case_numbers
            
            print(f"  {name}: {path}")
            print(f"    Cases: {len(cases)}")
            if case_numbers:
                print(f"    Range: {min(case_numbers)} - {max(case_numbers)}")
                print(f"    First 5: {case_numbers[:5]}")
                print(f"    Last 5: {case_numbers[-5:]}")
        else:
            print(f"  {name}: NOT FOUND at {path}")
    
    # Analyze coarse data
    print("\nCoarse Resolution Data:")
    for name, path in coarse_paths.items():
        if os.path.exists(path):
            cases = sorted([d for d in os.listdir(path) if d.startswith('run_')])
            case_numbers = [int(c.split('_')[1]) for c in cases]
            results['raw_data'][f'coarse_{name}'] = case_numbers
            
            print(f"  {name}: {path}")
            print(f"    Cases: {len(cases)}")
            if case_numbers:
                print(f"    Range: {min(case_numbers)} - {max(case_numbers)}")
                print(f"    First 5: {case_numbers[:5]}")
                print(f"    Last 5: {case_numbers[-5:]}")
        else:
            print(f"  {name}: NOT FOUND at {path}")
    
    # 2. Check processed NPY files
    print("\n2. PROCESSED NPY FILES:")
    print("-" * 40)
    
    processed_paths = {
        'train': '/data/ahmed_data/ahmed_data/train/',
        'validation': '/data/ahmed_data/ahmed_data/validation/',
        'test': '/data/ahmed_data/ahmed_data/test/'
    }
    
    for name, path in processed_paths.items():
        if os.path.exists(path):
            npy_files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
            case_numbers = []
            for f in npy_files:
                if 'run_' in f:
                    try:
                        # Extract case number from filename like "run_123.npy"
                        case_num = int(f.split('run_')[1].split('.')[0])
                        case_numbers.append(case_num)
                    except:
                        pass
            
            results['processed_data'][name] = case_numbers
            
            print(f"  {name}: {path}")
            print(f"    NPY files: {len(npy_files)}")
            if case_numbers:
                print(f"    Case range: {min(case_numbers)} - {max(case_numbers)}")
                print(f"    First 5 cases: {case_numbers[:5]}")
                print(f"    Last 5 cases: {case_numbers[-5:]}")
                
                # Check if files have enhanced features (8 surface features)
                if npy_files:
                    sample_file = os.path.join(path, npy_files[0])
                    try:
                        data = np.load(sample_file, allow_pickle=True).item()
                        if 'surface_fields' in data:
                            surface_shape = data['surface_fields'].shape
                            print(f"    Surface fields shape: {surface_shape}")
                            if surface_shape[-1] == 8:
                                print(f"    ✅ Enhanced features detected (8 features)")
                            else:
                                print(f"    ⚠️  Standard features ({surface_shape[-1]} features)")
                    except Exception as e:
                        print(f"    Could not read sample file: {e}")
        else:
            print(f"  {name}: NOT FOUND at {path}")
    
    # 3. Determine the actual split
    print("\n3. DATA SPLIT ANALYSIS:")
    print("-" * 40)
    
    # Based on directory analysis, determine split
    all_cases = set()
    train_cases = set()
    val_cases = set()
    test_cases = set()
    
    # From raw data directories
    if 'fine_raw' in results['raw_data']:
        all_cases.update(results['raw_data']['fine_raw'])
        
    if 'fine_raw_test' in results['raw_data']:
        test_cases.update(results['raw_data']['fine_raw_test'])
    
    # From processed NPY files
    if 'train' in results['processed_data']:
        train_cases.update(results['processed_data']['train'])
    
    if 'validation' in results['processed_data']:
        val_cases.update(results['processed_data']['validation'])
        
    if 'test' in results['processed_data']:
        test_cases.update(results['processed_data']['test'])
    
    # If test cases aren't explicitly separated, infer from case numbers
    # Standard practice: cases 451-500 are often test cases
    if not test_cases and all_cases:
        potential_test = [c for c in all_cases if c >= 451 and c <= 500]
        if potential_test:
            test_cases.update(potential_test)
            train_cases = all_cases - test_cases - val_cases
    
    # Summary statistics
    print(f"Total unique cases found: {len(all_cases)}")
    if all_cases:
        print(f"  Range: {min(all_cases)} - {max(all_cases)}")
    
    print(f"\nTRAINING SET:")
    if train_cases:
        train_list = sorted(list(train_cases))
        print(f"  Cases: {len(train_list)}")
        print(f"  Range: {min(train_list)} - {max(train_list)}")
        print(f"  First 10: {train_list[:10]}")
        print(f"  Last 10: {train_list[-10:]}")
    else:
        print("  No explicit training cases identified")
    
    print(f"\nVALIDATION SET:")
    if val_cases:
        val_list = sorted(list(val_cases))
        print(f"  Cases: {len(val_list)}")
        print(f"  Range: {min(val_list)} - {max(val_list)}")
        print(f"  Examples: {val_list[:10]}")
    else:
        print("  No explicit validation cases identified")
    
    print(f"\nTEST SET:")
    if test_cases:
        test_list = sorted(list(test_cases))
        print(f"  Cases: {len(test_list)}")
        print(f"  Range: {min(test_list)} - {max(test_list)}")
        print(f"  First 10: {test_list[:10]}")
        print(f"  Last 10: {test_list[-10:]}")
    else:
        print("  No explicit test cases identified")
    
    # 4. Check config for data paths
    print("\n4. CONFIG.YAML SETTINGS:")
    print("-" * 40)
    
    config_path = Path("conf/config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Data paths in config:")
        print(f"  Training data: {config.get('data', {}).get('input_dir', 'Not set')}")
        print(f"  Validation data: {config.get('data', {}).get('input_dir_val', 'Not set')}")
        print(f"  Test data: {config.get('eval', {}).get('test_path', 'Not set')}")
        
        # Check enhanced features setting
        enhanced = config.get('data_processor', {}).get('use_enhanced_features', False)
        print(f"\n  Enhanced features enabled: {enhanced}")
    
    # 5. Save results to JSON for reference
    output_file = 'data_split_analysis.json'
    results['summary'] = {
        'total_cases': len(all_cases),
        'train_cases': sorted(list(train_cases)),
        'val_cases': sorted(list(val_cases)),
        'test_cases': sorted(list(test_cases)),
        'train_count': len(train_cases),
        'val_count': len(val_cases),
        'test_count': len(test_cases)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n5. DETAILED RESULTS SAVED TO: {output_file}")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS:")
    print("-" * 40)
    
    if test_cases:
        print("✅ Test cases identified. These should NOT have been seen during training:")
        test_list = sorted(list(test_cases))
        print(f"   Cases {min(test_list)} - {max(test_list)} ({len(test_list)} cases)")
        print("\n   You can safely test on these cases for unbiased evaluation.")
    else:
        print("⚠️  No explicit test cases found. Consider:")
        print("   - Using cases 451-500 as test set (standard practice)")
        print("   - Creating a proper train/val/test split")
    
    if not val_cases:
        print("\n⚠️  No validation set found. The model may have been trained without validation.")
        print("   This could lead to overfitting. Consider creating a validation set.")
    
    # Check for data leakage
    if train_cases and test_cases:
        overlap = train_cases.intersection(test_cases)
        if overlap:
            print("\n❌ WARNING: Data leakage detected!")
            print(f"   {len(overlap)} cases appear in both train and test sets:")
            print(f"   {sorted(list(overlap))[:20]}")
        else:
            print("\n✅ No data leakage detected between train and test sets.")
    
    return results


def check_specific_case(case_number):
    """Check where a specific case appears in the dataset"""
    
    print(f"\nChecking case run_{case_number}:")
    print("-" * 40)
    
    locations = []
    
    # Check raw directories
    paths_to_check = [
        ('/data/ahmed_data/ahmed_data/raw/', 'Fine Raw'),
        ('/data/ahmed_data/ahmed_data/raw_test/', 'Fine Test'),
        ('/data/ahmed_data/ahmed_data_rans/raw/', 'Coarse Raw'),
        ('/data/ahmed_data/ahmed_data_rans/raw_test/', 'Coarse Test'),
        ('/data/ahmed_data/ahmed_data/train/', 'Processed Train'),
        ('/data/ahmed_data/ahmed_data/validation/', 'Processed Val'),
        ('/data/ahmed_data/ahmed_data/test/', 'Processed Test'),
    ]
    
    for path, name in paths_to_check:
        case_dir = os.path.join(path, f'run_{case_number}')
        case_file = os.path.join(path, f'run_{case_number}.npy')
        
        if os.path.exists(case_dir):
            locations.append(name + " (directory)")
            print(f"  ✅ Found in {name}: {case_dir}")
        elif os.path.exists(case_file):
            locations.append(name + " (NPY file)")
            print(f"  ✅ Found in {name}: {case_file}")
    
    if not locations:
        print(f"  ❌ Case run_{case_number} not found in any location")
    else:
        print(f"\n  Summary: Case run_{case_number} found in:")
        for loc in locations:
            print(f"    - {loc}")
    
    return locations


if __name__ == "__main__":
    # Main analysis
    results = analyze_data_directories()
    
    # Check a few specific cases
    print("\n" + "="*80)
    print("SPOT CHECK OF SPECIFIC CASES")
    print("="*80)
    
    # Check some boundary cases
    test_cases = [1, 100, 200, 300, 400, 450, 451, 475, 500]
    for case in test_cases:
        check_specific_case(case)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings have been saved to 'data_split_analysis.json'")
    print("\nIMPORTANT: Only test on cases that were NOT in the training set!")
    print("Typically, cases 451-500 are reserved for testing.")

