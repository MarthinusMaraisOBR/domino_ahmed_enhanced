#!/usr/bin/env python3
"""diagnose_predictions.py - Quantitative analysis of model predictions"""

import numpy as np
import torch
from pathlib import Path
import pyvista as pv

def analyze_predictions():
    """Extract and analyze prediction statistics."""
    
    print("="*80)
    print("PHASE 1: QUANTITATIVE PREDICTION ANALYSIS")
    print("="*80)
    
    # Load multiple test predictions
    test_cases = []
    for i in range(451, 456):
        vtp_path = Path(f"/data/ahmed_data/predictions_fixed/boundary_{i}_comprehensive_comparison.vtp")
        if vtp_path.exists():
            mesh = pv.read(str(vtp_path))
            test_cases.append({
                'case': i,
                'coarse_p': np.array(mesh['Coarse_Pressure']),
                'fine_p': np.array(mesh['Fine_Pressure_GroundTruth_Interpolated']),
                'pred_p': np.array(mesh['Predicted_Pressure']),
                'coarse_wss': np.array(mesh['Coarse_WallShearStress']),
                'fine_wss': np.array(mesh['Fine_WallShearStress_GroundTruth_Interpolated']),
                'pred_wss': np.array(mesh['Predicted_WallShearStress'])
            })
    
    print(f"\nLoaded {len(test_cases)} test cases\n")
    
    # Analyze each case
    for case in test_cases:
        print(f"Case {case['case']}:")
        print("-"*40)
        
        # Pressure statistics
        print("Pressure Statistics:")
        for name, data in [('Coarse', case['coarse_p']), 
                           ('Fine', case['fine_p']), 
                           ('Predicted', case['pred_p'])]:
            print(f"  {name:10}: mean={data.mean():8.5f}, std={data.std():8.5f}, "
                  f"min={data.min():8.5f}, max={data.max():8.5f}, "
                  f"range={data.max()-data.min():8.5f}")
        
        # Check if predictions are constant
        pred_unique = np.unique(case['pred_p'].round(decimals=6))
        print(f"\n  Unique prediction values: {len(pred_unique)}")
        if len(pred_unique) < 10:
            print(f"  ⚠️ WARNING: Only {len(pred_unique)} unique values!")
            print(f"  Values: {pred_unique[:10]}")
        
        # Compute relative standard deviation
        coarse_rsd = case['coarse_p'].std() / (np.abs(case['coarse_p'].mean()) + 1e-10)
        fine_rsd = case['fine_p'].std() / (np.abs(case['fine_p'].mean()) + 1e-10)
        pred_rsd = case['pred_p'].std() / (np.abs(case['pred_p'].mean()) + 1e-10)
        
        print(f"\n  Relative Std Dev:")
        print(f"    Coarse: {coarse_rsd:.4f}")
        print(f"    Fine:   {fine_rsd:.4f}")
        print(f"    Pred:   {pred_rsd:.4f}")
        
        if pred_rsd < 0.01:
            print(f"  🔴 CRITICAL: Predictions are essentially constant!")
        
        print()
    
    return test_cases

if __name__ == "__main__":
    analyze_predictions()
