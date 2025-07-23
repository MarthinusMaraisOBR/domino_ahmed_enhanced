#!/usr/bin/env python3
"""
Calculate Relative L2 and Area-Weighted Relative L2 Errors for Ahmed Body DoMINO Model
Based on DoMINO paper methodology (Equation 2)

This script processes the test results from your 500-epoch trained model and calculates
accurate relative L2 errors and area-weighted relative L2 errors as defined in the DoMINO paper.
"""

import os
import re
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def calculate_relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate relative L2 error as defined in DoMINO paper Equation 2:
    ε = √(∑(yT - yP)²) / √(∑yT²)
    
    Args:
        y_true: Ground truth values [n_points, n_features]
        y_pred: Predicted values [n_points, n_features]
    
    Returns:
        Relative L2 error as a scalar
    """
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum(y_true ** 2))
    
    # Avoid division by zero
    if denominator < 1e-12:
        return 0.0
    
    return numerator / denominator

def calculate_area_weighted_relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray, areas: np.ndarray) -> float:
    """
    Calculate area-weighted relative L2 error:
    ε = √(∑A*(yT - yP)²) / √(∑A*yT²)
    where A is the area of each surface element
    
    Args:
        y_true: Ground truth values [n_points, n_features]
        y_pred: Predicted values [n_points, n_features]
        areas: Area of each surface element [n_points,]
    
    Returns:
        Area-weighted relative L2 error as a scalar
    """
    # Ensure areas has the right shape for broadcasting
    if len(areas.shape) == 1:
        areas = areas[:, np.newaxis]
    
    numerator = np.sqrt(np.sum(areas * (y_true - y_pred) ** 2))
    denominator = np.sqrt(np.sum(areas * y_true ** 2))
    
    # Avoid division by zero
    if denominator < 1e-12:
        return 0.0
    
    return numerator / denominator

def load_vtp_data(vtp_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ground truth and predicted data from VTP file.
    
    Args:
        vtp_path: Path to VTP file with predictions
        
    Returns:
        Tuple of (ground_truth_fields, predicted_fields, surface_areas, surface_normals)
    """
    # Read VTP file
    reader = pv.get_reader(vtp_path)
    mesh = reader.read()
    
    # Extract ground truth fields (original CFD data)
    try:
        # Try to get original field names
        pressure_true = mesh.cell_data.get('pMean', mesh.cell_data.get('pressure', None))
        wall_shear_true = mesh.cell_data.get('wallShearStressMean', mesh.cell_data.get('wallShearStress', None))
        
        if pressure_true is None or wall_shear_true is None:
            raise ValueError("Could not find ground truth fields in VTP file")
            
        # Extract predicted fields
        pressure_pred = mesh.cell_data.get('pMeanPred', None)
        wall_shear_pred = mesh.cell_data.get('wallShearStressMeanPred', None)
        
        if pressure_pred is None or wall_shear_pred is None:
            raise ValueError("Could not find predicted fields in VTP file")
        
        # Combine fields [pressure, wall_shear_x, wall_shear_y, wall_shear_z]
        ground_truth = np.column_stack([
            pressure_true.reshape(-1, 1),
            wall_shear_true
        ])
        
        predictions = np.column_stack([
            pressure_pred.reshape(-1, 1),
            wall_shear_pred
        ])
        
    except Exception as e:
        print(f"Error extracting fields from {vtp_path}: {e}")
        print("Available cell data fields:", list(mesh.cell_data.keys()))
        raise
    
    # Calculate surface areas and normals
    mesh_with_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = mesh_with_areas.cell_data['Area']
    surface_normals = mesh.cell_normals
    
    return ground_truth, predictions, surface_areas, surface_normals

def process_test_directory(test_dir: str, predictions_dir: str, case_range: Tuple[int, int] = (451, 500)) -> Dict:
    """
    Process all test cases and calculate L2 errors.
    
    Args:
        test_dir: Directory containing ground truth VTP files
        predictions_dir: Directory containing predicted VTP files
        case_range: Range of case numbers to process (start, end) inclusive
        
    Returns:
        Dictionary with results for each test case
    """
    results = {}
    start_case, end_case = case_range
    
    print(f"Processing test cases {start_case} to {end_case}...")
    print("="*60)
    
    for case_num in range(start_case, end_case + 1):
        # Paths to files
        ground_truth_vtp = os.path.join(test_dir, f"run_{case_num}", f"boundary_{case_num}.vtp")
        predicted_vtp = os.path.join(predictions_dir, f"boundary_{case_num}_predicted.vtp")
        
        # Check if files exist
        if not os.path.exists(ground_truth_vtp):
            print(f"⚠️  Ground truth file not found: {ground_truth_vtp}")
            continue
            
        if not os.path.exists(predicted_vtp):
            print(f"⚠️  Predicted file not found: {predicted_vtp}")
            continue
        
        try:
            # Load ground truth data
            reader_true = pv.get_reader(ground_truth_vtp)
            mesh_true = reader_true.read()
            
            # Load predicted data
            reader_pred = pv.get_reader(predicted_vtp)
            mesh_pred = reader_pred.read()
            
            # Extract fields from ground truth
            pressure_true = None
            wall_shear_true = None
            
            # Try different possible field names
            for pressure_name in ['pMean', 'pressure', 'Pressure', 'p']:
                if pressure_name in mesh_true.cell_data:
                    pressure_true = mesh_true.cell_data[pressure_name]
                    break
                    
            for wall_shear_name in ['wallShearStressMean', 'wallShearStress', 'WallShearStress', 'tau_wall']:
                if wall_shear_name in mesh_true.cell_data:
                    wall_shear_true = mesh_true.cell_data[wall_shear_name]
                    break
            
            if pressure_true is None or wall_shear_true is None:
                print(f"⚠️  Could not find ground truth fields for case {case_num}")
                print(f"Available fields: {list(mesh_true.cell_data.keys())}")
                continue
            
            # Extract fields from predictions
            pressure_pred = None
            wall_shear_pred = None
            
            for pressure_name in ['pMeanPred', 'pressurePred', 'PressurePred']:
                if pressure_name in mesh_pred.cell_data:
                    pressure_pred = mesh_pred.cell_data[pressure_name]
                    break
                    
            for wall_shear_name in ['wallShearStressMeanPred', 'wallShearStressPred', 'WallShearStressPred']:
                if wall_shear_name in mesh_pred.cell_data:
                    wall_shear_pred = mesh_pred.cell_data[wall_shear_name]
                    break
            
            if pressure_pred is None or wall_shear_pred is None:
                print(f"⚠️  Could not find predicted fields for case {case_num}")
                print(f"Available fields: {list(mesh_pred.cell_data.keys())}")
                continue
            
            # Combine fields [pressure, wall_shear_x, wall_shear_y, wall_shear_z]
            ground_truth = np.column_stack([
                pressure_true.reshape(-1, 1),
                wall_shear_true
            ])
            
            predictions = np.column_stack([
                pressure_pred.reshape(-1, 1),
                wall_shear_pred
            ])
            
            # Calculate surface areas
            mesh_with_areas = mesh_true.compute_cell_sizes(length=False, area=True, volume=False)
            surface_areas = mesh_with_areas.cell_data['Area']
            
            # Calculate errors for each field
            field_names = ['pressure', 'wall_shear_x', 'wall_shear_y', 'wall_shear_z']
            case_results = {}
            
            for i, field_name in enumerate(field_names):
                field_true = ground_truth[:, i:i+1]
                field_pred = predictions[:, i:i+1]
                
                # Calculate regular relative L2 error
                rel_l2_error = calculate_relative_l2_error(field_true, field_pred)
                
                # Calculate area-weighted relative L2 error
                area_weighted_rel_l2_error = calculate_area_weighted_relative_l2_error(
                    field_true, field_pred, surface_areas
                )
                
                case_results[field_name] = {
                    'rel_l2': rel_l2_error,
                    'area_weighted_rel_l2': area_weighted_rel_l2_error
                }
            
            results[f'run_{case_num}'] = case_results
            print(f"✅ Processed case {case_num}")
            
        except Exception as e:
            print(f"❌ Error processing case {case_num}: {e}")
            continue
    
    return results

def calculate_average_errors(results: Dict) -> Dict:
    """
    Calculate average errors across all test cases.
    
    Args:
        results: Dictionary with results for each test case
        
    Returns:
        Dictionary with averaged error metrics
    """
    if not results:
        return {}
    
    field_names = ['pressure', 'wall_shear_x', 'wall_shear_y', 'wall_shear_z']
    averaged_results = {}
    
    for field_name in field_names:
        rel_l2_errors = []
        area_weighted_errors = []
        
        for case_name, case_results in results.items():
            if field_name in case_results:
                rel_l2_errors.append(case_results[field_name]['rel_l2'])
                area_weighted_errors.append(case_results[field_name]['area_weighted_rel_l2'])
        
        if rel_l2_errors:
            averaged_results[field_name] = {
                'rel_l2': np.mean(rel_l2_errors),
                'rel_l2_std': np.std(rel_l2_errors),
                'area_weighted_rel_l2': np.mean(area_weighted_errors),
                'area_weighted_rel_l2_std': np.std(area_weighted_errors),
                'num_cases': len(rel_l2_errors)
            }
    
    return averaged_results

def print_results_table(averaged_results: Dict):
    """
    Print results in DoMINO paper format.
    """
    print("\n" + "="*80)
    print("AHMED BODY DoMINO MODEL - COMPREHENSIVE ERROR ANALYSIS")
    print("="*80)
    print("Model: 500-epoch trained DoMINO (surface-only)")
    print("Test Cases: 50 (run_451 to run_500)")
    print("Methodology: DoMINO Paper Equation 2")
    print("="*80)
    
    print(f"\n📊 AVERAGE ERROR METRICS ACROSS TEST SET:")
    print("─"*80)
    print(f"{'FIELD':<20} {'REL. L-2':<12} {'±STD':<8} {'AREA-WEIGHTED':<15} {'±STD':<8} {'CASES':<6}")
    print("─"*80)
    
    field_display_names = {
        'pressure': 'PRESSURE',
        'wall_shear_x': 'X-WALL-SHEAR', 
        'wall_shear_y': 'Y-WALL-SHEAR',
        'wall_shear_z': 'Z-WALL-SHEAR'
    }
    
    for field_name, display_name in field_display_names.items():
        if field_name in averaged_results:
            result = averaged_results[field_name]
            print(f"{display_name:<20} {result['rel_l2']:<12.4f} {result['rel_l2_std']:<8.4f} "
                  f"{result['area_weighted_rel_l2']:<15.4f} {result['area_weighted_rel_l2_std']:<8.4f} "
                  f"{result['num_cases']:<6}")
    
    print("─"*80)
    
    # Comparison to DoMINO paper
    domino_paper_results = {
        'pressure': {'rel_l2': 0.1505, 'area_weighted_rel_l2': 0.1181},
        'wall_shear_x': {'rel_l2': 0.2124, 'area_weighted_rel_l2': 0.1279},
        'wall_shear_y': {'rel_l2': 0.3020, 'area_weighted_rel_l2': 0.2769},
        'wall_shear_z': {'rel_l2': 0.3359, 'area_weighted_rel_l2': 0.2290}
    }
    
    print(f"\n📋 COMPARISON TO DOMINO PAPER BENCHMARKS:")
    print("─"*80)
    print(f"{'FIELD':<20} {'YOUR REL L-2':<12} {'PAPER REL L-2':<12} {'YOUR AREA-WT':<12} {'PAPER AREA-WT':<12} {'STATUS':<10}")
    print("─"*80)
    
    for field_name, display_name in field_display_names.items():
        if field_name in averaged_results and field_name in domino_paper_results:
            your_result = averaged_results[field_name]
            paper_result = domino_paper_results[field_name]
            
            rel_l2_better = "✅" if your_result['rel_l2'] < paper_result['rel_l2'] else "⚠️"
            area_better = "✅" if your_result['area_weighted_rel_l2'] < paper_result['area_weighted_rel_l2'] else "⚠️"
            overall_status = "✅ Better" if (your_result['rel_l2'] < paper_result['rel_l2'] and 
                                           your_result['area_weighted_rel_l2'] < paper_result['area_weighted_rel_l2']) else "Mixed"
            
            print(f"{display_name:<20} {your_result['rel_l2']:<12.4f} {paper_result['rel_l2']:<12.4f} "
                  f"{your_result['area_weighted_rel_l2']:<12.4f} {paper_result['area_weighted_rel_l2']:<12.4f} "
                  f"{overall_status:<10}")
    
    print("─"*80)

def save_results_to_csv(averaged_results: Dict, individual_results: Dict, output_dir: str):
    """
    Save results to CSV files for further analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save averaged results
    avg_data = []
    for field_name, result in averaged_results.items():
        avg_data.append({
            'Field': field_name,
            'Rel_L2_Mean': result['rel_l2'],
            'Rel_L2_Std': result['rel_l2_std'],
            'Area_Weighted_Rel_L2_Mean': result['area_weighted_rel_l2'],
            'Area_Weighted_Rel_L2_Std': result['area_weighted_rel_l2_std'],
            'Num_Cases': result['num_cases']
        })
    
    avg_df = pd.DataFrame(avg_data)
    avg_csv_path = os.path.join(output_dir, "averaged_l2_errors.csv")
    avg_df.to_csv(avg_csv_path, index=False)
    print(f"\n💾 Averaged results saved to: {avg_csv_path}")
    
    # Save individual results
    individual_data = []
    for case_name, case_results in individual_results.items():
        for field_name, metrics in case_results.items():
            individual_data.append({
                'Case': case_name,
                'Field': field_name,
                'Rel_L2': metrics['rel_l2'],
                'Area_Weighted_Rel_L2': metrics['area_weighted_rel_l2']
            })
    
    individual_df = pd.DataFrame(individual_data)
    individual_csv_path = os.path.join(output_dir, "individual_l2_errors.csv")
    individual_df.to_csv(individual_csv_path, index=False)
    print(f"💾 Individual results saved to: {individual_csv_path}")

def main():
    """
    Main function to process test results and calculate L2 errors.
    """
    parser = argparse.ArgumentParser(description='Calculate L2 errors for Ahmed body DoMINO model')
    parser.add_argument('--test_dir', type=str, default='/data/ahmed_data/ahmed_data/raw_test/',
                        help='Directory containing ground truth test data')
    parser.add_argument('--predictions_dir', type=str, default='/data/ahmed_data/predictions/',
                        help='Directory containing model predictions')
    parser.add_argument('--output_dir', type=str, default='/data/ahmed_data/analysis/',
                        help='Directory to save analysis results')
    parser.add_argument('--start_case', type=int, default=451,
                        help='Starting case number')
    parser.add_argument('--end_case', type=int, default=500,
                        help='Ending case number')
    
    args = parser.parse_args()
    
    # Process test cases
    print("🚀 Starting L2 Error Analysis for Ahmed Body DoMINO Model")
    print("="*60)
    
    results = process_test_directory(
        test_dir=args.test_dir,
        predictions_dir=args.predictions_dir,
        case_range=(args.start_case, args.end_case)
    )
    
    if not results:
        print("❌ No results to process. Check your file paths and data.")
        return
    
    # Calculate averages
    averaged_results = calculate_average_errors(results)
    
    # Print results
    print_results_table(averaged_results)
    
    # Save results
    save_results_to_csv(averaged_results, results, args.output_dir)
    
    print(f"\n🎉 Analysis complete! Processed {len(results)} test cases.")
    print("="*60)

if __name__ == "__main__":
    main()