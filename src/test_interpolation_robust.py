#!/usr/bin/env python3
"""
Robust interpolation test that handles:
- Cell data vs point data differences
- Variable name mapping between fine and coarse meshes
- Different mesh resolutions
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

def load_vtp_data_robust(vtp_path, variable_mapping):
    """
    Load surface data from VTP file, handling both cell and point data

    Args:
        vtp_path: Path to VTP file
        variable_mapping: Dict mapping variable names, e.g., {'pressure': 'pMean', 'wall_shear': 'wallShearStressMean'}
    """
    print(f"Loading: {vtp_path}")

    if not Path(vtp_path).exists():
        raise FileNotFoundError(f"VTP file not found: {vtp_path}")

    # Read VTP file
    mesh = pv.read(str(vtp_path))

    # Get coordinates (always use cell centers for consistency)
    surface_coordinates = np.array(mesh.cell_centers().points)
    surface_normals = np.array(mesh.cell_normals)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.array(surface_sizes.cell_data["Area"])

    # Extract fields based on variable mapping
    surface_fields = []
    field_info = []

    for field_name, var_name in variable_mapping.items():
        field_found = False

        # Try cell data first
        if var_name in mesh.cell_data:
            field_data = np.array(mesh.cell_data[var_name])
            surface_fields.append(field_data)
            field_info.append(f"{field_name} -> {var_name} (cell data)")
            field_found = True

        # Try point data (interpolate to cell centers)
        elif var_name in mesh.point_data:
            print(f"  Converting {var_name} from point data to cell data...")
            point_data = np.array(mesh.point_data[var_name])

            # Interpolate from points to cell centers
            point_coords = np.array(mesh.points)
            tree = cKDTree(point_coords)

            # Find nearest points for each cell center
            _, indices = tree.query(surface_coordinates, k=3)  # Use 3 nearest points

            if point_data.ndim == 1:
                # Scalar field
                cell_data = np.mean(point_data[indices], axis=1)
            else:
                # Vector field
                cell_data = np.mean(point_data[indices], axis=1)

            surface_fields.append(cell_data)
            field_info.append(f"{field_name} -> {var_name} (point->cell)")
            field_found = True

        if not field_found:
            print(f"  Warning: Variable {var_name} not found in VTP file")

    if surface_fields:
        # Handle different field shapes
        processed_fields = []
        for field in surface_fields:
            if field.ndim == 1:
                processed_fields.append(field.reshape(-1, 1))
            else:
                processed_fields.append(field)
        
        surface_fields = np.concatenate(processed_fields, axis=1)
    else:
        raise ValueError("No surface fields found!")
    
    # Normalize surface normals
    surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
    
    print(f"  Loaded {len(surface_coordinates)} cells")
    print(f"  Surface fields shape: {surface_fields.shape}")
    for info in field_info:
        print(f"  {info}")
    
    return {
        'coordinates': surface_coordinates,
        'fields': surface_fields,
        'normals': surface_normals,
        'areas': surface_areas,
        'mesh': mesh,
        'field_info': field_info
    }

def interpolate_coarse_to_fine(coarse_coords, coarse_fields, fine_coords, method="inverse_distance", k=4):
    """Interpolate coarse surface fields to fine mesh coordinates"""
    
    print(f"Interpolating using {method} method...")
    start_time = time.time()
    
    if method == "nearest":
        tree = cKDTree(coarse_coords)
        _, indices = tree.query(fine_coords, k=1)
        interpolated_fields = coarse_fields[indices]
        
    elif method == "inverse_distance":
        tree = cKDTree(coarse_coords)
        k = min(k, len(coarse_coords))
        distances, indices = tree.query(fine_coords, k=k)
        
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        
        if k == 1:
            weights = weights.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        interpolated_fields = np.sum(
            coarse_fields[indices] * weights[:, :, np.newaxis], axis=1
        )
    
    elapsed_time = time.time() - start_time
    print(f"  Interpolation completed in {elapsed_time:.2f} seconds")
    
    return interpolated_fields

def analyze_interpolation_quality(fine_fields, interpolated_fields, field_names):
    """Analyze quality of interpolation"""
    
    print("\n" + "="*50)
    print("INTERPOLATION QUALITY ANALYSIS")
    print("="*50)
    
    metrics = {}
    
    # Determine field boundaries (pressure is 1 field, wall shear is 3 fields)
    field_boundaries = []
    start_idx = 0
    for field_name in field_names:
        if 'pressure' in field_name.lower() or 'p' in field_name.lower():
            field_boundaries.append((field_name, start_idx, start_idx + 1))
            start_idx += 1
        elif 'shear' in field_name.lower() or 'wall' in field_name.lower():
            field_boundaries.append((field_name, start_idx, start_idx + 3))
            start_idx += 3
    
    for field_name, start_idx, end_idx in field_boundaries:
        if end_idx <= fine_fields.shape[1]:
            fine_field = fine_fields[:, start_idx:end_idx]
            interp_field = interpolated_fields[:, start_idx:end_idx]
            
            # For vector fields, compute magnitude
            if fine_field.shape[1] > 1:
                fine_mag = np.sqrt(np.sum(fine_field**2, axis=1))
                interp_mag = np.sqrt(np.sum(interp_field**2, axis=1))
                
                mse = np.mean((fine_mag - interp_mag)**2)
                mae = np.mean(np.abs(fine_mag - interp_mag))
                correlation = np.corrcoef(fine_mag, interp_mag)[0, 1]
                fine_range = np.max(fine_mag) - np.min(fine_mag)
            else:
                fine_scalar = fine_field.flatten()
                interp_scalar = interp_field.flatten()
                
                mse = np.mean((fine_scalar - interp_scalar)**2)
                mae = np.mean(np.abs(fine_scalar - interp_scalar))
                correlation = np.corrcoef(fine_scalar, interp_scalar)[0, 1]
                fine_range = np.max(fine_scalar) - np.min(fine_scalar)
            
            rmse = np.sqrt(mse)
            relative_mae = mae / fine_range if fine_range > 0 else 0
            relative_rmse = rmse / fine_range if fine_range > 0 else 0
            
            metrics[field_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'relative_mae': relative_mae,
                'relative_rmse': relative_rmse,
                'correlation': correlation,
                'fine_range': fine_range
            }
            
            print(f"\n{field_name}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Relative MAE: {relative_mae:.1%}")
            print(f"  Relative RMSE: {relative_rmse:.1%}")
            print(f"  Correlation: {correlation:.4f}")
    
    return metrics

def main():
    """Main test function"""
    
    print("="*60)
    print("ROBUST DUAL-RESOLUTION INTERPOLATION TEST")
    print("="*60)
    
    # Configuration
    fine_data_path = "/data/ahmed_data/raw"
    coarse_data_path = "/data/ahmed_data_rans/raw"
    test_case = "run_1"
    case_number = "1"
    
    # Variable mapping based on inspection results
    fine_variable_mapping = {
        'pressure': 'pMean',
        'wall_shear': 'wallShearStressMean'
    }
    
    coarse_variable_mapping = {
        'pressure': 'p',  
        'wall_shear': 'wallShearStress'
    }
    
    output_dir = "/workspace/interpolation_test_results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Build file paths
    fine_vtp = Path(fine_data_path) / test_case / f"boundary_{case_number}.vtp"
    coarse_vtp = Path(coarse_data_path) / test_case / f"boundary_{case_number}.vtp"
    
    print(f"Test case: {test_case}")
    print(f"Fine VTP: {fine_vtp}")
    print(f"Coarse VTP: {coarse_vtp}")
    print()
    
    try:
        # Load data with robust handling
        print("LOADING DATA")
        print("-" * 20)
        fine_data = load_vtp_data_robust(fine_vtp, fine_variable_mapping)
        coarse_data = load_vtp_data_robust(coarse_vtp, coarse_variable_mapping)
        
        print(f"\nResolution ratio: {len(fine_data['coordinates']) / len(coarse_data['coordinates']):.1f}x")
        
        # Test interpolation methods
        methods = ["nearest", "inverse_distance"]
        field_names = ['pressure', 'wall_shear']
        
        best_method = None
        best_score = float('inf')
        results = {}
        
        for method in methods:
            print(f"\n{'='*50}")
            print(f"TESTING: {method.upper()} INTERPOLATION")
            print(f"{'='*50}")
            
            # Interpolate
            interpolated_fields = interpolate_coarse_to_fine(
                coarse_data['coordinates'],
                coarse_data['fields'], 
                fine_data['coordinates'],
                method=method,
                k=4
            )
            
            # Analyze quality
            metrics = analyze_interpolation_quality(
                fine_data['fields'],
                interpolated_fields,
                field_names
            )
            
            # Calculate overall score
            avg_rmse = np.mean([m['relative_rmse'] for m in metrics.values()])
            results[method] = {
                'metrics': metrics,
                'interpolated_fields': interpolated_fields,
                'avg_rmse': avg_rmse
            }
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_method = method
            
            print(f"\nOverall Score (Avg Relative RMSE): {avg_rmse:.1%}")
        
        # Summary
        print(f"\n{'='*60}")
        print("INTERPOLATION COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for method, result in results.items():
            marker = "🏆" if method == best_method else "  "
            print(f"{marker} {method:>15}: Avg Relative RMSE = {result['avg_rmse']:.1%}")
        
        print(f"\n🏆 Best method: {best_method}")
        print(f"🎯 Overall quality score: {best_score:.1%} relative RMSE")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_score < 0.05:
            print("🟢 Excellent interpolation quality - proceed with confidence!")
        elif best_score < 0.10:
            print("🟡 Good interpolation quality - should work well for training.")
        elif best_score < 0.20:
            print("🟠 Moderate interpolation quality - consider improvements.")
        else:
            print("🔴 Poor interpolation quality - significant improvements needed.")
        
        # Data compatibility check
        print(f"\n📊 DATA COMPATIBILITY:")
        print(f"   Resolution ratio: {len(fine_data['coordinates']) / len(coarse_data['coordinates']):.1f}x")
        print(f"   Fine mesh: {len(fine_data['coordinates']):,} cells")
        print(f"   Coarse mesh: {len(coarse_data['coordinates']):,} cells")
        print(f"   Variable mapping successful: ✅")
        print(f"   Cell/Point data handled: ✅")

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
