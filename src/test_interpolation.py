#!/usr/bin/env python3
"""
Test script to validate interpolation between coarse and fine VTP files

This script:
1. Loads a pair of coarse/fine VTP files
2. Tests different interpolation methods
3. Analyzes interpolation quality
4. Saves visualization data

Run this inside your enhanced container:
cd /workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src
python test_interpolation.py
"""

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

# Import DoMINO utilities (if available)
try:
    from physicsnemo.utils.domino.utils import get_fields, get_node_to_elem
    DOMINO_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: DoMINO utils not available, using basic VTK functions")
    DOMINO_UTILS_AVAILABLE = False

def get_fields_basic(celldata, variable_names):
    """Basic field extraction if DoMINO utils not available"""
    fields = []
    for var_name in variable_names:
        if celldata.GetArray(var_name):
            array = celldata.GetArray(var_name)
            # Convert VTK array to numpy
            field_data = np.zeros((array.GetNumberOfTuples(), array.GetNumberOfComponents()))
            for i in range(array.GetNumberOfTuples()):
                for j in range(array.GetNumberOfComponents()):
                    field_data[i, j] = array.GetComponent(i, j)
            fields.append(field_data)
        else:
            print(f"Warning: Variable {var_name} not found in VTP file")
    return fields

def load_vtp_data(vtp_path, surface_variables):
    """Load surface data from VTP file"""
    print(f"Loading: {vtp_path}")

    if not Path(vtp_path).exists():
        raise FileNotFoundError(f"VTP file not found: {vtp_path}")

    # Read VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    polydata = reader.GetOutput()

    # Extract solution fields
    if DOMINO_UTILS_AVAILABLE:
        celldata_all = get_node_to_elem(polydata)
        celldata = celldata_all.GetCellData()
        surface_fields = get_fields(celldata, surface_variables)
    else:
        celldata = polydata.GetCellData()
        surface_fields = get_fields_basic(celldata, surface_variables)

    surface_fields = np.concatenate(surface_fields, axis=-1)

    # Extract geometry data
    mesh = pv.PolyData(polydata)
    surface_coordinates = np.array(mesh.cell_centers().points)
    surface_normals = np.array(mesh.cell_normals)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_sizes = np.array(surface_sizes.cell_data["Area"])

    # Normalize surface normals
    surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]

    print(f"  Loaded {len(surface_coordinates)} cells")
    print(f"  Surface fields shape: {surface_fields.shape}")
    print(f"  Field variables: {surface_variables}")

    return {
        'coordinates': surface_coordinates,
        'fields': surface_fields,
        'normals': surface_normals,
        'areas': surface_sizes,
        'polydata': polydata
    }

def interpolate_coarse_to_fine(coarse_coords, coarse_fields, fine_coords, method="inverse_distance", k=4):
    """Interpolate coarse surface fields to fine mesh coordinates"""

    print(f"Interpolating using {method} method...")
    start_time = time.time()

    if method == "nearest":
        # Simple nearest neighbor interpolation
        tree = cKDTree(coarse_coords)
        _, indices = tree.query(fine_coords, k=1)
        interpolated_fields = coarse_fields[indices]

    elif method == "inverse_distance":
        # Inverse distance weighting with k nearest neighbors
        tree = cKDTree(coarse_coords)
        k = min(k, len(coarse_coords))
        distances, indices = tree.query(fine_coords, k=k)

        # Handle case where distance is exactly zero
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)

        # Handle single neighbor case
        if k == 1:
            weights = weights.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Normalize weights
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Interpolate fields
        interpolated_fields = np.sum(
            coarse_fields[indices] * weights[:, :, np.newaxis], axis=1
        )

    elif method == "rbf":
        # Radial Basis Function interpolation
        try:
            from scipy.interpolate import RBFInterpolator

            interpolated_fields = np.zeros((len(fine_coords), coarse_fields.shape[1]))

            for field_idx in range(coarse_fields.shape[1]):
                rbf = RBFInterpolator(
                    coarse_coords, 
                    coarse_fields[:, field_idx],
                    kernel='thin_plate_spline',
                    epsilon=1e-6
                )
                interpolated_fields[:, field_idx] = rbf(fine_coords)
        except ImportError:
            print("Warning: scipy.interpolate.RBFInterpolator not available, falling back to nearest neighbor")
            return interpolate_coarse_to_fine(coarse_coords, coarse_fields, fine_coords, "nearest", k)

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    elapsed_time = time.time() - start_time
    print(f"  Interpolation completed in {elapsed_time:.2f} seconds")

    return interpolated_fields

def analyze_interpolation_quality(fine_fields, interpolated_fields, field_names):
    """Analyze quality of interpolation"""

    print("\n" + "="*50)
    print("INTERPOLATION QUALITY ANALYSIS")
    print("="*50)

    metrics = {}

    for i, field_name in enumerate(field_names):
        if i < fine_fields.shape[1]:
            fine_field = fine_fields[:, i]
            interp_field = interpolated_fields[:, i]

            # Calculate metrics
            mse = np.mean((fine_field - interp_field)**2)
            mae = np.mean(np.abs(fine_field - interp_field))
            rmse = np.sqrt(mse)

            # Relative errors
            fine_range = np.max(fine_field) - np.min(fine_field)
            relative_mae = mae / fine_range if fine_range > 0 else 0
            relative_rmse = rmse / fine_range if fine_range > 0 else 0

            # Correlation
            correlation = np.corrcoef(fine_field, interp_field)[0, 1]

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
            print(f"  Fine field range: [{np.min(fine_field):.6f}, {np.max(fine_field):.6f}]")

    return metrics

def create_comparison_plots(fine_data, coarse_data, interpolated_fields, field_names, output_dir):
    """Create comparison plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nCreating comparison plots in: {output_dir}")

    # Plot field comparisons
    for i, field_name in enumerate(field_names):
        if i < fine_data['fields'].shape[1]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            fine_field = fine_data['fields'][:, i] 
            interp_field = interpolated_fields[:, i]

            # Fine field
            scatter1 = axes[0].scatter(fine_data['coordinates'][:, 0], 
                                     fine_data['coordinates'][:, 1], 
                                     c=fine_field, cmap='viridis', s=1)
            axes[0].set_title(f'{field_name} - Fine')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[0])

            # Interpolated field
            scatter2 = axes[1].scatter(fine_data['coordinates'][:, 0],
                                     fine_data['coordinates'][:, 1], 
                                     c=interp_field, cmap='viridis', s=1)
            axes[1].set_title(f'{field_name} - Interpolated')
            axes[1].set_xlabel('X') 
            axes[1].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[1])

            # Error
            error = fine_field - interp_field
            scatter3 = axes[2].scatter(fine_data['coordinates'][:, 0],
                                     fine_data['coordinates'][:, 1],
                                     c=error, cmap='RdBu', s=1)
            axes[2].set_title(f'{field_name} - Error')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            plt.colorbar(scatter3, ax=axes[2])

            plt.tight_layout()
            plt.savefig(output_dir / f'{field_name}_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

    # Mesh resolution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coarse mesh
    axes[0].scatter(coarse_data['coordinates'][:, 0], 
                   coarse_data['coordinates'][:, 1], s=2, alpha=0.6)
    axes[0].set_title(f'Coarse Mesh ({len(coarse_data["coordinates"])} cells)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Fine mesh
    axes[1].scatter(fine_data['coordinates'][:, 0],
                   fine_data['coordinates'][:, 1], s=0.5, alpha=0.6)
    axes[1].set_title(f'Fine Mesh ({len(fine_data["coordinates"])} cells)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(output_dir / 'mesh_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Plots saved successfully!")

def save_interpolated_vtp(fine_data, interpolated_fields, field_names, output_path):
    """Save interpolated fields to VTP file for visualization"""

    # Create new polydata with interpolated fields
    polydata = fine_data['polydata']

    # Add interpolated fields as cell data
    for i, field_name in enumerate(field_names):
        if i < interpolated_fields.shape[1]:
            from vtk.util import numpy_support
            field_array = numpy_support.numpy_to_vtk(interpolated_fields[:, i:i+1])
            field_array.SetName(f"{field_name}_interpolated")
            polydata.GetCellData().AddArray(field_array)

    # Write VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(polydata)
    writer.Write()

    print(f"Interpolated VTP saved to: {output_path}")

def main():
    """Main test function"""
    
    print("="*60)
    print("DOMINO DUAL-RESOLUTION INTERPOLATION TEST")
    print("="*60)
    
    # Configuration - UPDATED FOR YOUR DATA STRUCTURE
    fine_data_path = "/data/ahmed_data/raw"           # Fine data inside container
    coarse_data_path = "/data/ahmed_data_rans/raw"    # Coarse data inside container
    
    # Test case - UPDATED FOR YOUR NAMING CONVENTION  
    test_case = "run_1"  # Your cases are run_1, run_2, etc.
    case_number = "1"    # Extract number for file naming
    
    # Surface variables - We'll check what's actually available
    surface_variables = ["pMean", "wallShearStress"]  # Common CFD variables
    
    # Output directory
    output_dir = "/workspace/interpolation_test_results"
    
    # Build file paths - UPDATED FOR YOUR NAMING PATTERN
    fine_vtp = Path(fine_data_path) / test_case / f"boundary_{case_number}.vtp"
    coarse_vtp = Path(coarse_data_path) / test_case / f"boundary_{case_number}.vtp"
    
    print(f"Test case: {test_case}")
    print(f"Fine VTP: {fine_vtp}")
    print(f"Coarse VTP: {coarse_vtp}")
    print(f"Surface variables: {surface_variables}")
    print()
    
    # Check files exist and show available files if not
    if not fine_vtp.exists():
        print(f"❌ Fine VTP file not found: {fine_vtp}")
        print("Available fine cases:")
        try:
            fine_base = Path(fine_data_path)
            if fine_base.exists():
                for case_dir in sorted(fine_base.iterdir()):
                    if case_dir.is_dir():
                        files = list(case_dir.glob("*.vtp"))
                        print(f"  {case_dir.name}: {[f.name for f in files]}")
            else:
                print(f"  Fine data path doesn't exist: {fine_base}")
        except Exception as e:
            print(f"  Error listing fine data: {e}")
        return
    
    if not coarse_vtp.exists():
        print(f"❌ Coarse VTP file not found: {coarse_vtp}")
        print("Available coarse cases:")
        try:
            coarse_base = Path(coarse_data_path)
            if coarse_base.exists():
                for case_dir in sorted(coarse_base.iterdir()):
                    if case_dir.is_dir():
                        files = list(case_dir.glob("*.vtp"))
                        print(f"  {case_dir.name}: {[f.name for f in files]}")
            else:
                print(f"  Coarse data path doesn't exist: {coarse_base}")
        except Exception as e:
            print(f"  Error listing coarse data: {e}")
        return
    
    try:
        # Load data
        print("LOADING DATA")
        print("-" * 20)
        fine_data = load_vtp_data(fine_vtp, surface_variables)
        coarse_data = load_vtp_data(coarse_vtp, surface_variables)
        
        print(f"\nResolution ratio: {len(fine_data['coordinates']) / len(coarse_data['coordinates']):.1f}x")
        
        # Test different interpolation methods
        methods = ["nearest", "inverse_distance"]
        
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
                surface_variables
            )
            
            # Calculate overall score (average relative RMSE)
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
        
        # Create visualizations using best method
        print(f"\nCreating visualizations using {best_method} method...")
        create_comparison_plots(
            fine_data,
            coarse_data,
            results[best_method]['interpolated_fields'],
            surface_variables,
            output_dir
        )
        
        # Save interpolated VTP
        output_vtp = Path(output_dir) / f"{test_case}_interpolated.vtp"
        save_interpolated_vtp(
            fine_data,
            results[best_method]['interpolated_fields'],
            surface_variables,
            output_vtp
        )
        
        print(f"\n✅ INTERPOLATION TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Best interpolation method: {best_method}")
        print(f"🎯 Overall quality score: {best_score:.1%} relative RMSE")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best_score < 0.05:  # < 5% relative error
            print("🟢 Excellent interpolation quality - proceed with confidence!")
        elif best_score < 0.10:  # < 10% relative error  
            print("🟡 Good interpolation quality - should work well for training.")
        else:
            print("🔴 Poor interpolation quality - consider:")
            print("   - Higher resolution coarse mesh")
            print("   - Different interpolation method")
            print("   - Feature engineering approaches")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to diagnose the issue
        print(f"\n🔍 DIAGNOSTICS:")
        
        # Check if files can be read
        try:
            import pyvista as pv
            fine_mesh = pv.read(str(fine_vtp))
            print(f"✅ Fine VTP readable - {fine_mesh.n_cells} cells")
            print(f"   Available cell data: {list(fine_mesh.cell_data.keys())}")
        except Exception as e2:
            print(f"❌ Cannot read fine VTP: {e2}")
            
        try:
            coarse_mesh = pv.read(str(coarse_vtp))
            print(f"✅ Coarse VTP readable - {coarse_mesh.n_cells} cells")
            print(f"   Available cell data: {list(coarse_mesh.cell_data.keys())}")
        except Exception as e2:
            print(f"❌ Cannot read coarse VTP: {e2}")

if __name__ == "__main__":
    main()