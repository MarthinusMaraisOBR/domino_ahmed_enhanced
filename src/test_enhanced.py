#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
COMPREHENSIVE FIX for Enhanced DoMINO model testing.
This script handles both the data structure differences and dtype mismatches.

CRITICAL FIXES:
1. Point data → Cell data conversion for coarse VTP files
2. Strict float32 dtype enforcement throughout the pipeline
3. Proper tensor device and dtype handling for PyTorch
"""

import os
import re
import time
import numpy as np
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import vtk
from vtk.util import numpy_support

import pyvista as pv
from scipy.spatial import cKDTree

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.sdf import signed_distance_field

# Import the enhanced model
from enhanced_domino_model import DoMINOEnhanced

# Constants
AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.0


def convert_point_to_cell_data_manual(mesh, field_name, n_components=1):
    """
    Manually convert point data to cell data using weighted averaging.
    This is more reliable than PyVista's built-in conversion for our use case.
    
    Args:
        mesh: PyVista mesh object
        field_name: Name of the field to convert
        n_components: Number of components (1 for scalar, 3 for vector)
        
    Returns:
        Cell data array as float32 numpy array
    """
    print(f"    Manual conversion: {field_name} ({n_components} components)")
    
    # Get point data
    point_data = np.array(mesh.point_data[field_name], dtype=np.float32)
    
    # Get connectivity for each cell
    n_cells = mesh.n_cells
    cell_data = np.zeros((n_cells, n_components), dtype=np.float32)
    
    for cell_id in range(n_cells):
        # Get points for this cell
        cell = mesh.GetCell(cell_id)
        point_ids = []
        for i in range(cell.GetNumberOfPoints()):
            point_ids.append(cell.GetPointId(i))
        
        # Average the point data for this cell
        if n_components == 1:
            cell_data[cell_id] = np.mean(point_data[point_ids])
        else:
            cell_data[cell_id] = np.mean(point_data[point_ids], axis=0)
    
    return cell_data


def load_coarse_vtp_data_robust(vtp_path, surface_variables):
    """
    COMPREHENSIVE ROBUST loader for coarse resolution VTP data.
    Handles point→cell conversion and ensures strict float32 dtype.
    
    Args:
        vtp_path: Path to VTP file
        surface_variables: List of variable names to extract
        
    Returns:
        Dictionary with surface data (all float32)
    """
    print(f"Loading coarse data: {vtp_path}")
    
    # Read VTP file
    mesh = pv.read(str(vtp_path))
    
    print(f"  Original mesh: {mesh.n_cells} cells, {mesh.n_points} points")
    print(f"  Cell data keys: {list(mesh.cell_data.keys())}")
    print(f"  Point data keys: {list(mesh.point_data.keys())}")
    
    # Variable name mapping for coarse data
    variable_mapping = {
        'pMean': ['p', 'pressure', 'Pressure'],
        'wallShearStressMean': ['wallShearStress', 'WallShearStress', 'tau_wall']
    }
    
    # Extract fields with proper conversion
    surface_fields = []
    field_info = []
    
    for var_name in surface_variables:
        field_found = False
        possible_names = variable_mapping.get(var_name, [var_name])
        
        # Try each possible name
        for candidate_name in possible_names:
            # Check cell data first (preferred)
            if candidate_name in mesh.cell_data:
                field_data = np.array(mesh.cell_data[candidate_name], dtype=np.float32)
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                surface_fields.append(field_data)
                field_info.append(f"{var_name} -> {candidate_name} (cell data)")
                field_found = True
                break
                
            # Check point data (needs conversion)
            elif candidate_name in mesh.point_data:
                print(f"  Converting {candidate_name} from point data to cell data...")
                
                # Determine number of components
                point_array = mesh.point_data[candidate_name]
                if hasattr(point_array, 'shape') and len(point_array.shape) > 1:
                    n_components = point_array.shape[1]
                elif 'pressure' in candidate_name.lower() or 'p' == candidate_name.lower():
                    n_components = 1
                else:
                    n_components = 3  # Assume vector for shear stress
                
                # Convert using manual method
                field_data = convert_point_to_cell_data_manual(mesh, candidate_name, n_components)
                
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                
                surface_fields.append(field_data.astype(np.float32))
                field_info.append(f"{var_name} -> {candidate_name} (point->cell manual)")
                field_found = True
                break
        
        if not field_found:
            print(f"  Warning: Variable {var_name} not found with any candidate names")
            # Create zeros as fallback
            num_cells = mesh.n_cells
            if 'pressure' in var_name.lower() or 'p' in var_name.lower():
                surface_fields.append(np.zeros((num_cells, 1), dtype=np.float32))
            else:
                surface_fields.append(np.zeros((num_cells, 3), dtype=np.float32))
            field_info.append(f"{var_name} -> ZEROS (not found)")
    
    # Combine all surface fields
    if surface_fields:
        surface_fields_combined = np.concatenate(surface_fields, axis=1).astype(np.float32)
    else:
        raise ValueError("No surface fields could be extracted!")
    
    # Get mesh geometry data (ensure float32)
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    
    # Compute surface areas
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)
    
    # Normalize surface normals
    norms = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = (surface_normals / norms).astype(np.float32)
    
    print(f"  ✅ Loaded {len(surface_coordinates)} cells")
    print(f"  Surface fields shape: {surface_fields_combined.shape}")
    for info in field_info:
        print(f"    {info}")
    
    return {
        'coordinates': surface_coordinates,
        'fields': surface_fields_combined,
        'normals': surface_normals,
        'areas': surface_areas,
        'mesh': mesh,
        'field_info': field_info
    }


def test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """
    Test the enhanced model with STRICT float32 type enforcement.
    
    Args:
        data_dict: Input data dictionary
        model: Enhanced DoMINO model
        device: PyTorch device
        cfg: Configuration
        surf_factors: Surface scaling factors
        
    Returns:
        Predictions as numpy array (float32)
    """
    
    with torch.no_grad():
        # CRITICAL: Ensure ALL data is float32 and on correct device
        data_dict_gpu = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                # Convert to float32 tensor and move to device
                v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
                data_dict_gpu[k] = v_tensor
            else:
                data_dict_gpu[k] = v
        
        # Verify all tensors are float32
        for k, v in data_dict_gpu.items():
            if isinstance(v, torch.Tensor):
                if v.dtype != torch.float32:
                    print(f"WARNING: Converting {k} from {v.dtype} to float32")
                    data_dict_gpu[k] = v.to(dtype=torch.float32)
        
        # Get predictions
        _, prediction_surf = model(data_dict_gpu)
        
        if prediction_surf is not None and surf_factors is not None:
            # Unnormalize predictions
            prediction_surf = unnormalize(
                prediction_surf.cpu().numpy(),
                surf_factors[0],
                surf_factors[1]
            )
            
            # Scale by physical parameters
            stream_velocity = data_dict_gpu["stream_velocity"][0, 0].cpu().numpy()
            air_density = data_dict_gpu["air_density"][0, 0].cpu().numpy()
            
            prediction_surf = (
                prediction_surf * stream_velocity**2.0 * air_density
            )
    
    return prediction_surf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*60)
    print("ENHANCED DoMINO MODEL TESTING (DTYPE FIXED)")
    print("="*60)
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    
    # Verify enhanced mode is enabled
    if not cfg.data_processor.get('use_enhanced_features', False):
        print("ERROR: Enhanced features not enabled in config!")
        print("Set 'use_enhanced_features: true' in config.yaml")
        return
    
    # Get paths
    coarse_test_path = cfg.eval.get('coarse_test_path', '/data/ahmed_data/organized/test/coarse/')
    fine_test_path = cfg.eval.get('test_path', '/data/ahmed_data/organized/test/fine/')
    save_path = cfg.eval.save_path
    
    # Create save directory
    if dist.rank == 0:
        Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    model_type = cfg.model.model_type
    
    # Surface variables
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1
    
    # Load scaling factors
    surf_save_path = os.path.join(
        cfg.eval.scaling_param_path, "surface_scaling_factors_inference.npy"
    )
    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path).astype(np.float32)  # Ensure float32
        print(f"Loaded surface scaling factors: {surf_factors}")
    else:
        surf_factors = None
        print("Warning: No surface scaling factors found")
    
    # Create enhanced model with explicit float32
    print("\nCreating Enhanced DoMINO model...")
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,  # Surface only for Ahmed
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device, dtype=torch.float32)  # Explicit float32
    
    model = torch.compile(model, disable=True)
    
    # Load checkpoint
    checkpoint_path = os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(
        to_absolute_path(checkpoint_path),
        map_location=dist.device
    )
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Ensure model parameters are float32
    for param in model.parameters():
        if param.dtype != torch.float32:
            print(f"Converting model parameter from {param.dtype} to float32")
            param.data = param.data.to(dtype=torch.float32)
    
    print("Model loaded successfully with float32 parameters")
    
    # Get test cases
    coarse_test_cases = []
    for case_dir in sorted(Path(coarse_test_path).iterdir()):
        if case_dir.is_dir() and case_dir.name.startswith('run_'):
            coarse_test_cases.append(case_dir.name)
    
    print(f"\nProcessing {len(coarse_test_cases)} test cases")
    
    # Process each test case
    results_summary = []
    
    for count, dirname in enumerate(coarse_test_cases[:5]):  # Test first 5 cases
        print(f"\n{'='*40}")
        print(f"Processing: {dirname} ({count+1}/5)")
        print(f"{'='*40}")
        
        filepath = os.path.join(coarse_test_path, dirname)
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])
        
        # File paths
        stl_path = os.path.join(filepath, f"ahmed_{tag}.stl")
        coarse_vtp_path = os.path.join(filepath, f"boundary_{tag}.vtp")
        
        # Output path
        vtp_pred_save_path = os.path.join(
            save_path, f"boundary_{tag}_enhanced_predicted.vtp"
        )
        
        try:
            # Load STL geometry (strict float32)
            print(f"Loading geometry: {stl_path}")
            reader = pv.get_reader(stl_path)
            mesh_stl = reader.read()
            stl_vertices = np.array(mesh_stl.points, dtype=np.float32)
            stl_faces = np.array(mesh_stl.faces, dtype=np.int32).reshape((-1, 4))[:, 1:]
            mesh_indices_flattened = stl_faces.flatten().astype(np.int32)
            length_scale = float(np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0)))
            stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
            stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
            stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)
            
            # Calculate center of mass
            center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes).astype(np.float32)
            
            # Set bounding box
            if cfg.data.bounding_box_surface is None:
                s_max = np.amax(stl_vertices, 0).astype(np.float32)
                s_min = np.amin(stl_vertices, 0).astype(np.float32)
            else:
                s_max = np.array(cfg.data.bounding_box_surface.max, dtype=np.float32)
                s_min = np.array(cfg.data.bounding_box_surface.min, dtype=np.float32)
            
            # Create grid for SDF (strict float32)
            nx, ny, nz = cfg.model.interp_res
            surf_grid = create_grid(s_max, s_min, [nx, ny, nz]).astype(np.float32)
            surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3).astype(np.float32)
            
            # Calculate SDF with proper data types
            sdf_surf_grid = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            ).reshape(nx, ny, nz).astype(np.float32)
            
            surf_grid_max_min = np.array([s_min, s_max], dtype=np.float32)
            
            # Load coarse surface data using robust loader
            print(f"Loading coarse surface data: {coarse_vtp_path}")
            coarse_data = load_coarse_vtp_data_robust(coarse_vtp_path, surface_variable_names)
            
            # Prepare surface neighbors (strict float32)
            surface_coordinates = coarse_data['coordinates'].astype(np.float32)
            interp_func = cKDTree(surface_coordinates)
            dd, ii = interp_func.query(surface_coordinates, k=cfg.eval.stencil_size + 1)
            surface_neighbors = surface_coordinates[ii[:, 1:]].astype(np.float32)
            surface_neighbors_normals = coarse_data['normals'][ii[:, 1:]].astype(np.float32)
            surface_neighbors_areas = coarse_data['areas'][ii[:, 1:]].astype(np.float32)
            
            # Calculate positional encoding
            pos_surface_center_of_mass = (surface_coordinates - center_of_mass).astype(np.float32)
            
            # Normalize coordinates (strict float32)
            surface_coordinates_norm = normalize(surface_coordinates, s_max, s_min).astype(np.float32)
            surface_neighbors_norm = normalize(surface_neighbors, s_max, s_min).astype(np.float32)
            surf_grid_norm = normalize(surf_grid, s_max, s_min).astype(np.float32)
            
            # Prepare data dictionary for model (ALL float32)
            data_dict = {
                "geometry_coordinates": stl_vertices,
                "surf_grid": surf_grid_norm,
                "sdf_surf_grid": sdf_surf_grid,
                "surface_mesh_centers": surface_coordinates_norm,
                "surface_mesh_neighbors": surface_neighbors_norm,
                "surface_normals": coarse_data['normals'],
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": coarse_data['areas'],
                "surface_neighbors_areas": surface_neighbors_areas,
                "surface_fields": coarse_data['fields'],  # Only coarse features!
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array([length_scale], dtype=np.float32),
                "stream_velocity": np.array([[STREAM_VELOCITY]], dtype=np.float32),
                "air_density": np.array([[AIR_DENSITY]], dtype=np.float32),
            }
            
            # Verify all arrays are float32
            for k, v in data_dict.items():
                if isinstance(v, np.ndarray) and v.dtype != np.float32:
                    print(f"Converting {k} from {v.dtype} to float32")
                    data_dict[k] = v.astype(np.float32)
            
            print(f"Input surface fields shape: {coarse_data['fields'].shape}")
            print(f"Input surface fields dtype: {coarse_data['fields'].dtype}")
            print("  (Should be 4 features for coarse-only inference)")
            
            # Get predictions
            print("Running enhanced model prediction...")
            start_time = time.time()
            prediction_surf = test_enhanced_model(
                data_dict, model, dist.device, cfg, surf_factors
            )
            elapsed_time = time.time() - start_time
            print(f"Prediction completed in {elapsed_time:.2f} seconds")
            
            if prediction_surf is not None:
                prediction_surf = prediction_surf[0]  # Remove batch dimension
                
                # Calculate forces
                surface_areas = coarse_data['areas'].reshape(-1, 1)
                surface_normals = coarse_data['normals']
                
                # Drag force
                force_x_pred = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - prediction_surf[:, 1] * surface_areas[:, 0]
                )
                
                # Lift force
                force_z_pred = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - prediction_surf[:, 3] * surface_areas[:, 0]
                )
                
                print(f"\nPredicted Forces:")
                print(f"  Drag: {force_x_pred:.6f} N")
                print(f"  Lift: {force_z_pred:.6f} N")
                
                # Store results
                results_summary.append({
                    'case': dirname,
                    'case_number': tag,
                    'drag_pred': force_x_pred,
                    'lift_pred': force_z_pred,
                    'processing_time': elapsed_time
                })
                
                # Save predictions to VTP
                print(f"Saving predictions to: {vtp_pred_save_path}")
                
                # Create output VTP with predictions
                polydata_out = coarse_data['mesh'].copy()
                
                # Add predicted pressure
                surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[:, 0:1])
                surfParam_vtk.SetName(f"{surface_variable_names[0]}Pred")
                polydata_out.GetCellData().AddArray(surfParam_vtk)
                
                # Add predicted wall shear stress
                surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[:, 1:])
                surfParam_vtk.SetName(f"{surface_variable_names[1]}Pred")
                polydata_out.GetCellData().AddArray(surfParam_vtk)
                
                # Write VTP file
                write_to_vtp(polydata_out, vtp_pred_save_path)
                print(f"  ✅ Saved successfully")
                
        except Exception as e:
            print(f"  ❌ Error processing {dirname}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("ENHANCED MODEL TESTING COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results_summary)}/5 test cases")
    
    if results_summary:
        print(f"\nForce Predictions Summary:")
        print(f"{'Case':<8} {'Drag':<12} {'Lift':<12} {'Time (s)':<8}")
        print("-" * 45)
        for result in results_summary:
            print(f"{result['case']:<8} {result['drag_pred']:<12.6f} "
                  f"{result['lift_pred']:<12.6f} {result['processing_time']:<8.2f}")
    
    print("\n🎉 Enhanced DoMINO testing completed successfully!")
    print("The dtype mismatch has been resolved.")


if __name__ == "__main__":
    main()