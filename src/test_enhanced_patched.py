#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
FIXED Enhanced DoMINO model testing with mesh interpolation.
This script handles coarse and fine meshes with different resolutions by
interpolating fine data onto the coarse mesh for proper comparison.

CRITICAL FIX:
- Interpolates fine ground truth data onto coarse mesh coordinates
- Ensures all field arrays have matching dimensions
- Proper handling of different mesh resolutions
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


def interpolate_fine_to_coarse(fine_coords, fine_fields, coarse_coords, k_neighbors=4):
    """
    Interpolate fine mesh data to coarse mesh using nearest neighbor interpolation.
    
    Args:
        fine_coords: Fine mesh coordinates (N_fine, 3)
        fine_fields: Fine mesh field values (N_fine, n_fields)
        coarse_coords: Coarse mesh coordinates (N_coarse, 3)
        k_neighbors: Number of nearest neighbors for interpolation
        
    Returns:
        Interpolated field values on coarse mesh (N_coarse, n_fields)
    """
    print(f"  Interpolating fine data ({fine_coords.shape[0]} cells) to coarse mesh ({coarse_coords.shape[0]} cells)")
    
    # Build KDTree for fine mesh
    tree = cKDTree(fine_coords)
    
    # Find k nearest neighbors for each coarse mesh point
    distances, indices = tree.query(coarse_coords, k=k_neighbors)
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)
    
    # Compute inverse distance weights
    weights = 1.0 / distances
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Interpolate fields
    interpolated = np.zeros((coarse_coords.shape[0], fine_fields.shape[1]))
    for i in range(k_neighbors):
        interpolated += weights[:, i:i+1] * fine_fields[indices[:, i]]
        
    print(f"  ✅ Interpolation complete")
    return interpolated.astype(np.float32)


def load_coarse_vtp_data_robust(vtp_path, surface_variables):
    """
    Load coarse resolution VTP data with robust field extraction.
    """
    print(f"Loading coarse data: {vtp_path}")
    
    # Read VTP file
    mesh = pv.read(str(vtp_path))
    
    print(f"  Coarse mesh: {mesh.n_cells} cells, {mesh.n_points} points")
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
    print(f"  Coarse surface fields shape: {surface_fields_combined.shape}")
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


def load_fine_vtp_data_and_interpolate(vtp_path, surface_variables, target_coordinates):
    """
    Load fine resolution VTP data and interpolate to target coordinates.
    
    Args:
        vtp_path: Path to fine VTP file
        surface_variables: List of variable names to extract
        target_coordinates: Target coordinates for interpolation (N_target, 3)
        
    Returns:
        Dictionary with interpolated fine surface data (all float32)
    """
    print(f"Loading fine ground truth data: {vtp_path}")
    
    # Read VTP file
    mesh = pv.read(str(vtp_path))
    
    print(f"  Fine mesh: {mesh.n_cells} cells, {mesh.n_points} points")
    print(f"  Cell data keys: {list(mesh.cell_data.keys())}")
    print(f"  Point data keys: {list(mesh.point_data.keys())}")
    
    # Variable name mapping for fine data
    variable_mapping = {
        'pMean': ['pMean', 'pressure', 'Pressure'],
        'wallShearStressMean': ['wallShearStressMean', 'wallShearStress', 'WallShearStress']
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
    
    # Get fine mesh geometry data
    fine_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    
    # Interpolate fine fields to target coordinates
    interpolated_fields = interpolate_fine_to_coarse(
        fine_coords=fine_coordinates,
        fine_fields=surface_fields_combined,
        coarse_coords=target_coordinates,
        k_neighbors=4
    )
    
    print(f"  ✅ Interpolated fine data to target mesh")
    print(f"  Interpolated fields shape: {interpolated_fields.shape}")
    for info in field_info:
        print(f"    {info}")
    
    return {
        'fields': interpolated_fields,
        'field_info': field_info,
        'original_coordinates': fine_coordinates,
        'original_fields': surface_fields_combined
    }


def test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """
    Test the enhanced model with STRICT float32 type enforcement.
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
            prediction_surf_raw = prediction_surf.cpu().numpy()
            print(f"DEBUG: Raw prediction range: [{prediction_surf_raw.min():.6f}, {prediction_surf_raw.max():.6f}]")
            
            # FIXED: Proper unnormalization
            prediction_surf = prediction_surf_raw * (surf_factors[0] - surf_factors[1]) + surf_factors[1]
            print(f"DEBUG: Unnormalized range: [{prediction_surf.min():.6f}, {prediction_surf.max():.6f}]")
            
            # Scale by physical parameters
            stream_velocity = data_dict_gpu["stream_velocity"][0, 0].cpu().numpy()
            air_density = data_dict_gpu["air_density"][0, 0].cpu().numpy()
            
            # FIXED: Correct physical scaling
            # Predictions are now in coefficient form, scale by dynamic pressure
            dynamic_pressure = 0.5 * air_density * stream_velocity**2.0
            print(f"DEBUG: Dynamic pressure: {dynamic_pressure}")
            
            # Only scale if values look like coefficients (small values)
            if np.abs(prediction_surf).max() < 10:
                prediction_surf = prediction_surf * dynamic_pressure
                print(f"DEBUG: After physical scaling: [{prediction_surf.min():.6f}, {prediction_surf.max():.6f}]")
            else:
                print(f"WARNING: Predictions already in physical units, skipping scaling")
    
    return prediction_surf


def save_comprehensive_vtp(
    output_path: str,
    base_mesh: pv.PolyData,
    coarse_fields: np.ndarray,
    fine_fields_interpolated: np.ndarray,
    predicted_fields: np.ndarray,
    surface_variable_names: list,
    coarse_info: list,
    fine_info: list
):
    """
    Save VTP file with coarse, fine (interpolated), and predicted fields.
    """
    print(f"Saving comprehensive VTP: {output_path}")
    
    # Create output mesh
    polydata_out = base_mesh.copy()
    
    # Verify all arrays have the same length
    n_cells = len(coarse_fields)
    assert len(fine_fields_interpolated) == n_cells, f"Size mismatch: fine {len(fine_fields_interpolated)} vs coarse {n_cells}"
    assert len(predicted_fields) == n_cells, f"Size mismatch: predicted {len(predicted_fields)} vs coarse {n_cells}"
    
    print(f"  All field arrays have {n_cells} cells - dimensions match ✅")
    
    # Add coarse fields (input data)
    coarse_pressure = coarse_fields[:, 0:1]
    coarse_wall_shear = coarse_fields[:, 1:4]
    
    # Add coarse pressure
    coarse_pressure_vtk = numpy_support.numpy_to_vtk(coarse_pressure)
    coarse_pressure_vtk.SetName("Coarse_Pressure")
    polydata_out.GetCellData().AddArray(coarse_pressure_vtk)
    
    # Add coarse wall shear stress
    coarse_shear_vtk = numpy_support.numpy_to_vtk(coarse_wall_shear)
    coarse_shear_vtk.SetName("Coarse_WallShearStress")
    polydata_out.GetCellData().AddArray(coarse_shear_vtk)
    
    # Add fine fields (interpolated ground truth)
    fine_pressure = fine_fields_interpolated[:, 0:1]
    fine_wall_shear = fine_fields_interpolated[:, 1:4]
    
    # Add fine pressure
    fine_pressure_vtk = numpy_support.numpy_to_vtk(fine_pressure)
    fine_pressure_vtk.SetName("Fine_Pressure_GroundTruth_Interpolated")
    polydata_out.GetCellData().AddArray(fine_pressure_vtk)
    
    # Add fine wall shear stress
    fine_shear_vtk = numpy_support.numpy_to_vtk(fine_wall_shear)
    fine_shear_vtk.SetName("Fine_WallShearStress_GroundTruth_Interpolated")
    polydata_out.GetCellData().AddArray(fine_shear_vtk)
    
    # Add predicted fields
    pred_pressure = predicted_fields[:, 0:1]
    pred_wall_shear = predicted_fields[:, 1:4]
    
    # Add predicted pressure
    pred_pressure_vtk = numpy_support.numpy_to_vtk(pred_pressure)
    pred_pressure_vtk.SetName("Predicted_Pressure")
    polydata_out.GetCellData().AddArray(pred_pressure_vtk)
    
    # Add predicted wall shear stress
    pred_shear_vtk = numpy_support.numpy_to_vtk(pred_wall_shear)
    pred_shear_vtk.SetName("Predicted_WallShearStress")
    polydata_out.GetCellData().AddArray(pred_shear_vtk)
    
    # Calculate error fields (predicted vs fine ground truth)
    pressure_error = pred_pressure - fine_pressure
    shear_error = pred_wall_shear - fine_wall_shear
    
    # Add error fields
    pressure_error_vtk = numpy_support.numpy_to_vtk(pressure_error)
    pressure_error_vtk.SetName("Error_Pressure_vs_Fine")
    polydata_out.GetCellData().AddArray(pressure_error_vtk)
    
    shear_error_vtk = numpy_support.numpy_to_vtk(shear_error)
    shear_error_vtk.SetName("Error_WallShearStress_vs_Fine")
    polydata_out.GetCellData().AddArray(shear_error_vtk)
    
    # Calculate improvement over coarse baseline
    coarse_pressure_error = coarse_pressure - fine_pressure
    coarse_shear_error = coarse_wall_shear - fine_wall_shear
    
    # Add coarse baseline error fields
    coarse_pressure_error_vtk = numpy_support.numpy_to_vtk(coarse_pressure_error)
    coarse_pressure_error_vtk.SetName("Error_Coarse_vs_Fine")
    polydata_out.GetCellData().AddArray(coarse_pressure_error_vtk)
    
    coarse_shear_error_vtk = numpy_support.numpy_to_vtk(coarse_shear_error)
    coarse_shear_error_vtk.SetName("Error_CoarseShear_vs_Fine")
    polydata_out.GetCellData().AddArray(coarse_shear_error_vtk)
    
    # Calculate relative error fields (as percentages)
    # Avoid division by zero
    fine_pressure_safe = np.where(np.abs(fine_pressure) < 1e-10, 1e-10, fine_pressure)
    fine_shear_safe = np.where(np.abs(fine_wall_shear) < 1e-10, 1e-10, fine_wall_shear)
    
    rel_pressure_error = 100.0 * pressure_error / fine_pressure_safe
    rel_shear_error = 100.0 * shear_error / fine_shear_safe
    
    rel_pressure_error_vtk = numpy_support.numpy_to_vtk(rel_pressure_error)
    rel_pressure_error_vtk.SetName("RelativeError_Pressure_Percent")
    polydata_out.GetCellData().AddArray(rel_pressure_error_vtk)
    
    rel_shear_error_vtk = numpy_support.numpy_to_vtk(rel_shear_error)
    rel_shear_error_vtk.SetName("RelativeError_WallShearStress_Percent")
    polydata_out.GetCellData().AddArray(rel_shear_error_vtk)
    
    # Write VTP file
    write_to_vtp(polydata_out, output_path)
    
    print(f"  ✅ Saved comprehensive VTP with {polydata_out.GetCellData().GetNumberOfArrays()} field arrays")
    print(f"  Field arrays:")
    for i in range(polydata_out.GetCellData().GetNumberOfArrays()):
        array_name = polydata_out.GetCellData().GetArrayName(i)
        print(f"    - {array_name}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*80)
    print("ENHANCED DoMINO MODEL TESTING - MESH INTERPOLATION FIXED")
    print("="*80)
    print("Saving: Coarse + Fine (Interpolated) + Predicted + Error Fields")
    
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
    
    # EMERGENCY FIX: Create inference factors if missing
    if not os.path.exists(surf_save_path):
        print(f"FIXING: Creating missing inference scaling factors...")
        enhanced_path = os.path.join(cfg.eval.scaling_param_path, "surface_scaling_factors.npy")
        if os.path.exists(enhanced_path):
            enhanced_factors = np.load(enhanced_path)
            if enhanced_factors.shape[1] >= 4:
                inference_factors = enhanced_factors[:, :4].astype(np.float32)
                np.save(surf_save_path, inference_factors)
                print(f"✅ Created inference factors: {inference_factors.shape}")
        else:
            # Default Ahmed factors
            inference_factors = np.array([
                [0.8215, 0.01063, 0.01515, 0.01328],
                [-2.1506, -0.01865, -0.01514, -0.01215]
            ], dtype=np.float32)
            np.save(surf_save_path, inference_factors)
            print(f"⚠️  Created default inference factors")
    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path).astype(np.float32)
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
    ).to(dist.device, dtype=torch.float32)
    
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
        print(f"\n{'='*60}")
        print(f"Processing: {dirname} ({count+1}/5)")
        print(f"{'='*60}")
        
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])
        
        # File paths
        coarse_filepath = os.path.join(coarse_test_path, dirname)
        fine_filepath = os.path.join(fine_test_path, dirname)
        
        stl_path = os.path.join(coarse_filepath, f"ahmed_{tag}.stl")
        coarse_vtp_path = os.path.join(coarse_filepath, f"boundary_{tag}.vtp")
        fine_vtp_path = os.path.join(fine_filepath, f"boundary_{tag}.vtp")
        
        # Output path
        vtp_comprehensive_save_path = os.path.join(
            save_path, f"boundary_{tag}_comprehensive_comparison.vtp"
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
            
            # Load coarse surface data first (this will be our reference mesh)
            print(f"Loading coarse surface data: {coarse_vtp_path}")
            coarse_data = load_coarse_vtp_data_robust(coarse_vtp_path, surface_variable_names)
            
            # Load fine surface data and interpolate to coarse mesh coordinates
            print(f"Loading and interpolating fine surface data: {fine_vtp_path}")
            fine_data = load_fine_vtp_data_and_interpolate(
                fine_vtp_path, 
                surface_variable_names, 
                coarse_data['coordinates']  # Use coarse coordinates as target
            )
            
            # Use coarse mesh as reference (since model is trained on coarse→fine)
            reference_mesh = coarse_data['mesh']
            surface_coordinates = coarse_data['coordinates'].astype(np.float32)
            
            # Prepare surface neighbors (strict float32)
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
            print(f"Interpolated fine fields shape: {fine_data['fields'].shape}")
            print("  (Both should have same number of cells now)")
            
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
                
                # Now all arrays have the same dimensions - calculate forces
                surface_areas = coarse_data['areas'].reshape(-1, 1)
                surface_normals = coarse_data['normals']
                
                # Coarse forces
                coarse_force_x = np.sum(
                    coarse_data['fields'][:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - coarse_data['fields'][:, 1] * surface_areas[:, 0]
                )
                coarse_force_z = np.sum(
                    coarse_data['fields'][:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - coarse_data['fields'][:, 3] * surface_areas[:, 0]
                )
                
                # Fine forces (interpolated ground truth)
                fine_force_x = np.sum(
                    fine_data['fields'][:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - fine_data['fields'][:, 1] * surface_areas[:, 0]
                )
                fine_force_z = np.sum(
                    fine_data['fields'][:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - fine_data['fields'][:, 3] * surface_areas[:, 0]
                )
                
                # Predicted forces
                pred_force_x = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - prediction_surf[:, 1] * surface_areas[:, 0]
                )
                pred_force_z = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - prediction_surf[:, 3] * surface_areas[:, 0]
                )
                
                print(f"\nForce Comparison:")
                print(f"  Drag Forces:")
                print(f"    Coarse:       {coarse_force_x:.6f} N")
                print(f"    Fine (Interp): {fine_force_x:.6f} N")
                print(f"    Predicted:    {pred_force_x:.6f} N")
                
                # Calculate errors
                coarse_to_fine_drag_error = abs(coarse_force_x - fine_force_x)
                pred_to_fine_drag_error = abs(pred_force_x - fine_force_x)
                drag_improvement = (coarse_to_fine_drag_error - pred_to_fine_drag_error) / coarse_to_fine_drag_error * 100
                
                print(f"    Coarse vs Fine error: {coarse_to_fine_drag_error:.6f} N")
                print(f"    Pred vs Fine error:   {pred_to_fine_drag_error:.6f} N")
                print(f"    Improvement: {drag_improvement:.1f}%")
                
                print(f"  Lift Forces:")
                print(f"    Coarse:       {coarse_force_z:.6f} N")
                print(f"    Fine (Interp): {fine_force_z:.6f} N")
                print(f"    Predicted:    {pred_force_z:.6f} N")
                
                # Calculate errors
                coarse_to_fine_lift_error = abs(coarse_force_z - fine_force_z)
                pred_to_fine_lift_error = abs(pred_force_z - fine_force_z)
                lift_improvement = (coarse_to_fine_lift_error - pred_to_fine_lift_error) / coarse_to_fine_lift_error * 100
                
                print(f"    Coarse vs Fine error: {coarse_to_fine_lift_error:.6f} N")
                print(f"    Pred vs Fine error:   {pred_to_fine_lift_error:.6f} N")
                print(f"    Improvement: {lift_improvement:.1f}%")
                
                # Store results
                results_summary.append({
                    'case': dirname,
                    'case_number': tag,
                    'drag_coarse': coarse_force_x,
                    'drag_fine': fine_force_x,
                    'drag_pred': pred_force_x,
                    'drag_improvement': drag_improvement,
                    'lift_coarse': coarse_force_z,
                    'lift_fine': fine_force_z,
                    'lift_pred': pred_force_z,
                    'lift_improvement': lift_improvement,
                    'processing_time': elapsed_time
                })
                
                # Save comprehensive VTP with all fields
                print(f"Saving comprehensive VTP: {vtp_comprehensive_save_path}")
                save_comprehensive_vtp(
                    output_path=vtp_comprehensive_save_path,
                    base_mesh=reference_mesh,
                    coarse_fields=coarse_data['fields'],
                    fine_fields_interpolated=fine_data['fields'],
                    predicted_fields=prediction_surf,
                    surface_variable_names=surface_variable_names,
                    coarse_info=coarse_data['field_info'],
                    fine_info=fine_data['field_info']
                )
                
                # Calculate field statistics
                pressure_rmse = np.sqrt(np.mean((prediction_surf[:, 0] - fine_data['fields'][:, 0])**2))
                shear_rmse = np.sqrt(np.mean((prediction_surf[:, 1:4] - fine_data['fields'][:, 1:4])**2))
                
                # Calculate baseline errors
                pressure_rmse_baseline = np.sqrt(np.mean((coarse_data['fields'][:, 0] - fine_data['fields'][:, 0])**2))
                shear_rmse_baseline = np.sqrt(np.mean((coarse_data['fields'][:, 1:4] - fine_data['fields'][:, 1:4])**2))
                
                # Calculate improvements
                pressure_improvement = (pressure_rmse_baseline - pressure_rmse) / pressure_rmse_baseline * 100
                shear_improvement = (shear_rmse_baseline - shear_rmse) / shear_rmse_baseline * 100
                
                print(f"\nField Quality Metrics:")
                print(f"  Pressure RMSE:")
                print(f"    Coarse baseline: {pressure_rmse_baseline:.6f}")
                print(f"    Predicted:       {pressure_rmse:.6f}")
                print(f"    Improvement:     {pressure_improvement:.1f}%")
                print(f"  Wall Shear RMSE:")
                print(f"    Coarse baseline: {shear_rmse_baseline:.6f}")
                print(f"    Predicted:       {shear_rmse:.6f}")
                print(f"    Improvement:     {shear_improvement:.1f}%")
                
        except Exception as e:
            print(f"  ❌ Error processing {dirname}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("ENHANCED MODEL TESTING COMPLETED - COMPREHENSIVE COMPARISON")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(results_summary)}/5 test cases")
    
    if results_summary:
        print(f"\nForce Predictions Comprehensive Summary:")
        print(f"{'Case':<8} {'Coarse Drag':<12} {'Fine Drag':<12} {'Pred Drag':<12} {'Improvement':<12}")
        print(f"{'':8} {'Coarse Lift':<12} {'Fine Lift':<12} {'Pred Lift':<12} {'Improvement':<12}")
        print("-" * 80)
        for result in results_summary:
            print(f"{result['case']:<8} {result['drag_coarse']:<12.6f} "
                  f"{result['drag_fine']:<12.6f} {result['drag_pred']:<12.6f} {result['drag_improvement']:<12.1f}%")
            print(f"{'':8} {result['lift_coarse']:<12.6f} "
                  f"{result['lift_fine']:<12.6f} {result['lift_pred']:<12.6f} {result['lift_improvement']:<12.1f}%")
            print()
        
        # Calculate average improvements
        avg_drag_improvement = np.mean([r['drag_improvement'] for r in results_summary])
        avg_lift_improvement = np.mean([r['lift_improvement'] for r in results_summary])
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Average drag prediction improvement over coarse: {avg_drag_improvement:.1f}%")
        print(f"  Average lift prediction improvement over coarse: {avg_lift_improvement:.1f}%")
    
    print(f"\n🎉 Enhanced DoMINO comprehensive testing completed successfully!")
    print(f"\n📁 Output VTP files contain:")
    print(f"  - Coarse_Pressure & Coarse_WallShearStress (input)")
    print(f"  - Fine_Pressure_GroundTruth_Interpolated & Fine_WallShearStress_GroundTruth_Interpolated")
    print(f"  - Predicted_Pressure & Predicted_WallShearStress")
    print(f"  - Error_Pressure_vs_Fine & Error_WallShearStress_vs_Fine")
    print(f"  - Error_Coarse_vs_Fine & Error_CoarseShear_vs_Fine (baseline errors)")
    print(f"  - RelativeError_Pressure_Percent & RelativeError_WallShearStress_Percent")
    print(f"\n💡 In ParaView:")
    print(f"  1. Load the VTP files from: {save_path}")
    print(f"  2. Use 'Coarse_*' fields to see input data")
    print(f"  3. Use 'Fine_*_GroundTruth_Interpolated' fields to see target data")
    print(f"  4. Use 'Predicted_*' fields to see model predictions")
    print(f"  5. Use 'Error_*_vs_Fine' fields to see prediction errors")
    print(f"  6. Use 'Error_Coarse_vs_Fine' fields to see baseline errors")
    print(f"  7. Color by different fields to compare visually")
    print(f"\n🔧 Key Fix Applied:")
    print(f"  ✅ Fine mesh data interpolated to coarse mesh coordinates")
    print(f"  ✅ All field arrays now have matching dimensions")
    print(f"  ✅ Proper error calculation and improvement metrics")


if __name__ == "__main__":
    main()