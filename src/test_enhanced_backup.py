#!/usr/bin/env python3
"""
FIXED Enhanced DoMINO model testing with VTP file saving.
This version actually saves the VTP files with all prediction fields.
"""

import os
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime

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
AIR_DENSITY = 1.0
STREAM_VELOCITY = 1.0


def write_to_vtp(polydata, filename):
    """Write polydata to VTP file."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()  # For debugging - can see values in text editor
    writer.Write()
    print(f"    ✓ VTP file written: {filename}")


def convert_point_to_cell_data_manual(mesh, field_name, n_components=1):
    """Manually convert point data to cell data using weighted averaging."""
    print(f"    Manual conversion: {field_name} ({n_components} components)")
    
    point_data = np.array(mesh.point_data[field_name], dtype=np.float32)
    n_cells = mesh.n_cells
    cell_data = np.zeros((n_cells, n_components), dtype=np.float32)
    
    for cell_id in range(n_cells):
        cell = mesh.GetCell(cell_id)
        point_ids = []
        for i in range(cell.GetNumberOfPoints()):
            point_ids.append(cell.GetPointId(i))
        
        if n_components == 1:
            cell_data[cell_id] = np.mean(point_data[point_ids])
        else:
            cell_data[cell_id] = np.mean(point_data[point_ids], axis=0)
    
    return cell_data


def interpolate_fine_to_coarse(fine_coords, fine_fields, coarse_coords, k_neighbors=4):
    """Interpolate fine mesh data to coarse mesh using nearest neighbor interpolation."""
    print(f"  Interpolating fine data ({fine_coords.shape[0]} cells) to coarse mesh ({coarse_coords.shape[0]} cells)")
    
    tree = cKDTree(fine_coords)
    distances, indices = tree.query(coarse_coords, k=k_neighbors)
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / distances
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    interpolated = np.zeros((coarse_coords.shape[0], fine_fields.shape[1]))
    for i in range(k_neighbors):
        interpolated += weights[:, i:i+1] * fine_fields[indices[:, i]]
        
    print(f"  ✅ Interpolation complete")
    return interpolated.astype(np.float32)


def load_coarse_vtp_data_robust(vtp_path, surface_variables):
    """Load coarse resolution VTP data with robust field extraction."""
    print(f"Loading coarse data: {vtp_path}")
    
    mesh = pv.read(str(vtp_path))
    print(f"  Coarse mesh: {mesh.n_cells} cells, {mesh.n_points} points")
    
    variable_mapping = {
        'pMean': ['p', 'pressure', 'Pressure'],
        'wallShearStressMean': ['wallShearStress', 'WallShearStress', 'tau_wall']
    }
    
    surface_fields = []
    field_info = []
    
    for var_name in surface_variables:
        field_found = False
        possible_names = variable_mapping.get(var_name, [var_name])
        
        for candidate_name in possible_names:
            if candidate_name in mesh.cell_data:
                field_data = np.array(mesh.cell_data[candidate_name], dtype=np.float32)
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                surface_fields.append(field_data)
                field_info.append(f"{var_name} -> {candidate_name} (cell data)")
                field_found = True
                break
            elif candidate_name in mesh.point_data:
                print(f"  Converting {candidate_name} from point data to cell data...")
                point_array = mesh.point_data[candidate_name]
                if hasattr(point_array, 'shape') and len(point_array.shape) > 1:
                    n_components = point_array.shape[1]
                elif 'pressure' in candidate_name.lower() or 'p' == candidate_name.lower():
                    n_components = 1
                else:
                    n_components = 3
                
                field_data = convert_point_to_cell_data_manual(mesh, candidate_name, n_components)
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                
                surface_fields.append(field_data.astype(np.float32))
                field_info.append(f"{var_name} -> {candidate_name} (point->cell manual)")
                field_found = True
                break
        
        if not field_found:
            print(f"  Warning: Variable {var_name} not found")
            num_cells = mesh.n_cells
            if 'pressure' in var_name.lower():
                surface_fields.append(np.zeros((num_cells, 1), dtype=np.float32))
            else:
                surface_fields.append(np.zeros((num_cells, 3), dtype=np.float32))
            field_info.append(f"{var_name} -> ZEROS (not found)")
    
    if surface_fields:
        surface_fields_combined = np.concatenate(surface_fields, axis=1).astype(np.float32)
    else:
        raise ValueError("No surface fields could be extracted!")
    
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)
    norms = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = (surface_normals / norms).astype(np.float32)
    
    print(f"  ✅ Loaded {len(surface_coordinates)} cells")
    
    return {
        'coordinates': surface_coordinates,
        'fields': surface_fields_combined,
        'normals': surface_normals,
        'areas': surface_areas,
        'mesh': mesh,
        'field_info': field_info
    }


def load_fine_vtp_data_and_interpolate(vtp_path, surface_variables, target_coordinates):
    """Load fine resolution VTP data and interpolate to target coordinates."""
    print(f"Loading fine ground truth data: {vtp_path}")
    
    mesh = pv.read(str(vtp_path))
    print(f"  Fine mesh: {mesh.n_cells} cells, {mesh.n_points} points")
    
    variable_mapping = {
        'pMean': ['pMean', 'pressure', 'Pressure'],
        'wallShearStressMean': ['wallShearStressMean', 'wallShearStress', 'WallShearStress']
    }
    
    surface_fields = []
    field_info = []
    
    for var_name in surface_variables:
        field_found = False
        possible_names = variable_mapping.get(var_name, [var_name])
        
        for candidate_name in possible_names:
            if candidate_name in mesh.cell_data:
                field_data = np.array(mesh.cell_data[candidate_name], dtype=np.float32)
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                surface_fields.append(field_data)
                field_info.append(f"{var_name} -> {candidate_name} (cell data)")
                field_found = True
                break
            elif candidate_name in mesh.point_data:
                print(f"  Converting {candidate_name} from point data to cell data...")
                point_array = mesh.point_data[candidate_name]
                if hasattr(point_array, 'shape') and len(point_array.shape) > 1:
                    n_components = point_array.shape[1]
                elif 'pressure' in candidate_name.lower():
                    n_components = 1
                else:
                    n_components = 3
                
                field_data = convert_point_to_cell_data_manual(mesh, candidate_name, n_components)
                if field_data.ndim == 1:
                    field_data = field_data.reshape(-1, 1)
                
                surface_fields.append(field_data.astype(np.float32))
                field_info.append(f"{var_name} -> {candidate_name} (point->cell manual)")
                field_found = True
                break
        
        if not field_found:
            print(f"  Warning: Variable {var_name} not found")
            num_cells = mesh.n_cells
            if 'pressure' in var_name.lower():
                surface_fields.append(np.zeros((num_cells, 1), dtype=np.float32))
            else:
                surface_fields.append(np.zeros((num_cells, 3), dtype=np.float32))
            field_info.append(f"{var_name} -> ZEROS (not found)")
    
    if surface_fields:
        surface_fields_combined = np.concatenate(surface_fields, axis=1).astype(np.float32)
    else:
        raise ValueError("No surface fields could be extracted!")
    
    fine_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    
    interpolated_fields = interpolate_fine_to_coarse(
        fine_coords=fine_coordinates,
        fine_fields=surface_fields_combined,
        coarse_coords=target_coordinates,
        k_neighbors=4
    )
    
    print(f"  ✅ Interpolated fine data to target mesh")
    
    return {
        'fields': interpolated_fields,
        'field_info': field_info,
        'original_coordinates': fine_coordinates,
        'original_fields': surface_fields_combined
    }


def test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """Test the enhanced model with STRICT float32 type enforcement."""
    with torch.no_grad():
        data_dict_gpu = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
                data_dict_gpu[k] = v_tensor
            else:
                data_dict_gpu[k] = v
        
        for k, v in data_dict_gpu.items():
            if isinstance(v, torch.Tensor):
                if v.dtype != torch.float32:
                    print(f"WARNING: Converting {k} from {v.dtype} to float32")
                    data_dict_gpu[k] = v.to(dtype=torch.float32)
        
        _, prediction_surf = model(data_dict_gpu)
        
        if prediction_surf is not None and surf_factors is not None:
            if surf_factors.shape[1] == 8 and prediction_surf.shape[-1] == 4:
                surf_factors_inference = surf_factors[:, :4]
                print(f"Adjusted scaling factors from shape {surf_factors.shape} to {surf_factors_inference.shape}")
            else:
                surf_factors_inference = surf_factors
            
            prediction_surf = unnormalize(
                prediction_surf.cpu().numpy(),
                surf_factors_inference[0],
                surf_factors_inference[1]
            )
            
            print("✅ Predictions generated (V=1, ρ=1)")
    
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
    """Save VTP file with coarse, fine (interpolated), and predicted fields."""
    print(f"\n📁 Saving comprehensive VTP: {output_path}")
    
    polydata_out = base_mesh.copy()
    
    n_cells = len(coarse_fields)
    assert len(fine_fields_interpolated) == n_cells, f"Size mismatch: fine {len(fine_fields_interpolated)} vs coarse {n_cells}"
    assert len(predicted_fields) == n_cells, f"Size mismatch: predicted {len(predicted_fields)} vs coarse {n_cells}"
    
    print(f"  All field arrays have {n_cells} cells ✅")
    
    # Add timestamp info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_array = vtk.vtkStringArray()
    timestamp_array.SetName("ProcessingTimestamp")
    timestamp_array.InsertNextValue(timestamp)
    polydata_out.GetFieldData().AddArray(timestamp_array)
    
    # Add coarse fields
    coarse_pressure = coarse_fields[:, 0:1]
    coarse_wall_shear = coarse_fields[:, 1:4]
    
    coarse_pressure_vtk = numpy_support.numpy_to_vtk(coarse_pressure)
    coarse_pressure_vtk.SetName("Coarse_Pressure")
    polydata_out.GetCellData().AddArray(coarse_pressure_vtk)
    
    coarse_shear_vtk = numpy_support.numpy_to_vtk(coarse_wall_shear)
    coarse_shear_vtk.SetName("Coarse_WallShearStress")
    polydata_out.GetCellData().AddArray(coarse_shear_vtk)
    
    # Add fine fields
    fine_pressure = fine_fields_interpolated[:, 0:1]
    fine_wall_shear = fine_fields_interpolated[:, 1:4]
    
    fine_pressure_vtk = numpy_support.numpy_to_vtk(fine_pressure)
    fine_pressure_vtk.SetName("Fine_Pressure_GroundTruth_Interpolated")
    polydata_out.GetCellData().AddArray(fine_pressure_vtk)
    
    fine_shear_vtk = numpy_support.numpy_to_vtk(fine_wall_shear)
    fine_shear_vtk.SetName("Fine_WallShearStress_GroundTruth_Interpolated")
    polydata_out.GetCellData().AddArray(fine_shear_vtk)
    
    # Add predicted fields
    pred_pressure = predicted_fields[:, 0:1]
    pred_wall_shear = predicted_fields[:, 1:4]
    
    pred_pressure_vtk = numpy_support.numpy_to_vtk(pred_pressure)
    pred_pressure_vtk.SetName("Predicted_Pressure")
    polydata_out.GetCellData().AddArray(pred_pressure_vtk)
    
    pred_shear_vtk = numpy_support.numpy_to_vtk(pred_wall_shear)
    pred_shear_vtk.SetName("Predicted_WallShearStress")
    polydata_out.GetCellData().AddArray(pred_shear_vtk)
    
    # Calculate error fields
    pressure_error = pred_pressure - fine_pressure
    shear_error = pred_wall_shear - fine_wall_shear
    
    pressure_error_vtk = numpy_support.numpy_to_vtk(pressure_error)
    pressure_error_vtk.SetName("Error_Pressure_vs_Fine")
    polydata_out.GetCellData().AddArray(pressure_error_vtk)
    
    shear_error_vtk = numpy_support.numpy_to_vtk(shear_error)
    shear_error_vtk.SetName("Error_WallShearStress_vs_Fine")
    polydata_out.GetCellData().AddArray(shear_error_vtk)
    
    # Calculate absolute error
    pressure_error_abs = np.abs(pressure_error)
    shear_error_abs = np.linalg.norm(shear_error, axis=1, keepdims=True)
    
    pressure_error_abs_vtk = numpy_support.numpy_to_vtk(pressure_error_abs)
    pressure_error_abs_vtk.SetName("Error_Pressure_Absolute")
    polydata_out.GetCellData().AddArray(pressure_error_abs_vtk)
    
    shear_error_abs_vtk = numpy_support.numpy_to_vtk(shear_error_abs)
    shear_error_abs_vtk.SetName("Error_WallShearStress_Magnitude")
    polydata_out.GetCellData().AddArray(shear_error_abs_vtk)
    
    # Calculate baseline errors (coarse vs fine)
    coarse_pressure_error = coarse_pressure - fine_pressure
    coarse_shear_error = coarse_wall_shear - fine_wall_shear
    
    coarse_pressure_error_vtk = numpy_support.numpy_to_vtk(coarse_pressure_error)
    coarse_pressure_error_vtk.SetName("Error_Coarse_vs_Fine")
    polydata_out.GetCellData().AddArray(coarse_pressure_error_vtk)
    
    coarse_shear_error_vtk = numpy_support.numpy_to_vtk(coarse_shear_error)
    coarse_shear_error_vtk.SetName("Error_CoarseShear_vs_Fine")
    polydata_out.GetCellData().AddArray(coarse_shear_error_vtk)
    
    # Calculate relative errors
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
    
    # Print statistics
    print(f"  ✅ Saved VTP with {polydata_out.GetCellData().GetNumberOfArrays()} field arrays:")
    for i in range(polydata_out.GetCellData().GetNumberOfArrays()):
        array_name = polydata_out.GetCellData().GetArrayName(i)
        print(f"      - {array_name}")
    
    # Print field statistics for verification
    print(f"\n  📊 Field Statistics:")
    print(f"    Coarse Pressure: mean={coarse_pressure.mean():.4f}, std={coarse_pressure.std():.4f}")
    print(f"    Fine Pressure:   mean={fine_pressure.mean():.4f}, std={fine_pressure.std():.4f}")
    print(f"    Predicted Pressure: mean={pred_pressure.mean():.4f}, std={pred_pressure.std():.4f}")
    print(f"    Pressure Error: mean={pressure_error.mean():.4f}, max={np.abs(pressure_error).max():.4f}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*80)
    print("ENHANCED DoMINO MODEL TESTING - WITH VTP FILE GENERATION")
    print("="*80)
    print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    DistributedManager.initialize()
    dist = DistributedManager()
    
    if not cfg.data_processor.get('use_enhanced_features', False):
        print("ERROR: Enhanced features not enabled in config!")
        return
    
    # Get paths - create dated output directory
    coarse_test_path = cfg.eval.get('coarse_test_path', '/data/ahmed_data/organized/test/coarse/')
    fine_test_path = cfg.eval.get('test_path', '/data/ahmed_data/organized/test/fine/')
    
    # Create output directory with date
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(cfg.eval.save_path, f"predictions_{date_str}")
    
    if dist.rank == 0:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {save_path}")
    
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
        surf_factors = np.load(surf_save_path).astype(np.float32)
        print(f"Loaded scaling factors: shape {surf_factors.shape}")
    else:
        surf_factors = None
        print("Warning: No scaling factors found")
    
    # Create model
    print("\nCreating Enhanced DoMINO model...")
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
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
    
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)
    
    print("Model loaded successfully")
    
    # Get test cases
    coarse_test_cases = []
    for case_dir in sorted(Path(coarse_test_path).iterdir()):
        if case_dir.is_dir() and case_dir.name.startswith('run_'):
            coarse_test_cases.append(case_dir.name)
    
    num_cases = min(5, len(coarse_test_cases))  # Process up to 5 cases
    print(f"\nProcessing {num_cases} test cases")
    
    results_summary = []
    
    for count, dirname in enumerate(coarse_test_cases[:num_cases]):
        print(f"\n{'='*60}")
        print(f"Processing: {dirname} ({count+1}/{num_cases})")
        print(f"{'='*60}")
        
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])
        
        # File paths
        coarse_filepath = os.path.join(coarse_test_path, dirname)
        fine_filepath = os.path.join(fine_test_path, dirname)
        
        stl_path = os.path.join(coarse_filepath, f"ahmed_{tag}.stl")
        coarse_vtp_path = os.path.join(coarse_filepath, f"boundary_{tag}.vtp")
        fine_vtp_path = os.path.join(fine_filepath, f"boundary_{tag}.vtp")
        
        # Output path
        vtp_output_path = os.path.join(save_path, f"run_{tag}_prediction.vtp")
        
        try:
            # Load STL geometry
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
            
            center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes).astype(np.float32)
            
            if cfg.data.bounding_box_surface is None:
                s_max = np.amax(stl_vertices, 0).astype(np.float32)
                s_min = np.amin(stl_vertices, 0).astype(np.float32)
            else:
                s_max = np.array(cfg.data.bounding_box_surface.max, dtype=np.float32)
                s_min = np.array(cfg.data.bounding_box_surface.min, dtype=np.float32)
            
            nx, ny, nz = cfg.model.interp_res
            surf_grid = create_grid(s_max, s_min, [nx, ny, nz]).astype(np.float32)
            surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3).astype(np.float32)
            
            sdf_surf_grid = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            ).reshape(nx, ny, nz).astype(np.float32)
            
            surf_grid_max_min = np.array([s_min, s_max], dtype=np.float32)
            
            # Load coarse data
            coarse_data = load_coarse_vtp_data_robust(coarse_vtp_path, surface_variable_names)
            
            # Load and interpolate fine data
            fine_data = load_fine_vtp_data_and_interpolate(
                fine_vtp_path, 
                surface_variable_names, 
                coarse_data['coordinates']
            )
            
            reference_mesh = coarse_data['mesh']
            surface_coordinates = coarse_data['coordinates'].astype(np.float32)
            
            interp_func = cKDTree(surface_coordinates)
            dd, ii = interp_func.query(surface_coordinates, k=cfg.eval.stencil_size + 1)
            surface_neighbors = surface_coordinates[ii[:, 1:]].astype(np.float32)
            surface_neighbors_normals = coarse_data['normals'][ii[:, 1:]].astype(np.float32)
            surface_neighbors_areas = coarse_data['areas'][ii[:, 1:]].astype(np.float32)
            
            pos_surface_center_of_mass = (surface_coordinates - center_of_mass).astype(np.float32)
            
            surface_coordinates_norm = normalize(surface_coordinates, s_max, s_min).astype(np.float32)
            surface_neighbors_norm = normalize(surface_neighbors, s_max, s_min).astype(np.float32)
            surf_grid_norm = normalize(surf_grid, s_max, s_min).astype(np.float32)
            
            # Prepare data for model
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
                "surface_fields": coarse_data['fields'],
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array([length_scale], dtype=np.float32),
                "stream_velocity": np.array([[STREAM_VELOCITY]], dtype=np.float32),
                "air_density": np.array([[AIR_DENSITY]], dtype=np.float32),
            }
            
            for k, v in data_dict.items():
                if isinstance(v, np.ndarray) and v.dtype != np.float32:
                    data_dict[k] = v.astype(np.float32)
            
            # Get predictions
            print("Running model prediction...")
            start_time = time.time()
            prediction_surf = test_enhanced_model(
                data_dict, model, dist.device, cfg, surf_factors
            )
            elapsed_time = time.time() - start_time
            print(f"Prediction completed in {elapsed_time:.2f} seconds")
            
            if prediction_surf is not None:
                prediction_surf = prediction_surf[0] if isinstance(prediction_surf, np.ndarray) else prediction_surf[0].cpu().numpy()
                
                # CRITICAL: Save VTP file!
                save_comprehensive_vtp(
                    output_path=vtp_output_path,
                    base_mesh=reference_mesh,
                    coarse_fields=coarse_data['fields'],
                    fine_fields_interpolated=fine_data['fields'],
                    predicted_fields=prediction_surf,
                    surface_variable_names=surface_variable_names,
                    coarse_info=coarse_data['field_info'],
                    fine_info=fine_data['field_info']
                )
                
                # Calculate coefficients for summary
                surface_areas = coarse_data['areas'].reshape(-1, 1)
                surface_normals = coarse_data['normals']
                
                print(f"\nIntegrated Coefficient Comparison:")
                
                # Drag coefficient
                Cdx_coarse = np.sum(
                    coarse_data['fields'][:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - coarse_data['fields'][:, 1] * surface_areas[:, 0]
                )
                Cdx_fine = np.sum(
                    fine_data['fields'][:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - fine_data['fields'][:, 1] * surface_areas[:, 0]
                )
                Cdx_pred = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 0] * surface_areas[:, 0]
                    - prediction_surf[:, 1] * surface_areas[:, 0]
                )
                
                coarse_to_fine_drag_error = abs(Cdx_coarse - Cdx_fine)
                pred_to_fine_drag_error = abs(Cdx_pred - Cdx_fine)
                if coarse_to_fine_drag_error > 1e-10:
                    drag_improvement = (coarse_to_fine_drag_error - pred_to_fine_drag_error) / coarse_to_fine_drag_error * 100
                else:
                    drag_improvement = 0.0
                
                # Lift coefficient
                Clz_coarse = np.sum(
                    coarse_data['fields'][:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - coarse_data['fields'][:, 3] * surface_areas[:, 0]
                )
                Clz_fine = np.sum(
                    fine_data['fields'][:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - fine_data['fields'][:, 3] * surface_areas[:, 0]
                )
                Clz_pred = np.sum(
                    prediction_surf[:, 0] * surface_normals[:, 2] * surface_areas[:, 0]
                    - prediction_surf[:, 3] * surface_areas[:, 0]
                )
                
                coarse_to_fine_lift_error = abs(Clz_coarse - Clz_fine)
                pred_to_fine_lift_error = abs(Clz_pred - Clz_fine)
                if coarse_to_fine_lift_error > 1e-10:
                    lift_improvement = (coarse_to_fine_lift_error - pred_to_fine_lift_error) / coarse_to_fine_lift_error * 100
                else:
                    lift_improvement = 0.0
                
                print(f"  Drag: Coarse={Cdx_coarse:.6f}, Fine={Cdx_fine:.6f}, Pred={Cdx_pred:.6f}, Improvement={drag_improvement:.1f}%")
                print(f"  Lift: Coarse={Clz_coarse:.6f}, Fine={Clz_fine:.6f}, Pred={Clz_pred:.6f}, Improvement={lift_improvement:.1f}%")
                
                results_summary.append({
                    'case': dirname,
                    'drag_improvement': drag_improvement,
                    'lift_improvement': lift_improvement,
                    'vtp_file': vtp_output_path
                })
                
        except Exception as e:
            print(f"  ❌ Error processing {dirname}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(results_summary)}/{num_cases} test cases")
    
    if results_summary:
        avg_drag_improvement = np.mean([r['drag_improvement'] for r in results_summary])
        avg_lift_improvement = np.mean([r['lift_improvement'] for r in results_summary])
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Average drag improvement: {avg_drag_improvement:.1f}%")
        print(f"  Average lift improvement: {avg_lift_improvement:.1f}%")
        
        print(f"\n✅ VTP FILES SAVED TO: {save_path}")
        print("\n📁 Generated VTP files:")
        for result in results_summary:
            print(f"  - {os.path.basename(result['vtp_file'])}")
        
        print(f"\n💡 To visualize in ParaView:")
        print(f"  1. Open ParaView")
        print(f"  2. File -> Open -> Navigate to: {save_path}")
        print(f"  3. Select all VTP files")
        print(f"  4. Color by: Predicted_Pressure, Fine_Pressure_GroundTruth_Interpolated, etc.")
        print(f"  5. Compare Error_Pressure_vs_Fine to see where model differs from ground truth")


if __name__ == "__main__":
    main()