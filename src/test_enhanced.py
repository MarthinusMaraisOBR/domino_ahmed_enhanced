# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Testing script for Enhanced DoMINO model.
This script loads coarse RANS data and predicts fine resolution surface fields.
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

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.sdf import signed_distance_field

# Import the enhanced model
from enhanced_domino_model import DoMINOEnhanced

# Constants
AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.0


def load_coarse_vtp_data(vtp_path, surface_variables):
    """Load coarse resolution surface data from VTP file"""
    print(f"Loading coarse data: {vtp_path}")
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    polydata = reader.GetOutput()
    
    # Get cell data
    celldata = polydata.GetCellData()
    
    # Extract fields with variable name mapping
    surface_fields = []
    for var_name in surface_variables:
        # Try different possible names for coarse data
        coarse_names = [var_name, var_name.replace('Mean', ''), 'p', 'wallShearStress']
        
        field_found = False
        for coarse_name in coarse_names:
            if celldata.GetArray(coarse_name):
                array = celldata.GetArray(coarse_name)
                field_data = np.zeros((array.GetNumberOfTuples(), array.GetNumberOfComponents()))
                for i in range(array.GetNumberOfTuples()):
                    for j in range(array.GetNumberOfComponents()):
                        field_data[i, j] = array.GetComponent(i, j)
                surface_fields.append(field_data)
                print(f"  Found {var_name} as {coarse_name}")
                field_found = True
                break
        
        if not field_found:
            print(f"  Warning: {var_name} not found, using zeros")
            num_cells = polydata.GetNumberOfCells()
            surface_fields.append(np.zeros((num_cells, 1)))
    
    surface_fields = np.concatenate(surface_fields, axis=-1)
    
    # Get mesh data
    mesh = pv.PolyData(polydata)
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)
    
    # Normalize normals
    surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
    
    return {
        'coordinates': surface_coordinates,
        'fields': surface_fields,
        'normals': surface_normals,
        'areas': surface_areas,
        'polydata': polydata
    }


def test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """Test the enhanced model with coarse input data"""
    
    with torch.no_grad():
        # Move data to device
        data_dict = {k: torch.from_numpy(v).unsqueeze(0).to(device) 
                     for k, v in data_dict.items()}
        
        # Get predictions
        _, prediction_surf = model(data_dict)
        
        if prediction_surf is not None and surf_factors is not None:
            # Unnormalize predictions
            prediction_surf = unnormalize(
                prediction_surf.cpu().numpy(),
                surf_factors[0],
                surf_factors[1]
            )
            
            # Scale by physical parameters
            stream_velocity = data_dict["stream_velocity"][0, 0].cpu().numpy()
            air_density = data_dict["air_density"][0, 0].cpu().numpy()
            
            prediction_surf = (
                prediction_surf * stream_velocity**2.0 * air_density
            )
    
    return prediction_surf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*60)
    print("ENHANCED DoMINO MODEL TESTING")
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
    coarse_test_path = cfg.eval.get('coarse_test_path', '/data/ahmed_data_rans/raw_test/')
    fine_test_path = cfg.eval.test_path  # For comparison if available
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
        cfg.eval.scaling_param_path, "surface_scaling_factors.npy"
    )
    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
        print(f"Loaded surface scaling factors: {surf_factors}")
    else:
        surf_factors = None
        print("Warning: No surface scaling factors found")
    
    # Create enhanced model
    print("\nCreating Enhanced DoMINO model...")
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,  # Surface only for Ahmed
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)
    
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
    print("Model loaded successfully")
    
    # Get test cases
    dirnames = get_filenames(coarse_test_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / dist.world_size) if dist.world_size > 0 else len(dirnames)
    dirnames_per_gpu = dirnames[int(num_files * dev_id):int(num_files * (dev_id + 1))]
    
    print(f"\nProcessing {len(dirnames_per_gpu)} test cases on GPU {dev_id}")
    
    # Process each test case
    for count, dirname in enumerate(dirnames_per_gpu):
        print(f"\n{'='*40}")
        print(f"Processing: {dirname}")
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
        
        # Load STL geometry
        print(f"Loading geometry: {stl_path}")
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)
        
        # Calculate center of mass
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)
        
        # Set bounding box
        if cfg.data.bounding_box_surface is None:
            s_max = np.amax(stl_vertices, 0)
            s_min = np.amin(stl_vertices, 0)
        else:
            s_max = np.asarray(cfg.data.bounding_box_surface.max)
            s_min = np.asarray(cfg.data.bounding_box_surface.min)
        
        # Create grid for SDF
        nx, ny, nz = cfg.model.interp_res
        surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)
        
        # Calculate SDF
        sdf_surf_grid = signed_distance_field(
            stl_vertices,
            mesh_indices_flattened,
            surf_grid_reshaped,
            use_sign_winding_number=True,
        ).reshape(nx, ny, nz)
        
        surf_grid = np.float32(surf_grid)
        sdf_surf_grid = np.float32(sdf_surf_grid)
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))
        
        # Load coarse surface data
        print(f"Loading coarse surface data: {coarse_vtp_path}")
        coarse_data = load_coarse_vtp_data(coarse_vtp_path, surface_variable_names)
        
        # Prepare surface neighbors (for model)
        from scipy.spatial import KDTree
        surface_coordinates = coarse_data['coordinates']
        interp_func = KDTree(surface_coordinates)
        dd, ii = interp_func.query(surface_coordinates, k=cfg.eval.stencil_size + 1)
        surface_neighbors = surface_coordinates[ii[:, 1:]]
        surface_neighbors_normals = coarse_data['normals'][ii[:, 1:]]
        surface_neighbors_areas = coarse_data['areas'][ii[:, 1:]]
        
        # Calculate positional encoding
        pos_surface_center_of_mass = surface_coordinates - center_of_mass
        
        # Normalize coordinates
        surface_coordinates_norm = normalize(surface_coordinates, s_max, s_min)
        surface_neighbors_norm = normalize(surface_neighbors, s_max, s_min)
        surf_grid_norm = normalize(surf_grid, s_max, s_min)
        
        # Prepare data dictionary for model
        # IMPORTANT: For inference, we only provide coarse features (4 features)
        data_dict = {
            "geometry_coordinates": np.float32(stl_vertices),
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
            "length_scale": np.array(length_scale, dtype=np.float32),
            "stream_velocity": np.array([[STREAM_VELOCITY]], dtype=np.float32),
            "air_density": np.array([[AIR_DENSITY]], dtype=np.float32),
        }
        
        print(f"Input surface fields shape: {coarse_data['fields'].shape}")
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
            print(f"  Drag: {force_x_pred:.3f} N")
            print(f"  Lift: {force_z_pred:.3f} N")
            
            # Save predictions to VTP
            print(f"Saving predictions to: {vtp_pred_save_path}")
            
            # Create output VTP with predictions
            polydata_out = coarse_data['polydata']
            
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
            print(f"  Saved successfully")
            
            # If fine data is available, compare
            fine_vtp_path = os.path.join(fine_test_path, dirname, f"boundary_{tag}.vtp")
            if os.path.exists(fine_vtp_path):
                print(f"\nComparing with fine resolution data...")
                # Load fine data for comparison
                fine_mesh = pv.read(fine_vtp_path)
                # ... (comparison code if needed)
    
    print(f"\n{'='*60}")
    print("ENHANCED MODEL TESTING COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {save_path}")
    print("\nTo visualize results:")
    print("1. Open ParaView")
    print("2. Load the predicted VTP files")
    print("3. Color by 'pMeanPred' for pressure")
    print("4. Color by 'wallShearStressMeanPred' for wall shear stress")


if __name__ == "__main__":
    main()
