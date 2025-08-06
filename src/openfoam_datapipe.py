# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds 
in npy format. 

"""

import time, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset

# Add these imports at the top of the file
from scipy.spatial import cKDTree
from typing import Optional

AIR_DENSITY = 1.205
STREAM_VELOCITY = 1


class DriveSimPaths:
    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / "body.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / "VTK/simpleFoam_steady_3000/internal.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / "VTK/simpleFoam_steady_3000/boundary/aero_suv.vtp"


class DrivAerAwsPaths:
    @staticmethod
    def _get_index(car_dir: Path) -> str:
        return car_dir.name.removeprefix("run_")

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / f"drivaer_{DrivAerAwsPaths._get_index(car_dir)}.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / f"volume_{DrivAerAwsPaths._get_index(car_dir)}.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / f"boundary_{DrivAerAwsPaths._get_index(car_dir)}.vtp"


# ADDED: Ahmed dataset paths
class AhmedPaths:
    @staticmethod
    def _get_index(car_dir: Path) -> str:
        return car_dir.name.removeprefix("run_")

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / f"ahmed_{AhmedPaths._get_index(car_dir)}.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / f"volume_{AhmedPaths._get_index(car_dir)}.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / f"boundary_{AhmedPaths._get_index(car_dir)}.vtp"


class OpenFoamDataset(Dataset):
    """
    Datapipe for converting openfoam dataset to npy

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        kind: Literal["drivesim", "drivaer_aws", "ahmed"] = "drivesim",  # UPDATED: Added "ahmed"
        surface_variables: Optional[list] = [
            "pMean",
            "wallShearStress",
        ],
        volume_variables: Optional[list] = ["UMean", "pMean"],
        device: int = 0,
        model_type=None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path

        supported_kinds = ["drivesim", "drivaer_aws", "ahmed"]  # UPDATED: Added "ahmed"
        assert (
            kind in supported_kinds
        ), f"kind should be one of {supported_kinds}, got {kind}"
        
        # UPDATED: Added Ahmed path getter
        if kind == "drivesim":
            self.path_getter = DriveSimPaths
        elif kind == "drivaer_aws":
            self.path_getter = DrivAerAwsPaths
        else:  # ahmed
            self.path_getter = AhmedPaths

        assert self.data_path.exists(), f"Path {self.data_path} does not exist"

        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"

        self.filenames = get_filenames(self.data_path)
        random.shuffle(self.filenames)
        self.indices = np.array(len(self.filenames))

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        car_dir = self.data_path / cfd_filename

        stl_path = self.path_getter.geometry_path(car_dir)
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        if self.model_type == "volume" or self.model_type == "combined":
            filepath = self.path_getter.volume_path(car_dir)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()

            # Get the unstructured grid data
            polydata = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata, self.volume_variables
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)

            # Non-dimensionalize volume fields
            volume_fields[:, :3] = volume_fields[:, :3] / STREAM_VELOCITY
            volume_fields[:, 3:4] = volume_fields[:, 3:4] / (
                AIR_DENSITY * STREAM_VELOCITY**2.0
            )

            volume_fields[:, 4:] = volume_fields[:, 4:] / (
                STREAM_VELOCITY * length_scale
            )
        else:
            volume_fields = None
            volume_coordinates = None

        if self.model_type == "surface" or self.model_type == "combined":
            surface_filepath = self.path_getter.surface_path(car_dir)
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(surface_filepath)
            reader.Update()
            polydata = reader.GetOutput()

            celldata_all = get_node_to_elem(polydata)
            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, self.surface_variables)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata)
            surface_coordinates = np.array(mesh.cell_centers().points)

            surface_normals = np.array(mesh.cell_normals)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"])

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )

            # Non-dimensionalize surface fields
            surface_fields = surface_fields / (AIR_DENSITY * STREAM_VELOCITY**2.0)
        else:
            surface_fields = None
            surface_coordinates = None
            surface_normals = None
            surface_sizes = None

        # Add the parameters to the dictionary
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": np.float32(surface_coordinates),
            "surface_normals": np.float32(surface_normals),
            "surface_areas": np.float32(surface_sizes),
            "volume_fields": np.float32(volume_fields),
            "volume_mesh_centers": np.float32(volume_coordinates),
            "surface_fields": np.float32(surface_fields),
            "filename": cfd_filename,
            "stream_velocity": STREAM_VELOCITY,
            "air_density": AIR_DENSITY,
        }

# Add this class after the existing OpenFoamDataset class
class OpenFoamDatasetEnhanced(Dataset):
    """
    Enhanced dataset that loads both fine and coarse resolution data
    and interpolates coarse data to fine mesh for dual-resolution training.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        coarse_data_path: Union[str, Path],
        kind: Literal["drivesim", "drivaer_aws", "ahmed"] = "ahmed",
        surface_variables: Optional[list] = ["pMean", "wallShearStressMean"],
        volume_variables: Optional[list] = ["UMean", "pMean"],
        device: int = 0,
        model_type=None,
        coarse_variable_mapping: Optional[dict] = None,
    ):
        # Initialize parent class attributes
        super().__init__()
        
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(coarse_data_path, str):
            coarse_data_path = Path(coarse_data_path)
            
        data_path = data_path.expanduser()
        coarse_data_path = coarse_data_path.expanduser()
        
        self.data_path = data_path
        self.coarse_data_path = coarse_data_path
        
        supported_kinds = ["drivesim", "drivaer_aws", "ahmed"]
        assert kind in supported_kinds, f"kind should be one of {supported_kinds}, got {kind}"
        
        if kind == "drivesim":
            self.path_getter = DriveSimPaths
        elif kind == "drivaer_aws":
            self.path_getter = DrivAerAwsPaths
        else:  # ahmed
            self.path_getter = AhmedPaths
            
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.coarse_data_path.exists(), f"Coarse path {self.coarse_data_path} does not exist"
        
        self.filenames = get_filenames(self.data_path)
        random.shuffle(self.filenames)
        self.indices = np.array(len(self.filenames))
        
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.device = device
        self.model_type = model_type
        
        # Default coarse variable mapping for Ahmed dataset
        if coarse_variable_mapping is None:
            self.coarse_variable_mapping = {
                "pMean": "p",
                "wallShearStressMean": "wallShearStress"
            }
        else:
            self.coarse_variable_mapping = coarse_variable_mapping
            
    def __len__(self):
        return len(self.filenames)
    
    def _interpolate_coarse_to_fine(
        self, 
        coarse_coords: np.ndarray, 
        coarse_fields: np.ndarray,
        fine_coords: np.ndarray,
        k_neighbors: int = 4
    ) -> np.ndarray:
        """
        Interpolate coarse mesh data to fine mesh using inverse distance weighting.
        
        Args:
            coarse_coords: Coarse mesh coordinates (M, 3)
            coarse_fields: Coarse mesh field values (M, n_fields)
            fine_coords: Fine mesh coordinates (N, 3)
            k_neighbors: Number of nearest neighbors for interpolation
            
        Returns:
            Interpolated field values on fine mesh (N, n_fields)
        """
        # Build KDTree for coarse mesh
        tree = cKDTree(coarse_coords)
        
        # Find k nearest neighbors for each fine mesh point
        distances, indices = tree.query(fine_coords, k=k_neighbors)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Compute inverse distance weights
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Interpolate fields
        interpolated = np.zeros((fine_coords.shape[0], coarse_fields.shape[1]))
        for i in range(k_neighbors):
            interpolated += weights[:, i:i+1] * coarse_fields[indices[:, i]]
            
        return interpolated
    
    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        
        # Get fine resolution data (same as original)
        car_dir = self.data_path / cfd_filename
        
        # Load STL
        stl_path = self.path_getter.geometry_path(car_dir)
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)
        
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        
        # Volume data handling (if needed)
        if self.model_type == "volume" or self.model_type == "combined":
            filepath = self.path_getter.volume_path(car_dir)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            polydata = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata, self.volume_variables
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)
            
            # Non-dimensionalize volume fields
            volume_fields[:, :3] = volume_fields[:, :3] / STREAM_VELOCITY
            volume_fields[:, 3:4] = volume_fields[:, 3:4] / (
                AIR_DENSITY * STREAM_VELOCITY**2.0
            )
            volume_fields[:, 4:] = volume_fields[:, 4:] / (
                STREAM_VELOCITY * length_scale
            )
        else:
            volume_fields = None
            volume_coordinates = None
            
        # Fine surface data
        if self.model_type == "surface" or self.model_type == "combined":
            # Fine surface data
            surface_filepath = self.path_getter.surface_path(car_dir)
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(surface_filepath)
            reader.Update()
            polydata = reader.GetOutput()
            
            celldata_all = get_node_to_elem(polydata)
            celldata = celldata_all.GetCellData()
            surface_fields_fine = get_fields(celldata, self.surface_variables)
            surface_fields_fine = np.concatenate(surface_fields_fine, axis=-1)
            
            mesh = pv.PolyData(polydata)
            surface_coordinates = np.array(mesh.cell_centers().points)
            
            surface_normals = np.array(mesh.cell_normals)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"])
            
            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )
            
            # Non-dimensionalize fine surface fields
            surface_fields_fine = surface_fields_fine / (AIR_DENSITY * STREAM_VELOCITY**2.0)
            
            # Load coarse surface data
            coarse_car_dir = self.coarse_data_path / cfd_filename
            coarse_surface_filepath = self.path_getter.surface_path(coarse_car_dir)
            
            reader_coarse = vtk.vtkXMLPolyDataReader()
            reader_coarse.SetFileName(coarse_surface_filepath)
            reader_coarse.Update()
            polydata_coarse = reader_coarse.GetOutput()
            
            # Handle point data to cell data conversion for coarse mesh
            celldata_coarse_all = get_node_to_elem(polydata_coarse)
            celldata_coarse = celldata_coarse_all.GetCellData()
            
            # Map coarse variable names and extract fields
            coarse_surface_vars = []
            for var in self.surface_variables:
                coarse_var_name = self.coarse_variable_mapping.get(var, var)
                coarse_surface_vars.append(coarse_var_name)
            
            surface_fields_coarse = get_fields(celldata_coarse, coarse_surface_vars)
            surface_fields_coarse = np.concatenate(surface_fields_coarse, axis=-1)
            
            # Get coarse mesh coordinates
            mesh_coarse = pv.PolyData(polydata_coarse)
            surface_coordinates_coarse = np.array(mesh_coarse.cell_centers().points)
            
            # Interpolate coarse fields to fine mesh
            surface_fields_coarse_interp = self._interpolate_coarse_to_fine(
                surface_coordinates_coarse,
                surface_fields_coarse,
                surface_coordinates,
                k_neighbors=4
            )
            
            # Non-dimensionalize interpolated coarse fields
            surface_fields_coarse_interp = surface_fields_coarse_interp / (
                AIR_DENSITY * STREAM_VELOCITY**2.0
            )
            
            # Concatenate fine and interpolated coarse fields
            surface_fields_enhanced = np.concatenate(
                [surface_fields_fine, surface_fields_coarse_interp], 
                axis=-1
            )
            
            print(f"Enhanced surface fields shape: {surface_fields_enhanced.shape}")
            
        else:
            surface_fields_enhanced = None
            surface_coordinates = None
            surface_normals = None
            surface_sizes = None
            
        # Return enhanced data dictionary
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": np.float32(surface_coordinates),
            "surface_normals": np.float32(surface_normals),
            "surface_areas": np.float32(surface_sizes),
            "volume_fields": np.float32(volume_fields),
            "volume_mesh_centers": np.float32(volume_coordinates),
            "surface_fields": np.float32(surface_fields_enhanced),  # Now contains 8 features
            "filename": cfd_filename,
            "stream_velocity": STREAM_VELOCITY,
            "air_density": AIR_DENSITY,
        }



# Update the main section to test enhanced dataset
if __name__ == "__main__":
    # Test standard dataset
    print("Testing standard dataset...")
    fm_data = OpenFoamDataset(
        data_path="/data/ahmed_data/raw/",
        kind="ahmed",
        surface_variables=["pMean", "wallShearStressMean"],
        model_type="surface"
    )
    d_dict = fm_data[0]
    print(f"Standard surface fields shape: {d_dict['surface_fields'].shape}")
    
    # Test enhanced dataset
    print("\nTesting enhanced dataset...")
    fm_data_enhanced = OpenFoamDatasetEnhanced(
        data_path="/data/ahmed_data/raw/",
        coarse_data_path="/data/ahmed_data_rans/raw/",
        kind="ahmed",
        surface_variables=["pMean", "wallShearStressMean"],
        model_type="surface",
        coarse_variable_mapping={
            "pMean": "p",
            "wallShearStressMean": "wallShearStress"
        }
    )
    d_dict_enhanced = fm_data_enhanced[0]
    print(f"Enhanced surface fields shape: {d_dict_enhanced['surface_fields'].shape}")
    
    # Validate interpolation quality
    fine_fields = d_dict_enhanced['surface_fields'][:, :4]
    coarse_interp = d_dict_enhanced['surface_fields'][:, 4:]
    
    # Calculate relative RMSE
    rmse = np.sqrt(np.mean((fine_fields - coarse_interp)**2, axis=0))
    rel_rmse = rmse / np.sqrt(np.mean(fine_fields**2, axis=0))
    print(f"\nInterpolation quality (relative RMSE):")
    print(f"  Pressure: {rel_rmse[0]:.3%}")
    print(f"  Wall shear stress: {rel_rmse[1:].mean():.3%}")