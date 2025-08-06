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
This code runs the data processing in parallel to load OpenFoam files, process them 
and save in the npy format for faster processing in the DoMINO datapipes. Several 
parameters such as number of processors, input and output paths, etc. can be 
configured in config.yaml in the data_processing tab.
"""

from openfoam_datapipe import OpenFoamDataset, OpenFoamDatasetEnhanced
from physicsnemo.utils.domino.utils import *
import multiprocessing
import hydra, time
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import traceback


def process_files(*args_list):
    ids = args_list[0]
    processor_id = args_list[1]
    fm_data = args_list[2]
    output_dir = args_list[3]
    use_enhanced = args_list[4]
    
    for j in ids:
        fname = fm_data.filenames[j]
        if len(os.listdir(os.path.join(fm_data.data_path, fname))) == 0:
            print(f"Skipping {fname} - empty.")
            continue
        outname = os.path.join(output_dir, fname)
        print("Filename:%s on processor: %d" % (outname, processor_id))
        filename = f"{outname}.npy"
        if os.path.exists(filename):
            print(f"Skipping {filename} - already exists.")
            continue
        start_time = time.time()
        
        try:
            data_dict = fm_data[j]
            
            # Validate enhanced features if enabled
            if use_enhanced and j == ids[0]:  # Only validate first sample per processor
                surface_fields_shape = data_dict['surface_fields'].shape
                expected_features = 8  # 4 fine + 4 coarse
                if surface_fields_shape[-1] != expected_features:
                    print(f"WARNING: Expected {expected_features} surface features, got {surface_fields_shape[-1]}")
                else:
                    print(f"✅ Enhanced features validated: surface_fields shape = {surface_fields_shape}")
                    
                    # Check for NaN values in interpolated data
                    if np.any(np.isnan(data_dict['surface_fields'])):
                        print("WARNING: NaN values detected in surface fields!")
                        nan_count = np.sum(np.isnan(data_dict['surface_fields']))
                        print(f"  Total NaN values: {nan_count}")
                    else:
                        print("✅ No NaN values in interpolated data")
            
            np.save(filename, data_dict)
            print("Time taken for %d = %f" % (j, time.time() - start_time))
            
        except Exception as e:
            print(f"ERROR processing file {fname}: {str(e)}")
            if use_enhanced:
                print("  This might be due to interpolation issues. Check coarse/fine mesh compatibility.")
            traceback.print_exc()
            continue


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    phase = "train"
    
    # Check if enhanced features are enabled
    use_enhanced_features = cfg.data_processor.get('use_enhanced_features', False)
    
    if use_enhanced_features:
        print("\n🚀 ENHANCED FEATURES ENABLED - Using dual-resolution dataset")
        print(f"  Fine data path: {cfg.data_processor.input_dir}")
        print(f"  Coarse data path: {cfg.data_processor.coarse_input_dir}")
    else:
        print("\n📦 Standard features - Using single-resolution dataset")
    
    # UPDATED: Handle surface-only datasets (like Ahmed) - check if volume variables exist
    volume_variable_names = []
    num_vol_vars = 0
    if hasattr(cfg.variables, 'volume') and cfg.variables.volume is not None:
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
        print(f"📦 Volume variables found: {volume_variable_names} (total: {num_vol_vars})")
    else:
        print("📦 No volume variables - surface-only dataset")

    # Surface variables should always exist
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1
    print(f"🎯 Surface variables found: {surface_variable_names} (total: {num_surf_vars})")
    
    # Create dataset based on configuration
    if use_enhanced_features:
        # Get coarse variable mapping from config
        coarse_variable_mapping = {}
        if hasattr(cfg.variables.surface, 'enhanced_features') and \
           hasattr(cfg.variables.surface.enhanced_features, 'coarse_variable_mapping'):
            coarse_variable_mapping = dict(cfg.variables.surface.enhanced_features.coarse_variable_mapping)
        else:
            # Default mapping for Ahmed dataset
            coarse_variable_mapping = {
                "pMean": "p",
                "wallShearStressMean": "wallShearStress"
            }
        
        print(f"  Variable mapping: {coarse_variable_mapping}")
        
        # Create enhanced dataset
        fm_data = OpenFoamDatasetEnhanced(
            data_path=cfg.data_processor.input_dir,
            coarse_data_path=cfg.data_processor.coarse_input_dir,
            kind=cfg.data_processor.kind,
            volume_variables=volume_variable_names,
            surface_variables=surface_variable_names,
            model_type=cfg.model.model_type,
            coarse_variable_mapping=coarse_variable_mapping,
        )
        
        print(f"  Expected surface features: {cfg.variables.surface.enhanced_features.input_feature_count}")
        
    else:
        # Create standard dataset
        fm_data = OpenFoamDataset(
            cfg.data_processor.input_dir,
            kind=cfg.data_processor.kind,
            volume_variables=volume_variable_names,
            surface_variables=surface_variable_names,
            model_type=cfg.model.model_type,
        )
        
        expected_surf_features = num_surf_vars
        print(f"  Expected surface features: {expected_surf_features}")
    
    output_dir = cfg.data_processor.output_dir
    create_directory(output_dir)
    n_processors = cfg.data_processor.num_processors

    num_files = len(fm_data)
    print(f"📊 Processing {num_files} files using {n_processors} processors")
    
    # Test on first sample before parallel processing
    if use_enhanced_features:
        print("\n🔍 Testing enhanced dataset on first sample...")
        try:
            test_sample = fm_data[0]
            print(f"  ✅ Test successful! Surface fields shape: {test_sample['surface_fields'].shape}")
            
            # Calculate interpolation quality metrics
            fine_fields = test_sample['surface_fields'][:, :4]
            coarse_interp = test_sample['surface_fields'][:, 4:]
            
            # Non-zero mask to avoid division issues
            non_zero_mask = np.abs(fine_fields) > 1e-10
            
            # Element-wise relative error where fine fields are non-zero
            rel_error = np.zeros_like(fine_fields)
            rel_error[non_zero_mask] = np.abs(fine_fields[non_zero_mask] - coarse_interp[non_zero_mask]) / np.abs(fine_fields[non_zero_mask])
            
            # Average relative error per field
            avg_rel_error = np.mean(rel_error, axis=0)
            
            print(f"\n  📊 Interpolation quality (average relative error):")
            print(f"    Pressure: {avg_rel_error[0]:.3%}")
            print(f"    Wall shear stress X: {avg_rel_error[1]:.3%}")
            print(f"    Wall shear stress Y: {avg_rel_error[2]:.3%}")
            print(f"    Wall shear stress Z: {avg_rel_error[3]:.3%}")
            print(f"    Overall: {avg_rel_error.mean():.3%}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {str(e)}")
            traceback.print_exc()
            return
    
    # Parallel processing
    ids = np.arange(num_files)
    num_elements = int(num_files / n_processors) + 1
    process_list = []
    ctx = multiprocessing.get_context("spawn")
    
    for i in range(n_processors):
        if i != n_processors - 1:
            sf = ids[i * num_elements : i * num_elements + num_elements]
        else:
            sf = ids[i * num_elements :]
        
        process = ctx.Process(
            target=process_files, 
            args=(sf, i, fm_data, output_dir, use_enhanced_features)
        )
        
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    print("\n✅ Data processing completed!")
    
    if use_enhanced_features:
        print("\n📝 Enhanced dataset summary:")
        print(f"  - Fine resolution data: {cfg.data_processor.input_dir}")
        print(f"  - Coarse resolution data: {cfg.data_processor.coarse_input_dir}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Surface features per sample: 8 (4 fine + 4 coarse interpolated)")
        print("\n  Next steps:")
        print("  1. Run training with enhanced model")
        print("  2. Monitor convergence and force coefficient predictions")
        print("  3. Compare with baseline DoMINO results")


if __name__ == "__main__":
    main()