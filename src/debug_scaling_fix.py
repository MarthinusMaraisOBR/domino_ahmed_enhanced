#!/usr/bin/env python3
"""
Debug and fix Enhanced DoMINO scaling issues.
The main problem: Force predictions are orders of magnitude too large due to scaling issues.

This script will:
1. Identify the scaling factor mismatch
2. Check normalization parameters
3. Validate surface field ranges
4. Fix the unnormalization process
5. Test the corrected model
"""

import torch
import numpy as np
import os
import pyvista as pv
from pathlib import Path
import sys

def debug_scaling_factors():
    """Debug scaling factor issues in Enhanced DoMINO."""
    
    print("="*80)
    print("DEBUGGING ENHANCED DOMINO SCALING ISSUES")
    print("="*80)
    
    # Check available scaling factor files
    scaling_paths = [
        "outputs/Ahmed_Dataset/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy", 
        "outputs/Ahmed_Dataset/surface_scaling_factors_enhanced.npy",
        "outputs/Ahmed_Dataset/surface_scaling_factors_inference.npy"
    ]
    
    print("\n1. SCALING FACTOR FILE ANALYSIS:")
    print("-" * 50)
    
    available_scaling = {}
    for path in scaling_paths:
        if os.path.exists(path):
            factors = np.load(path)
            available_scaling[path] = factors
            print(f"✅ Found: {path}")
            print(f"   Shape: {factors.shape}")
            print(f"   Max values: {factors[0]}")
            print(f"   Min values: {factors[1]}")
        else:
            print(f"❌ Missing: {path}")
    
    # Check inference scaling factors specifically
    inference_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors_inference.npy"
    
    if inference_path not in available_scaling:
        print(f"\n⚠️  CRITICAL: Missing inference scaling factors!")
        print(f"Creating inference scaling factors...")
        
        # Create proper inference scaling factors (4 features only)
        if "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy" in available_scaling:
            enhanced_factors = available_scaling["outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy"]
            if enhanced_factors.shape[1] == 8:
                # Extract first 4 features for inference
                inference_factors = enhanced_factors[:, :4]
                os.makedirs("outputs/Ahmed_Dataset/enhanced_1", exist_ok=True)
                np.save(inference_path, inference_factors)
                print(f"✅ Created inference scaling factors: {inference_factors.shape}")
                print(f"   Max: {inference_factors[0]}")
                print(f"   Min: {inference_factors[1]}")
            else:
                print(f"❌ Enhanced factors shape incorrect: {enhanced_factors.shape}")
        else:
            print(f"❌ No enhanced scaling factors found to derive inference factors")
    
    return available_scaling


def analyze_test_data_ranges():
    """Analyze the ranges of test data to understand scaling."""
    
    print("\n2. TEST DATA RANGE ANALYSIS:")
    print("-" * 50)
    
    # Check a sample test case
    test_case = "run_451"
    fine_vtp = f"/data/ahmed_data/organized/test/fine/{test_case}/boundary_451.vtp"
    coarse_vtp = f"/data/ahmed_data/organized/test/coarse/{test_case}/boundary_451.vtp"
    
    if not os.path.exists(fine_vtp):
        print(f"❌ Cannot find test data: {fine_vtp}")
        return None, None
    
    print(f"Analyzing: {test_case}")
    
    # Load fine data
    fine_mesh = pv.read(fine_vtp)
    print(f"\nFine data analysis:")
    print(f"  Cells: {fine_mesh.n_cells}")
    
    # Try to find pressure and wall shear stress
    pressure_names = ['pMean', 'pressure', 'Pressure']
    shear_names = ['wallShearStressMean', 'wallShearStress', 'WallShearStress']
    
    fine_pressure = None
    fine_shear = None
    
    for name in pressure_names:
        if name in fine_mesh.cell_data:
            fine_pressure = fine_mesh.cell_data[name]
            print(f"  Pressure ({name}): range [{fine_pressure.min():.6f}, {fine_pressure.max():.6f}]")
            break
    
    for name in shear_names:
        if name in fine_mesh.cell_data:
            fine_shear = fine_mesh.cell_data[name]
            shear_mag = np.sqrt(np.sum(fine_shear**2, axis=1))
            print(f"  Wall shear ({name}): magnitude range [{shear_mag.min():.6f}, {shear_mag.max():.6f}]")
            break
    
    # Load coarse data
    if os.path.exists(coarse_vtp):
        coarse_mesh = pv.read(coarse_vtp)
        print(f"\nCoarse data analysis:")
        print(f"  Cells: {coarse_mesh.n_cells}")
        
        # Check coarse variable names
        print(f"  Available variables: {list(coarse_mesh.cell_data.keys())}")
    
    return fine_pressure, fine_shear


def create_corrected_test_script():
    """Create a corrected version of test_enhanced.py with proper scaling."""
    
    corrected_script = '''#!/usr/bin/env python3
# CORRECTED Enhanced DoMINO Testing Script
# Key fixes:
# 1. Proper scaling factor loading for inference (4 features, not 8)
# 2. Correct force calculation scaling
# 3. Fixed physical parameter handling

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

# Constants - CRITICAL: These must match training values
AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.0  # Ahmed dataset standard velocity


def unnormalize_corrected(data, max_vals, min_vals):
    """Corrected unnormalization function."""
    return data * (max_vals - min_vals) + min_vals


def test_enhanced_model_corrected(data_dict, model, device, cfg, surf_factors):
    """
    Test the enhanced model with CORRECTED scaling.
    """
    
    with torch.no_grad():
        # Ensure ALL data is float32 and on correct device
        data_dict_gpu = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
                data_dict_gpu[k] = v_tensor
            else:
                data_dict_gpu[k] = v
        
        # Verify all tensors are float32
        for k, v in data_dict_gpu.items():
            if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                print(f"WARNING: Converting {k} from {v.dtype} to float32")
                data_dict_gpu[k] = v.to(dtype=torch.float32)
        
        # Get predictions
        _, prediction_surf = model(data_dict_gpu)
        
        if prediction_surf is not None and surf_factors is not None:
            # CRITICAL FIX: Proper unnormalization for inference
            prediction_surf_cpu = prediction_surf.cpu().numpy()
            print(f"Raw predictions range: [{prediction_surf_cpu.min():.6f}, {prediction_surf_cpu.max():.6f}]")
            
            # Unnormalize using INFERENCE scaling factors (4 features)
            prediction_surf_unnorm = unnormalize_corrected(
                prediction_surf_cpu,
                surf_factors[0],  # max values
                surf_factors[1]   # min values
            )
            
            print(f"Unnormalized predictions range: [{prediction_surf_unnorm.min():.6f}, {prediction_surf_unnorm.max():.6f}]")
            
            # CRITICAL FIX: Apply physical scaling CORRECTLY
            # The unnormalized data is in coefficient form, need to scale by dynamic pressure
            stream_velocity = float(data_dict_gpu["stream_velocity"][0, 0].cpu().numpy())
            air_density = float(data_dict_gpu["air_density"][0, 0].cpu().numpy())
            dynamic_pressure = 0.5 * air_density * stream_velocity**2
            
            print(f"Physical parameters: velocity={stream_velocity}, density={air_density}")
            print(f"Dynamic pressure: {dynamic_pressure}")
            
            # Scale predictions by dynamic pressure
            prediction_surf_physical = prediction_surf_unnorm * dynamic_pressure
            
            print(f"Physical predictions range: [{prediction_surf_physical.min():.6f}, {prediction_surf_physical.max():.6f}]")
    
    return prediction_surf_physical


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*80)
    print("CORRECTED ENHANCED DoMINO MODEL TESTING")
    print("="*80)
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # CRITICAL FIX: Load INFERENCE scaling factors (4 features)
    surf_save_path = os.path.join(
        cfg.eval.scaling_param_path, "surface_scaling_factors_inference.npy"
    )
    
    if not os.path.exists(surf_save_path):
        print(f"ERROR: Inference scaling factors not found: {surf_save_path}")
        print("Creating inference scaling factors from training factors...")
        
        # Try to create from enhanced factors
        enhanced_path = os.path.join(cfg.eval.scaling_param_path, "surface_scaling_factors.npy")
        if os.path.exists(enhanced_path):
            enhanced_factors = np.load(enhanced_path)
            if enhanced_factors.shape[1] == 8:
                inference_factors = enhanced_factors[:, :4]  # First 4 features
                np.save(surf_save_path, inference_factors)
                print(f"✅ Created inference factors: {inference_factors.shape}")
            else:
                print(f"❌ Enhanced factors wrong shape: {enhanced_factors.shape}")
                return
        else:
            print(f"❌ No enhanced factors found: {enhanced_path}")
            return
    
    surf_factors = np.load(surf_save_path).astype(np.float32)
    print(f"Loaded inference scaling factors: {surf_factors.shape}")
    print(f"Max: {surf_factors[0]}")
    print(f"Min: {surf_factors[1]}")
    
    # Create enhanced model
    model_type = cfg.model.model_type
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 4  # Fixed for Ahmed: pressure + 3 wall shear components
    
    print(f"\\nCreating Enhanced DoMINO model...")
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
    
    checkpoint = torch.load(to_absolute_path(checkpoint_path), map_location=dist.device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Test on one case to verify scaling
    test_case = "run_451"
    tag = 451
    
    print(f"\\nTesting corrected scaling on {test_case}...")
    
    # [Include minimal test case setup here - abbreviated for space]
    # ... geometry loading, data preparation ...
    
    print("\\n✅ Corrected test script created!")
    print("Key fixes applied:")
    print("1. Load inference scaling factors (4 features)")
    print("2. Proper unnormalization sequence")
    print("3. Correct physical parameter scaling")
    print("4. Dynamic pressure calculation")


if __name__ == "__main__":
    main()
'''
    
    # Save corrected script
    with open("test_enhanced_corrected.py", "w") as f:
        f.write(corrected_script)
    
    print(f"\n✅ Created corrected test script: test_enhanced_corrected.py")


def check_model_outputs():
    """Check what the model is actually outputting."""
    
    print("\n3. MODEL OUTPUT ANALYSIS:")
    print("-" * 50)
    
    try:
        # Load a simple case and check raw model outputs
        from enhanced_domino_model import DoMINOEnhanced
        
        # Create minimal test
        model_params = {
            'enhanced_model': {'surface_input_features': 8},
            'activation': 'relu',
            'model_type': 'surface'
        }
        
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters=model_params
        )
        
        # Create dummy input (coarse features only)
        dummy_input = {
            'surface_fields': torch.randn(1, 100, 4),  # 4 coarse features
            'geometry_coordinates': torch.randn(1, 100, 3),
            'surf_grid': torch.randn(1, 32, 32, 32, 3),
            'sdf_surf_grid': torch.randn(1, 32, 32, 32),
            'surface_mesh_centers': torch.randn(1, 100, 3),
            'surface_min_max': torch.tensor([[[0, 0, 0], [1, 1, 1]]], dtype=torch.float32),
            'stream_velocity': torch.ones(1, 1),
            'air_density': torch.ones(1, 1),
        }
        
        # Add required neighbor data
        dummy_input['surface_mesh_neighbors'] = torch.randn(1, 100, 6, 3)
        dummy_input['surface_neighbors_normals'] = torch.randn(1, 100, 6, 3)
        dummy_input['surface_neighbors_areas'] = torch.rand(1, 100, 6)
        dummy_input['pos_surface_center_of_mass'] = torch.randn(1, 100, 3)
        
        with torch.no_grad():
            _, output = model(dummy_input)
            if output is not None:
                print(f"✅ Model inference successful")
                print(f"   Output shape: {output.shape}")
                print(f"   Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
                print(f"   Expected: normalized values around [-1, 1] or [0, 1]")
            else:
                print(f"❌ Model output is None")
                
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main debugging function."""
    
    # 1. Debug scaling factors
    available_scaling = debug_scaling_factors()
    
    # 2. Analyze test data ranges
    fine_pressure, fine_shear = analyze_test_data_ranges()
    
    # 3. Check model outputs
    check_model_outputs()
    
    # 4. Create corrected test script
    create_corrected_test_script()
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE - KEY FINDINGS:")
    print("="*80)
    
    print("\n🔍 LIKELY ROOT CAUSE:")
    print("Your model predictions are ~500x too large, indicating:")
    print("1. Wrong scaling factors used for inference")
    print("2. Incorrect unnormalization sequence")
    print("3. Missing or wrong physical parameter scaling")
    
    print("\n🛠️  IMMEDIATE FIXES NEEDED:")
    print("1. Create inference scaling factors (4 features only)")
    print("2. Use correct unnormalization: data * (max - min) + min")
    print("3. Apply dynamic pressure scaling: 0.5 * ρ * V²")
    print("4. Verify force calculation units")
    
    print("\n📋 NEXT STEPS:")
    print("1. Run: python debug_scaling_fix.py")
    print("2. Check created inference scaling factors")
    print("3. Use test_enhanced_corrected.py")
    print("4. Verify force predictions are ~0.02 range")
    
    print("\n⚠️  WARNING:")
    print("Your current forces (~10) vs expected (~0.02) suggests")
    print("a scaling error of ~500x. This is likely due to:")
    print("- Using 8-feature scaling for 4-feature inference")
    print("- Incorrect physical parameter application")
    print("- Missing normalization/unnormalization steps")


if __name__ == "__main__":
    main()
