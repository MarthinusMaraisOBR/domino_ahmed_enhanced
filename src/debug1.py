#!/usr/bin/env python3
"""
Debug and fix Enhanced DoMINO model predictions.

CRITICAL ISSUES IDENTIFIED:
1. Scaling factor mismatch between training and inference
2. Wrong surface scaling factors being used
3. Possible normalization issues in the pipeline

FIXES APPLIED:
- Use correct scaling factors for inference
- Debug normalization pipeline step by step
- Add validation checks for reasonable prediction ranges
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


def debug_scaling_factors():
    """Debug and find correct scaling factors."""
    print("🔍 DEBUGGING SCALING FACTORS")
    print("="*50)
    
    # Check all possible scaling factor files
    possible_paths = [
        "outputs/Ahmed_Dataset/surface_scaling_factors.npy",
        "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy", 
        "outputs/Ahmed_Dataset/1/surface_scaling_factors.npy",
        "surface_scaling_factors.npy",
        "/data/ahmed_data/surface_scaling_factors.npy"
    ]
    
    found_factors = {}
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                factors = np.load(path)
                found_factors[path] = factors
                print(f"✅ Found: {path}")
                print(f"   Shape: {factors.shape}")
                print(f"   Max values: {factors[0] if len(factors) > 0 else 'N/A'}")
                print(f"   Min values: {factors[1] if len(factors) > 1 else 'N/A'}")
                print()
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
    
    if not found_factors:
        print("❌ No scaling factors found!")
        return None
    
    # Use the most appropriate scaling factors
    # For enhanced model, prefer the enhanced version
    if "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy" in found_factors:
        chosen_path = "outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy"
    elif "outputs/Ahmed_Dataset/surface_scaling_factors.npy" in found_factors:
        chosen_path = "outputs/Ahmed_Dataset/surface_scaling_factors.npy"
    else:
        chosen_path = list(found_factors.keys())[0]
    
    print(f"🎯 Using scaling factors from: {chosen_path}")
    return found_factors[chosen_path].astype(np.float32)


def debug_model_output(prediction_raw, surf_factors, stream_velocity, air_density):
    """Debug the model output at each step."""
    print("🔍 DEBUGGING MODEL OUTPUT PIPELINE")
    print("="*50)
    
    print(f"1. Raw model output:")
    print(f"   Shape: {prediction_raw.shape}")
    print(f"   Min: {np.min(prediction_raw):.6f}")
    print(f"   Max: {np.max(prediction_raw):.6f}")
    print(f"   Mean: {np.mean(prediction_raw):.6f}")
    print(f"   Sample values: {prediction_raw[0, :5, 0]}")
    
    if surf_factors is not None:
        # Step 1: Unnormalize
        unnormalized = unnormalize(prediction_raw, surf_factors[0], surf_factors[1])
        print(f"\n2. After unnormalization:")
        print(f"   Shape: {unnormalized.shape}")
        print(f"   Min: {np.min(unnormalized):.6f}")
        print(f"   Max: {np.max(unnormalized):.6f}")
        print(f"   Mean: {np.mean(unnormalized):.6f}")
        print(f"   Sample values: {unnormalized[0, :5, 0]}")
        
        # Step 2: Physical scaling
        physical = unnormalized * stream_velocity**2.0 * air_density
        print(f"\n3. After physical scaling (v²ρ = {stream_velocity**2.0 * air_density:.1f}):")
        print(f"   Shape: {physical.shape}")
        print(f"   Min: {np.min(physical):.6f}")
        print(f"   Max: {np.max(physical):.6f}")
        print(f"   Mean: {np.mean(physical):.6f}")
        print(f"   Sample values: {physical[0, :5, 0]}")
        
        return physical
    else:
        print("\n❌ No scaling factors - cannot unnormalize!")
        return prediction_raw


def create_simple_test_case():
    """Create a simple test case with known scaling."""
    print("🧪 CREATING SIMPLE TEST CASE")
    print("="*30)
    
    # Create dummy data with reasonable CFD values
    n_points = 1000
    
    # Pressure coefficient values typically range from -2 to 1
    pressure_coeff = np.random.uniform(-1.0, 0.5, (n_points, 1)).astype(np.float32)
    
    # Wall shear stress typically small values
    wall_shear = np.random.uniform(-0.01, 0.01, (n_points, 3)).astype(np.float32)
    
    # Combine
    coarse_fields = np.concatenate([pressure_coeff, wall_shear], axis=1)
    
    print(f"Created test case:")
    print(f"  Pressure range: [{np.min(pressure_coeff):.3f}, {np.max(pressure_coeff):.3f}]")
    print(f"  Wall shear range: [{np.min(wall_shear):.6f}, {np.max(wall_shear):.6f}]")
    
    return coarse_fields


def test_model_with_simple_data(model, device, surf_factors):
    """Test model with simple known data."""
    print("\n🧪 TESTING MODEL WITH SIMPLE DATA")
    print("="*40)
    
    # Create simple test data
    batch_size = 1
    n_points = 100
    
    # Simple input fields (coarse data)
    surface_fields = np.random.uniform(-0.1, 0.1, (n_points, 4)).astype(np.float32)
    
    # Simple geometry data
    geometry_coords = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)
    surface_coords = np.random.uniform(-1, 1, (n_points, 3)).astype(np.float32)
    surface_normals = np.random.uniform(-1, 1, (n_points, 3)).astype(np.float32)
    surface_areas = np.random.uniform(0.001, 0.01, (n_points,)).astype(np.float32)
    
    # Normalize normals
    norms = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = surface_normals / norms
    
    # Create minimal data dict
    data_dict = {
        "geometry_coordinates": geometry_coords,
        "surface_mesh_centers": surface_coords,
        "surface_normals": surface_normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,  # 4 features
        "stream_velocity": np.array([[STREAM_VELOCITY]], dtype=np.float32),
        "air_density": np.array([[AIR_DENSITY]], dtype=np.float32),
    }
    
    # Convert to tensors
    data_dict_gpu = {}
    for k, v in data_dict.items():
        v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
        data_dict_gpu[k] = v_tensor
    
    print("Input data ranges:")
    print(f"  Surface fields: [{np.min(surface_fields):.6f}, {np.max(surface_fields):.6f}]")
    
    # Run model
    with torch.no_grad():
        try:
            _, prediction_surf = model(data_dict_gpu)
            
            if prediction_surf is not None:
                pred_np = prediction_surf.cpu().numpy()
                print(f"\nRaw prediction ranges:")
                print(f"  Shape: {pred_np.shape}")
                print(f"  Min: {np.min(pred_np):.6f}")
                print(f"  Max: {np.max(pred_np):.6f}")
                
                # Debug the unnormalization process
                final_pred = debug_model_output(
                    pred_np, surf_factors, STREAM_VELOCITY, AIR_DENSITY
                )
                
                # Check if values are reasonable
                pressure_range = [np.min(final_pred[0, :, 0]), np.max(final_pred[0, :, 0])]
                shear_range = [np.min(final_pred[0, :, 1:]), np.max(final_pred[0, :, 1:])]
                
                print(f"\nFinal prediction ranges:")
                print(f"  Pressure: [{pressure_range[0]:.6f}, {pressure_range[1]:.6f}]")
                print(f"  Wall shear: [{shear_range[0]:.6f}, {shear_range[1]:.6f}]")
                
                # Expected ranges for Ahmed body at 30 m/s
                expected_pressure_range = [-2000, 1000]  # Pa
                expected_shear_range = [-50, 50]  # Pa
                
                pressure_reasonable = (pressure_range[0] > expected_pressure_range[0] * 10 and 
                                     pressure_range[1] < expected_pressure_range[1] * 10)
                shear_reasonable = (shear_range[0] > expected_shear_range[0] * 10 and 
                                  shear_range[1] < expected_shear_range[1] * 10)
                
                print(f"\nReasonableness check:")
                print(f"  Pressure reasonable: {'✅' if pressure_reasonable else '❌'}")
                print(f"  Wall shear reasonable: {'✅' if shear_reasonable else '❌'}")
                
                return pressure_reasonable and shear_reasonable
            else:
                print("❌ Model returned None prediction")
                return False
                
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_model_state(model):
    """Check the internal state of the model."""
    print("🔍 CHECKING MODEL INTERNAL STATE")
    print("="*40)
    
    # Check if enhanced features are enabled
    if hasattr(model, 'use_enhanced_features'):
        print(f"Enhanced features enabled: {model.use_enhanced_features}")
        
    if hasattr(model, 'coarse_to_fine_model'):
        print("✅ Coarse-to-fine model exists")
        
        # Check a few parameters
        for name, param in model.coarse_to_fine_model.named_parameters():
            if 'weight' in name:
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                break
    else:
        print("❌ No coarse-to-fine model found")
    
    # Check model mode
    print(f"Model training mode: {model.training}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("="*60)
    print("🔧 ENHANCED DoMINO MODEL DEBUGGING")
    print("="*60)
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Step 1: Debug scaling factors
    surf_factors = debug_scaling_factors()
    
    # Step 2: Create and load model
    print("\n🤖 LOADING MODEL")
    print("="*20)
    
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1
    
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
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(to_absolute_path(checkpoint_path), map_location=dist.device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Step 3: Check model internal state
    check_model_state(model)
    
    # Step 4: Test with simple data
    is_reasonable = test_model_with_simple_data(model, dist.device, surf_factors)
    
    # Step 5: Provide recommendations
    print("\n📋 DEBUGGING SUMMARY AND RECOMMENDATIONS")
    print("="*50)
    
    if not is_reasonable:
        print("❌ Model predictions are not reasonable!")
        print("\nPossible issues:")
        print("1. 🔴 Wrong scaling factors being used")
        print("2. 🔴 Model was trained with different normalization")
        print("3. 🔴 Enhanced model not loading correctly")
        print("4. 🔴 Checkpoint file is from wrong training run")
        
        print("\nRecommended fixes:")
        print("1. ✅ Check that checkpoint is from enhanced training")
        print("2. ✅ Verify scaling factors match training data")
        print("3. ✅ Re-run training with proper enhanced config")
        print("4. ✅ Use original test.py for standard DoMINO model")
        
    else:
        print("✅ Model predictions appear reasonable!")
        print("The enhanced model should work correctly.")
    
    # Step 6: Show correct scaling factors to use
    if surf_factors is not None:
        print(f"\n📊 SCALING FACTORS TO USE:")
        print(f"  Shape: {surf_factors.shape}")
        print(f"  Max values: {surf_factors[0]}")
        print(f"  Min values: {surf_factors[1]}")
        
        # Save correct scaling factors for inference
        inference_path = os.path.join(cfg.eval.scaling_param_path, "surface_scaling_factors_inference.npy")
        os.makedirs(os.path.dirname(inference_path), exist_ok=True)
        np.save(inference_path, surf_factors)
        print(f"💾 Saved inference scaling factors to: {inference_path}")


if __name__ == "__main__":
    main()
