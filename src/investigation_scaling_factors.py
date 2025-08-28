#!/usr/bin/env python3
"""
Comprehensive Investigation of Scaling Factors and VTP Flow Fields
This script thoroughly analyzes scaling factors and VTP predictions to identify issues
"""

import torch
import numpy as np
import vtk
from vtk.util import numpy_support
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import json

class ScalingInvestigator:
    """Investigate scaling factor issues"""
    
    def __init__(self):
        self.base_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src")
        self.data_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data")
        
    def find_all_scaling_factors(self):
        """Find and analyze ALL scaling factor files"""
        print("=" * 80)
        print("SCALING FACTORS INVESTIGATION")
        print("=" * 80)
        
        # Search for all scaling files
        scaling_files = list(self.base_dir.glob("**/scaling*.pt"))
        scaling_files.extend(list(self.base_dir.glob("**/*scale*.pt")))
        
        print(f"\nFound {len(scaling_files)} scaling factor files:\n")
        
        all_scalings = {}
        for sf_path in scaling_files:
            print(f"📁 {sf_path.relative_to(self.base_dir)}")
            try:
                scaling = torch.load(sf_path, map_location='cpu')
                
                if isinstance(scaling, dict):
                    print("  Type: Dictionary")
                    for key, value in scaling.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                            print(f"      Mean: {value.mean():.6f}, Std: {value.std():.6f}")
                            print(f"      Min: {value.min():.6f}, Max: {value.max():.6f}")
                else:
                    print(f"  Type: Tensor, Shape: {scaling.shape}, dtype: {scaling.dtype}")
                    print(f"    Row 0 (mean): {scaling[0].numpy()}")
                    print(f"    Row 1 (std):  {scaling[1].numpy()}")
                    
                all_scalings[str(sf_path)] = scaling
                print()
            except Exception as e:
                print(f"  ❌ Error loading: {e}\n")
        
        return all_scalings
    
    def compare_data_statistics(self):
        """Compare statistics between raw data and scaled data"""
        print("\n" + "=" * 80)
        print("DATA STATISTICS COMPARISON")
        print("=" * 80)
        
        # Load a sample training file
        train_file = self.data_dir / "case_1.h5"
        test_file = self.data_dir / "case_451.h5"
        
        def analyze_h5_file(filepath, label):
            print(f"\n{label}: {filepath.name}")
            print("-" * 40)
            
            with h5py.File(filepath, 'r') as f:
                print("Available keys:", list(f.keys()))
                
                for key in f.keys():
                    data = f[key][:]
                    print(f"\n  {key}:")
                    print(f"    Shape: {data.shape}")
                    print(f"    Mean: {data.mean():.6f}, Std: {data.std():.6f}")
                    print(f"    Min: {data.min():.6f}, Max: {data.max():.6f}")
                    
                    if len(data.shape) == 2 and data.shape[1] >= 4:
                        # Analyze each feature
                        print("    Per-feature statistics:")
                        for i in range(min(4, data.shape[1])):
                            feat_data = data[:, i]
                            print(f"      Feature {i}: mean={feat_data.mean():.4f}, std={feat_data.std():.4f}, "
                                  f"min={feat_data.min():.4f}, max={feat_data.max():.4f}")
        
        if train_file.exists():
            analyze_h5_file(train_file, "TRAINING SAMPLE")
        
        if test_file.exists():
            analyze_h5_file(test_file, "TEST SAMPLE")
    
    def verify_scaling_application(self):
        """Verify how scaling is applied during training vs inference"""
        print("\n" + "=" * 80)
        print("SCALING APPLICATION VERIFICATION")
        print("=" * 80)
        
        # Load a test case
        test_case = self.data_dir / "case_451.h5"
        
        # Find scaling factors
        scaling_path = self.base_dir / "outputs/Ahmed_Dataset/enhanced_1/scaling_factors_enhanced.pt"
        if not scaling_path.exists():
            scaling_path = self.base_dir / "outputs/Ahmed_Dataset/enhanced_1/train/scaling_factors.pt"
        
        if not scaling_path.exists():
            print("❌ No scaling factors found for enhanced_1 model")
            return
        
        print(f"Loading scaling from: {scaling_path}")
        scaling = torch.load(scaling_path, map_location='cpu')
        print(f"Scaling shape: {scaling.shape}")
        
        with h5py.File(test_case, 'r') as f:
            if 'coarse_input' in f:
                coarse_data = f['coarse_input'][:]
                print(f"\nOriginal coarse data shape: {coarse_data.shape}")
                print(f"  Pressure: mean={coarse_data[:, 0].mean():.4f}, std={coarse_data[:, 0].std():.4f}")
                
                # Apply scaling for TRAINING (8 features)
                if scaling.shape[1] == 8:
                    print("\n🔍 TRAINING SCALING (8 features):")
                    # During training, coarse features are in positions 4:8
                    coarse_mean = scaling[0, 4:8]
                    coarse_std = scaling[1, 4:8]
                    scaled_train = (coarse_data - coarse_mean.numpy()) / coarse_std.numpy()
                    print(f"  Scaled pressure: mean={scaled_train[:, 0].mean():.4f}, std={scaled_train[:, 0].std():.4f}")
                
                # Apply scaling for INFERENCE (4 features)
                print("\n🔍 INFERENCE SCALING (4 features):")
                if scaling.shape[1] == 8:
                    # CRITICAL: For inference, we need the coarse scaling from training
                    # which is stored in positions 4:8
                    inference_mean = scaling[0, 4:8]
                    inference_std = scaling[1, 4:8]
                    print(f"  Using coarse subset from 8-feature scaling")
                elif scaling.shape[1] == 4:
                    inference_mean = scaling[0]
                    inference_std = scaling[1]
                    print(f"  Using 4-feature scaling directly")
                
                scaled_inference = (coarse_data - inference_mean.numpy()) / inference_std.numpy()
                print(f"  Scaled pressure: mean={scaled_inference[:, 0].mean():.4f}, std={scaled_inference[:, 0].std():.4f}")
                
                # Check if scaled data is reasonable
                if np.abs(scaled_inference).max() > 10:
                    print("\n⚠️ WARNING: Scaled values exceed reasonable range (>10 std)")
                    print("   This suggests incorrect scaling factors!")

class VTPAnalyzer:
    """Analyze VTP prediction files"""
    
    def __init__(self):
        self.pred_dir = Path("/data/ahmed_data/predictions_v2")
        
    def analyze_vtp_file(self, vtp_path):
        """Analyze a single VTP file"""
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_path))
        reader.Update()
        
        polydata = reader.GetOutput()
        point_data = polydata.GetPointData()
        
        print(f"\n📊 Analyzing: {vtp_path.name}")
        print("-" * 40)
        
        # Get all array names
        n_arrays = point_data.GetNumberOfArrays()
        print(f"Number of arrays: {n_arrays}")
        
        results = {}
        for i in range(n_arrays):
            array_name = point_data.GetArrayName(i)
            array = point_data.GetArray(array_name)
            
            if array:
                np_array = numpy_support.vtk_to_numpy(array)
                
                # Calculate statistics
                if len(np_array.shape) == 1:
                    mean_val = np_array.mean()
                    std_val = np_array.std()
                    min_val = np_array.min()
                    max_val = np_array.max()
                else:
                    # Vector field - compute magnitude
                    magnitude = np.linalg.norm(np_array, axis=1)
                    mean_val = magnitude.mean()
                    std_val = magnitude.std()
                    min_val = magnitude.min()
                    max_val = magnitude.max()
                
                results[array_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'shape': np_array.shape
                }
                
                print(f"  {array_name}:")
                print(f"    Shape: {np_array.shape}")
                print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
        
        return results
    
    def compare_predictions(self):
        """Compare coarse, fine, and predicted fields"""
        print("\n" + "=" * 80)
        print("VTP PREDICTION COMPARISON")
        print("=" * 80)
        
        # Analyze multiple test cases
        test_cases = ['run_451', 'run_455', 'run_460']
        
        comparison_data = {}
        
        for case in test_cases:
            vtp_file = self.pred_dir / f"{case}_prediction.vtp"
            if vtp_file.exists():
                results = self.analyze_vtp_file(vtp_file)
                comparison_data[case] = results
        
        # Create comparison table
        print("\n" + "=" * 80)
        print("FIELD COMPARISON SUMMARY")
        print("=" * 80)
        
        if comparison_data:
            # Compare key fields
            key_fields = ['Coarse_Pressure', 'Fine_Pressure_GroundTruth_Interpolated', 
                         'Predicted_Pressure', 'Error_Pressure_vs_Fine']
            
            print("\n📈 Pressure Field Statistics:")
            print("-" * 60)
            print(f"{'Case':<10} {'Field':<35} {'Mean':<12} {'Std':<12}")
            print("-" * 60)
            
            for case, data in comparison_data.items():
                for field in key_fields:
                    if field in data:
                        stats = data[field]
                        print(f"{case:<10} {field:<35} {stats['mean']:>11.6f} {stats['std']:>11.6f}")
                print()
        
        return comparison_data
    
    def check_physical_validity(self, comparison_data):
        """Check if predictions are physically valid"""
        print("\n" + "=" * 80)
        print("PHYSICAL VALIDITY CHECK")
        print("=" * 80)
        
        for case, data in comparison_data.items():
            print(f"\n{case}:")
            
            # Check if predicted values are in reasonable range
            if 'Predicted_Pressure' in data:
                pred = data['Predicted_Pressure']
                coarse = data.get('Coarse_Pressure', {})
                fine = data.get('Fine_Pressure_GroundTruth_Interpolated', {})
                
                print(f"  Pressure ranges:")
                print(f"    Coarse: [{coarse.get('min', 'N/A'):.4f}, {coarse.get('max', 'N/A'):.4f}]")
                print(f"    Fine:   [{fine.get('min', 'N/A'):.4f}, {fine.get('max', 'N/A'):.4f}]")
                print(f"    Pred:   [{pred['min']:.4f}, {pred['max']:.4f}]")
                
                # Check if prediction is within reasonable bounds
                if fine:
                    fine_range = fine['max'] - fine['min']
                    pred_range = pred['max'] - pred['min']
                    
                    if pred_range > 2 * fine_range:
                        print(f"  ⚠️ WARNING: Prediction range is {pred_range/fine_range:.1f}x larger than fine!")
                    
                    if pred['mean'] < fine['min'] or pred['mean'] > fine['max']:
                        print(f"  ⚠️ WARNING: Prediction mean outside fine bounds!")
                
                # Check error magnitudes
                if 'Error_Pressure_vs_Fine' in data:
                    error = data['Error_Pressure_vs_Fine']
                    relative_error = error['mean'] / abs(fine.get('mean', 1))
                    print(f"  Relative error: {relative_error*100:.1f}%")
                    
                    if abs(relative_error) > 0.5:
                        print(f"  ❌ ERROR: Relative error exceeds 50%!")

def create_scaling_fix_script():
    """Create a script to fix scaling factors for inference"""
    print("\n" + "=" * 80)
    print("CREATING SCALING FIX SCRIPT")
    print("=" * 80)
    
    script = '''#!/usr/bin/env python3
"""
Fix Scaling Factors for Enhanced DoMINO Inference
"""
import torch
import numpy as np
from pathlib import Path

def extract_inference_scaling():
    """Extract correct scaling factors for 4-feature inference"""
    
    # Path to your 8-feature scaling from training
    train_scaling_path = Path("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_enhanced.pt")
    
    if not train_scaling_path.exists():
        print("❌ Training scaling factors not found!")
        return
    
    # Load 8-feature scaling
    scaling_8feat = torch.load(train_scaling_path)
    print(f"Loaded training scaling: shape {scaling_8feat.shape}")
    print(f"  Full scaling matrix:")
    print(f"    Fine features (0:4) mean: {scaling_8feat[0, :4].numpy()}")
    print(f"    Fine features (0:4) std:  {scaling_8feat[1, :4].numpy()}")
    print(f"    Coarse features (4:8) mean: {scaling_8feat[0, 4:].numpy()}")
    print(f"    Coarse features (4:8) std:  {scaling_8feat[1, 4:].numpy()}")
    
    # Extract coarse features only (last 4) for inference
    scaling_inference = scaling_8feat[:, 4:8].clone()
    
    # Save inference scaling
    inference_scaling_path = Path("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_inference.pt")
    torch.save(scaling_inference, inference_scaling_path)
    
    print(f"\\n✅ Created inference scaling: shape {scaling_inference.shape}")
    print(f"  Saved to: {inference_scaling_path}")
    print(f"\\n  Inference scaling values:")
    print(f"    Mean: {scaling_inference[0].numpy()}")
    print(f"    Std:  {scaling_inference[1].numpy()}")
    
    return scaling_inference

def verify_scaling_consistency():
    """Verify that scaling is consistent with data"""
    import h5py
    
    # Load a test file
    test_file = Path("dataset/Ahmed_Full/ahmed_data/case_451.h5")
    
    with h5py.File(test_file, 'r') as f:
        if 'coarse_input' in f:
            coarse = f['coarse_input'][:]
            print(f"\\n📊 Test data statistics (case_451):")
            print(f"  Raw coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
            
            # Apply inference scaling
            scaling = torch.load("outputs/Ahmed_Dataset/enhanced_1/scaling_factors_inference.pt")
            scaled = (coarse - scaling[0].numpy()) / scaling[1].numpy()
            
            print(f"  Scaled pressure: mean={scaled[:, 0].mean():.4f}, std={scaled[:, 0].std():.4f}")
            
            if np.abs(scaled).max() > 10:
                print("  ⚠️ WARNING: Scaled values exceed ±10 std!")
            else:
                print("  ✅ Scaling looks reasonable")

if __name__ == "__main__":
    print("=" * 80)
    print("FIXING SCALING FACTORS FOR INFERENCE")
    print("=" * 80)
    
    scaling = extract_inference_scaling()
    verify_scaling_consistency()
    
    print("\\n" + "=" * 80)
    print("DONE! Use scaling_factors_inference.pt for testing")
    print("=" * 80)
'''
    
    with open("fix_scaling_for_inference.py", "w") as f:
        f.write(script)
    
    print("✅ Created: fix_scaling_for_inference.py")

def main():
    print("=" * 80)
    print("COMPREHENSIVE SCALING AND VTP INVESTIGATION")
    print("=" * 80)
    
    # Part 1: Investigate scaling factors
    scaling_investigator = ScalingInvestigator()
    all_scalings = scaling_investigator.find_all_scaling_factors()
    scaling_investigator.compare_data_statistics()
    scaling_investigator.verify_scaling_application()
    
    # Part 2: Analyze VTP predictions
    vtp_analyzer = VTPAnalyzer()
    comparison_data = vtp_analyzer.compare_predictions()
    if comparison_data:
        vtp_analyzer.check_physical_validity(comparison_data)
    
    # Part 3: Create fixing script
    create_scaling_fix_script()
    
    print("\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)
    
    print("\n🔍 Key Findings:")
    print("1. Check if 8-feature scaling is being used for 4-feature inference")
    print("2. Verify that coarse features use indices 4:8 from training scaling")
    print("3. Compare prediction ranges with ground truth in VTP files")
    print("4. Look for systematic bias in predictions")
    
    print("\n🔧 Next Steps:")
    print("1. Run: python fix_scaling_for_inference.py")
    print("2. Re-test with corrected scaling factors")
    print("3. Visualize VTP files in ParaView to check flow patterns")
    print("4. If issues persist, check model architecture for eval mode")

if __name__ == "__main__":
    main()