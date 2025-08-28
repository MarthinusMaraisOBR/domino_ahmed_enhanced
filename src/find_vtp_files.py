#!/usr/bin/env python3
"""
Find and Analyze VTP Prediction Files
Locates where the test script saved VTP files and analyzes them
"""

import os
import glob
from pathlib import Path
import vtk
from vtk.util import numpy_support
import numpy as np
import torch
import h5py

class VTPFinder:
    """Find and analyze VTP files"""
    
    def __init__(self):
        self.base_dirs = [
            Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src"),
            Path("/data"),
            Path("/workspace"),
            Path(".")
        ]
        
    def find_vtp_files(self):
        """Search for VTP files in common locations"""
        print("=" * 80)
        print("SEARCHING FOR VTP FILES")
        print("=" * 80)
        
        vtp_files = []
        
        # Search patterns
        patterns = [
            "**/predictions*/*.vtp",
            "**/*prediction*.vtp",
            "**/test_results/*.vtp",
            "**/output*/*.vtp",
            "**/*.vtp"
        ]
        
        for base_dir in self.base_dirs:
            if base_dir.exists():
                print(f"\n📁 Searching in: {base_dir}")
                
                for pattern in patterns:
                    found = list(base_dir.glob(pattern))
                    if found:
                        print(f"  Found {len(found)} VTP files with pattern '{pattern}'")
                        for f in found[:5]:  # Show first 5
                            print(f"    - {f.relative_to(base_dir) if base_dir in f.parents else f.name}")
                            vtp_files.append(f)
                        if len(found) > 5:
                            print(f"    ... and {len(found)-5} more")
        
        # Also check specific Enhanced DoMINO output directories
        specific_dirs = [
            Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/test_results"),
            Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/predictions"),
            Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_v2_physics/test_results"),
        ]
        
        print("\n📁 Checking specific Enhanced DoMINO directories:")
        for dir_path in specific_dirs:
            if dir_path.exists():
                vtp_in_dir = list(dir_path.glob("*.vtp"))
                if vtp_in_dir:
                    print(f"  ✓ {dir_path}: {len(vtp_in_dir)} VTP files")
                    vtp_files.extend(vtp_in_dir)
            else:
                print(f"  ✗ {dir_path}: Directory doesn't exist")
        
        return vtp_files
    
    def analyze_vtp_content(self, vtp_path):
        """Analyze the content of a VTP file"""
        try:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(vtp_path))
            reader.Update()
            
            polydata = reader.GetOutput()
            n_points = polydata.GetNumberOfPoints()
            n_cells = polydata.GetNumberOfCells()
            
            point_data = polydata.GetPointData()
            n_arrays = point_data.GetNumberOfArrays()
            
            print(f"\n📊 {vtp_path.name}")
            print(f"  Points: {n_points}, Cells: {n_cells}")
            print(f"  Arrays ({n_arrays}):")
            
            array_info = {}
            for i in range(n_arrays):
                array_name = point_data.GetArrayName(i)
                array = point_data.GetArray(array_name)
                if array:
                    np_array = numpy_support.vtk_to_numpy(array)
                    
                    if len(np_array.shape) == 1:
                        stats = {
                            'mean': np_array.mean(),
                            'std': np_array.std(),
                            'min': np_array.min(),
                            'max': np_array.max()
                        }
                    else:
                        magnitude = np.linalg.norm(np_array, axis=1)
                        stats = {
                            'mean': magnitude.mean(),
                            'std': magnitude.std(),
                            'min': magnitude.min(),
                            'max': magnitude.max()
                        }
                    
                    array_info[array_name] = stats
                    print(f"    {array_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            
            return array_info
            
        except Exception as e:
            print(f"  ❌ Error reading {vtp_path.name}: {e}")
            return None

class TestOutputFinder:
    """Find test script outputs and logs"""
    
    def __init__(self):
        self.base_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src")
        
    def find_test_outputs(self):
        """Find any outputs from test scripts"""
        print("\n" + "=" * 80)
        print("SEARCHING FOR TEST OUTPUTS")
        print("=" * 80)
        
        # Look for test logs
        log_patterns = [
            "**/test*.log",
            "**/test*.txt",
            "**/*test*.out",
            "**/predictions*.txt"
        ]
        
        print("\n📝 Test Logs:")
        for pattern in log_patterns:
            logs = list(self.base_dir.glob(pattern))
            if logs:
                for log in logs[:3]:
                    print(f"  {log.relative_to(self.base_dir)}")
                    # Read first few lines
                    try:
                        with open(log, 'r') as f:
                            lines = f.readlines()[:5]
                            for line in lines:
                                print(f"    > {line.strip()}")
                    except:
                        pass
        
        # Look for prediction directories
        pred_dirs = [
            d for d in self.base_dir.glob("**/prediction*") 
            if d.is_dir()
        ]
        
        if pred_dirs:
            print("\n📁 Prediction Directories:")
            for pred_dir in pred_dirs:
                print(f"  {pred_dir.relative_to(self.base_dir)}")
                # Count files
                vtp_count = len(list(pred_dir.glob("*.vtp")))
                h5_count = len(list(pred_dir.glob("*.h5")))
                print(f"    VTP files: {vtp_count}, H5 files: {h5_count}")

def create_vtp_generation_script():
    """Create a script to generate VTP files from model predictions"""
    print("\n" + "=" * 80)
    print("CREATING VTP GENERATION SCRIPT")
    print("=" * 80)
    
    script = '''#!/usr/bin/env python3
"""
Generate VTP Files from Enhanced DoMINO Predictions
Creates VTP files for visualization in ParaView
"""

import torch
import numpy as np
import vtk
from vtk.util import numpy_support
import h5py
from pathlib import Path

class VTPGenerator:
    def __init__(self, model_path, output_dir="vtp_predictions"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.checkpoint = torch.load(model_path, map_location='cuda:0')
        
        # Data paths
        self.data_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data")
        self.vtp_template_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_vtp")
        
    def load_test_case(self, case_id):
        """Load a test case"""
        h5_file = self.data_dir / f"case_{case_id}.h5"
        vtp_file = self.vtp_template_dir / f"run_{case_id}.vtp"
        
        # Load H5 data
        with h5py.File(h5_file, 'r') as f:
            if 'enhanced_data' in f:
                data = f['enhanced_data'][:]
                coarse = data[:, 4:8]  # Coarse features
                fine = data[:, :4]     # Fine features (ground truth)
            else:
                coarse = f['coarse_input'][:]
                fine = f['fine_output'][:] if 'fine_output' in f else None
        
        # Load VTP template
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_file))
        reader.Update()
        polydata = reader.GetOutput()
        
        return coarse, fine, polydata
    
    def create_prediction_vtp(self, case_id):
        """Create VTP with predictions"""
        print(f"\\nProcessing case {case_id}...")
        
        # Load data
        coarse, fine, polydata = self.load_test_case(case_id)
        
        # Make prediction (simplified - replace with your actual model inference)
        with torch.no_grad():
            coarse_tensor = torch.tensor(coarse, dtype=torch.float32).cuda()
            # Apply scaling if needed
            # prediction = model(coarse_tensor)
            # For now, just use coarse as placeholder
            prediction = coarse  # Replace with actual prediction
        
        # Add arrays to VTP
        point_data = polydata.GetPointData()
        
        # Add coarse input
        coarse_pressure = numpy_support.numpy_to_vtk(coarse[:, 0])
        coarse_pressure.SetName("Coarse_Pressure")
        point_data.AddArray(coarse_pressure)
        
        coarse_wss = numpy_support.numpy_to_vtk(coarse[:, 1:4])
        coarse_wss.SetName("Coarse_WallShearStress")
        point_data.AddArray(coarse_wss)
        
        # Add fine ground truth if available
        if fine is not None:
            fine_pressure = numpy_support.numpy_to_vtk(fine[:, 0])
            fine_pressure.SetName("Fine_Pressure_GroundTruth")
            point_data.AddArray(fine_pressure)
            
            fine_wss = numpy_support.numpy_to_vtk(fine[:, 1:4])
            fine_wss.SetName("Fine_WallShearStress_GroundTruth")
            point_data.AddArray(fine_wss)
        
        # Add prediction
        pred_pressure = numpy_support.numpy_to_vtk(prediction[:, 0])
        pred_pressure.SetName("Predicted_Pressure")
        point_data.AddArray(pred_pressure)
        
        pred_wss = numpy_support.numpy_to_vtk(prediction[:, 1:4])
        pred_wss.SetName("Predicted_WallShearStress")
        point_data.AddArray(pred_wss)
        
        # Add error if ground truth available
        if fine is not None:
            error_pressure = numpy_support.numpy_to_vtk(np.abs(prediction[:, 0] - fine[:, 0]))
            error_pressure.SetName("Error_Pressure")
            point_data.AddArray(error_pressure)
        
        # Write VTP
        output_file = self.output_dir / f"case_{case_id}_prediction.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_file))
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"  ✓ Saved: {output_file}")
        
        # Print statistics
        print(f"  Coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
        if fine is not None:
            print(f"  Fine pressure: mean={fine[:, 0].mean():.4f}, std={fine[:, 0].std():.4f}")
        print(f"  Predicted pressure: mean={prediction[:, 0].mean():.4f}, std={prediction[:, 0].std():.4f}")

def main():
    # Model path - update this
    model_path = "/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt"
    
    # Create generator
    generator = VTPGenerator(model_path)
    
    # Generate for test cases
    test_cases = [451, 455, 460, 470, 490]
    
    print("=" * 80)
    print("GENERATING VTP FILES")
    print("=" * 80)
    
    for case_id in test_cases:
        try:
            generator.create_prediction_vtp(case_id)
        except Exception as e:
            print(f"  ❌ Error processing case {case_id}: {e}")
    
    print(f"\\n✅ VTP files saved to: {generator.output_dir}")
    print("\\n📊 To visualize in ParaView:")
    print("  1. Open ParaView")
    print("  2. File -> Open -> Select VTP files")
    print("  3. Apply -> Color by different fields")

if __name__ == "__main__":
    main()
'''
    
    with open("generate_vtp_predictions.py", "w") as f:
        f.write(script)
    
    print("✅ Created: generate_vtp_predictions.py")

def main():
    print("=" * 80)
    print("VTP FILE INVESTIGATION")
    print("=" * 80)
    
    # Part 1: Find existing VTP files
    vtp_finder = VTPFinder()
    vtp_files = vtp_finder.find_vtp_files()
    
    # Part 2: Analyze found VTP files
    if vtp_files:
        print("\n" + "=" * 80)
        print("ANALYZING VTP CONTENT")
        print("=" * 80)
        
        # Analyze up to 3 VTP files
        for vtp_file in vtp_files[:3]:
            vtp_finder.analyze_vtp_content(vtp_file)
    else:
        print("\n⚠️ No VTP files found in common locations")
    
    # Part 3: Find test outputs
    test_finder = TestOutputFinder()
    test_finder.find_test_outputs()
    
    # Part 4: Create VTP generation script
    create_vtp_generation_script()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if vtp_files:
        print(f"\n✅ Found {len(vtp_files)} VTP files")
        print("\nMost likely locations:")
        # Group by directory
        dirs = {}
        for f in vtp_files:
            dir_path = f.parent
            if dir_path not in dirs:
                dirs[dir_path] = 0
            dirs[dir_path] += 1
        
        for dir_path, count in sorted(dirs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {dir_path}: {count} files")
    else:
        print("\n❌ No existing VTP files found")
        print("\n🔧 To generate VTP files:")
        print("  1. Run: python generate_vtp_predictions.py")
        print("  2. This will create VTP files in 'vtp_predictions' directory")
        print("  3. Open them in ParaView for visualization")
    
    print("\n💡 Next Steps:")
    print("  1. If VTP files exist, open them in ParaView")
    print("  2. If not, generate them with the script")
    print("  3. Check the flow field patterns visually")
    print("  4. Compare coarse vs fine vs predicted fields")

if __name__ == "__main__":
    main()
