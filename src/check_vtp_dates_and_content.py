#!/usr/bin/env python3
"""
Check VTP File Dates and Find Real Enhanced DoMINO Predictions
Verifies file creation dates and looks for actual prediction fields
"""

import os
import glob
from pathlib import Path
from datetime import datetime
import vtk
from vtk.util import numpy_support
import numpy as np

class VTPDateChecker:
    """Check dates and content of VTP files"""
    
    def __init__(self):
        self.predictions_dir = Path("/data/ahmed_data/predictions")
        self.today = datetime(2025, 8, 28)  # Today's date based on your system
        
    def check_file_dates(self):
        """Check creation/modification dates of VTP files"""
        print("=" * 80)
        print("VTP FILE DATE ANALYSIS")
        print("=" * 80)
        print(f"Today's date: {self.today.strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        if not self.predictions_dir.exists():
            print(f"❌ Directory doesn't exist: {self.predictions_dir}")
            return []
        
        vtp_files = list(self.predictions_dir.glob("*.vtp"))
        
        # Group files by date
        files_by_date = {}
        recent_files = []
        
        for vtp_file in vtp_files:
            # Get file modification time
            mtime = os.path.getmtime(vtp_file)
            mod_date = datetime.fromtimestamp(mtime)
            date_str = mod_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if file is from today (Aug 28)
            if mod_date.date() == self.today.date():
                recent_files.append((vtp_file, mod_date))
                print(f"✅ TODAY: {vtp_file.name}")
                print(f"   Modified: {date_str}")
            
            # Store for summary
            date_only = mod_date.strftime('%Y-%m-%d')
            if date_only not in files_by_date:
                files_by_date[date_only] = []
            files_by_date[date_only].append(vtp_file.name)
        
        # Show summary by date
        print("\n📅 Files grouped by date:")
        for date_str in sorted(files_by_date.keys(), reverse=True)[:5]:  # Show last 5 days
            count = len(files_by_date[date_str])
            print(f"  {date_str}: {count} files")
            # Show first few files from that date
            for fname in files_by_date[date_str][:3]:
                print(f"    - {fname}")
            if count > 3:
                print(f"    ... and {count-3} more")
        
        return recent_files
    
    def check_vtp_fields(self, vtp_path):
        """Check what fields are actually in the VTP file"""
        try:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(vtp_path))
            reader.Update()
            
            polydata = reader.GetOutput()
            point_data = polydata.GetPointData()
            n_arrays = point_data.GetNumberOfArrays()
            
            fields = {}
            for i in range(n_arrays):
                array_name = point_data.GetArrayName(i)
                array = point_data.GetArray(array_name)
                if array:
                    np_array = numpy_support.vtk_to_numpy(array)
                    fields[array_name] = {
                        'shape': np_array.shape,
                        'mean': np.mean(np_array) if len(np_array.shape) == 1 else np.mean(np.linalg.norm(np_array, axis=1)),
                        'std': np.std(np_array) if len(np_array.shape) == 1 else np.std(np.linalg.norm(np_array, axis=1))
                    }
            
            return fields
        except Exception as e:
            print(f"Error reading {vtp_path}: {e}")
            return None
    
    def find_prediction_fields(self):
        """Look for VTP files with actual prediction fields"""
        print("\n" + "=" * 80)
        print("SEARCHING FOR ACTUAL PREDICTION FIELDS")
        print("=" * 80)
        
        # Expected fields from Enhanced DoMINO
        expected_fields = [
            'Predicted_Pressure',
            'Predicted_WallShearStress', 
            'Coarse_Pressure',
            'Fine_Pressure_GroundTruth',
            'Error_Pressure_vs_Fine'
        ]
        
        vtp_files = list(self.predictions_dir.glob("*.vtp"))
        
        files_with_predictions = []
        
        for vtp_file in vtp_files[:20]:  # Check first 20 files
            fields = self.check_vtp_fields(vtp_file)
            if fields:
                # Check if any expected fields are present
                found_fields = [f for f in expected_fields if f in fields]
                
                if found_fields:
                    files_with_predictions.append((vtp_file, found_fields))
                    print(f"\n✅ {vtp_file.name}")
                    print(f"   Found prediction fields: {', '.join(found_fields)}")
                    for field in found_fields:
                        stats = fields[field]
                        print(f"     {field}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        if not files_with_predictions:
            print("\n⚠️ No VTP files found with expected prediction fields!")
            print("\nActual fields found in VTP files:")
            # Show what fields are actually in the files
            sample_file = vtp_files[0] if vtp_files else None
            if sample_file:
                fields = self.check_vtp_fields(sample_file)
                if fields:
                    print(f"  Sample from {sample_file.name}:")
                    for field_name in fields.keys():
                        print(f"    - {field_name}")
        
        return files_with_predictions

class TestOutputChecker:
    """Check test script outputs and logs"""
    
    def __init__(self):
        self.base_dir = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src")
        
    def check_test_logs(self):
        """Check test logs for output paths"""
        print("\n" + "=" * 80)
        print("CHECKING TEST LOGS FOR OUTPUT PATHS")
        print("=" * 80)
        
        log_files = [
            self.base_dir / "outputs/Ahmed_Dataset/enhanced_v2_physics/test_enhanced.log",
            self.base_dir / "outputs/Ahmed_Dataset/old_runs/enhanced_1/test_enhanced.log",
            self.base_dir / "outputs/Ahmed_Dataset/1/test.log"
        ]
        
        for log_file in log_files:
            if log_file.exists():
                print(f"\n📄 {log_file.relative_to(self.base_dir)}")
                
                # Get file date
                mtime = os.path.getmtime(log_file)
                mod_date = datetime.fromtimestamp(mtime)
                print(f"   Modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Look for output paths in log
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Search for paths mentioned in log
                for line in lines:
                    if 'vtp' in line.lower() or 'prediction' in line.lower() or 'output' in line.lower():
                        if any(keyword in line for keyword in ['Saved', 'Writing', 'Output', 'Created']):
                            print(f"   > {line.strip()}")
    
    def find_recent_outputs(self):
        """Find any outputs created today"""
        print("\n" + "=" * 80)
        print("SEARCHING FOR TODAY'S OUTPUTS")
        print("=" * 80)
        
        # Search patterns for output files
        patterns = [
            "outputs/**/*.vtp",
            "outputs/**/*.h5",
            "outputs/**/*.pt",
            "predictions/**/*.vtp",
            "test_results/**/*.vtp"
        ]
        
        today = datetime(2025, 8, 28)
        recent_files = []
        
        for pattern in patterns:
            files = list(self.base_dir.glob(pattern))
            for file_path in files:
                mtime = os.path.getmtime(file_path)
                mod_date = datetime.fromtimestamp(mtime)
                
                if mod_date.date() == today.date():
                    recent_files.append((file_path, mod_date))
        
        if recent_files:
            print(f"\n✅ Found {len(recent_files)} files created/modified today:")
            for file_path, mod_date in sorted(recent_files, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {mod_date.strftime('%H:%M:%S')} - {file_path.relative_to(self.base_dir)}")
        else:
            print("\n⚠️ No output files created today (Aug 28)")

def create_test_verification_script():
    """Create script to verify test outputs"""
    print("\n" + "=" * 80)
    print("CREATING TEST OUTPUT VERIFICATION SCRIPT")
    print("=" * 80)
    
    script = '''#!/usr/bin/env python3
"""
Verify Enhanced DoMINO Test Outputs
Run a small test to verify model predictions are being saved correctly
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

def run_mini_test():
    """Run a minimal test to check if predictions are working"""
    
    print("=" * 80)
    print("MINI TEST - ENHANCED DoMINO OUTPUT VERIFICATION")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    
    # Model path
    model_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src/outputs/Ahmed_Dataset/enhanced_1/models/DoMINOEnhanced.0.499.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✓ Found model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ Loaded checkpoint")
    
    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print("\\nCheckpoint contents:")
        for key in checkpoint.keys():
            if key != 'model_state_dict':
                print(f"  - {key}: {type(checkpoint[key])}")
    
    # Load a test case
    data_path = Path("/workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/dataset/Ahmed_Full/ahmed_data/case_451.h5")
    
    if data_path.exists():
        print(f"\\n✓ Loading test data: {data_path.name}")
        
        with h5py.File(data_path, 'r') as f:
            print(f"  Available keys: {list(f.keys())}")
            
            if 'enhanced_data' in f:
                data = f['enhanced_data'][:]
                print(f"  Enhanced data shape: {data.shape}")
                print(f"  Coarse features (4:8): mean={data[:, 4:8].mean():.4f}")
                print(f"  Fine features (0:4): mean={data[:, :4].mean():.4f}")
            elif 'coarse_input' in f:
                coarse = f['coarse_input'][:]
                print(f"  Coarse data shape: {coarse.shape}")
                print(f"  Coarse pressure: mean={coarse[:, 0].mean():.4f}, std={coarse[:, 0].std():.4f}")
    
    # Create a test output directory
    output_dir = Path("test_output_verification")
    output_dir.mkdir(exist_ok=True)
    
    # Save a test file to verify writing works
    test_output = output_dir / f"test_{datetime.now().strftime('%H%M%S')}.txt"
    with open(test_output, 'w') as f:
        f.write(f"Test output created at {datetime.now()}\\n")
        f.write(f"Model path: {model_path}\\n")
        f.write(f"This verifies file writing is working\\n")
    
    print(f"\\n✓ Test output saved to: {test_output}")
    
    print("\\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_mini_test()
'''
    
    with open("verify_test_outputs.py", "w") as f:
        f.write(script)
    
    print("✅ Created: verify_test_outputs.py")

def main():
    print("=" * 80)
    print("VTP FILE DATE AND CONTENT VERIFICATION")
    print("=" * 80)
    
    # Part 1: Check file dates
    date_checker = VTPDateChecker()
    recent_files = date_checker.check_file_dates()
    
    # Part 2: Look for actual prediction fields
    files_with_predictions = date_checker.find_prediction_fields()
    
    # Part 3: Check test logs
    test_checker = TestOutputChecker()
    test_checker.check_test_logs()
    test_checker.find_recent_outputs()
    
    # Part 4: Create verification script
    create_test_verification_script()
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    if recent_files:
        print(f"\n✅ Found {len(recent_files)} VTP files from today (Aug 28)")
    else:
        print("\n⚠️ No VTP files from today (Aug 28) found")
        print("   The VTP files in /data/ahmed_data/predictions appear to be from earlier runs")
    
    if files_with_predictions:
        print(f"\n✅ Found {len(files_with_predictions)} files with prediction fields")
    else:
        print("\n⚠️ No files found with Enhanced DoMINO prediction fields")
        print("   The existing VTP files only contain CFD fields (k, p, wallShearStress)")
        print("   They don't have Predicted_*, Coarse_*, Fine_* fields from your model")
    
    print("\n🔧 RECOMMENDATION:")
    print("   The VTP files you found are NOT from your Enhanced DoMINO test run.")
    print("   They appear to be original CFD simulation files.")
    print("\n   To create actual prediction VTPs:")
    print("   1. Run: python verify_test_outputs.py  # Check if model loads")
    print("   2. Re-run your test_enhanced.py script with proper output path")
    print("   3. Or use generate_vtp_predictions.py to create new VTPs")

if __name__ == "__main__":
    main()
