#!/usr/bin/env python3
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
        print(f"\nProcessing case {case_id}...")
        
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
    
    print(f"\n✅ VTP files saved to: {generator.output_dir}")
    print("\n📊 To visualize in ParaView:")
    print("  1. Open ParaView")
    print("  2. File -> Open -> Select VTP files")
    print("  3. Apply -> Color by different fields")

if __name__ == "__main__":
    main()
