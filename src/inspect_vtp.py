#!/usr/bin/env python3
"""
Quick script to inspect VTP files and see available variables
"""

import pyvista as pv
from pathlib import Path

def inspect_vtp_files():
    """Inspect your VTP files to see available variables"""

    print("="*50)
    print("VTP FILE INSPECTION")
    print("="*50)

    # Your data paths
    fine_data_path = "/data/ahmed_data/raw"
    coarse_data_path = "/data/ahmed_data_rans/raw"

    # Test case
    test_case = "run_1"
    case_number = "1"

    # Build file paths
    fine_vtp = Path(fine_data_path) / test_case / f"boundary_{case_number}.vtp"
    coarse_vtp = Path(coarse_data_path) / test_case / f"boundary_{case_number}.vtp"

    print(f"Inspecting case: {test_case}")
    print(f"Fine VTP: {fine_vtp}")
    print(f"Coarse VTP: {coarse_vtp}")
    print()

    # Check fine VTP
    print("FINE VTP FILE:")
    print("-" * 20)
    if fine_vtp.exists():
        try:
            mesh = pv.read(str(fine_vtp))
            print(f"✅ File readable")
            print(f"📊 Number of cells: {mesh.n_cells}")
            print(f"📊 Number of points: {mesh.n_points}")
            print(f"📋 Available cell data variables:")
            for i, key in enumerate(mesh.cell_data.keys()):
                array = mesh.cell_data[key]
                print(f"   {i+1}. {key} (shape: {array.shape})")
            print(f"📋 Available point data variables:")
            for i, key in enumerate(mesh.point_data.keys()):
                array = mesh.point_data[key]
                print(f"   {i+1}. {key} (shape: {array.shape})")
        except Exception as e:
            print(f"❌ Error reading fine VTP: {e}")
    else:
        print(f"❌ File not found: {fine_vtp}")

    print()

    # Check coarse VTP
    print("COARSE VTP FILE:")
    print("-" * 20)
    if coarse_vtp.exists():
        try:
            mesh = pv.read(str(coarse_vtp))
            print(f"✅ File readable")
            print(f"📊 Number of cells: {mesh.n_cells}")
            print(f"📊 Number of points: {mesh.n_points}")
            print(f"📋 Available cell data variables:")
            for i, key in enumerate(mesh.cell_data.keys()):
                array = mesh.cell_data[key]
                print(f"   {i+1}. {key} (shape: {array.shape})")
            print(f"📋 Available point data variables:")
            for i, key in enumerate(mesh.point_data.keys()):
                array = mesh.point_data[key]
                print(f"   {i+1}. {key} (shape: {array.shape})")
        except Exception as e:
            print(f"❌ Error reading coarse VTP: {e}")
    else:
        print(f"❌ File not found: {coarse_vtp}")

    print()

    # List available cases
    print("AVAILABLE CASES:")
    print("-" * 20)

    print("Fine data cases:")
    try:
        fine_base = Path(fine_data_path)
        if fine_base.exists():
            for case_dir in sorted(fine_base.iterdir()):
                if case_dir.is_dir():
                    vtp_files = list(case_dir.glob("*.vtp"))
                    stl_files = list(case_dir.glob("*.stl"))
                    print(f"  📁 {case_dir.name}: {len(vtp_files)} VTP, {len(stl_files)} STL")
        else:
            print(f"  ❌ Directory not found: {fine_base}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("Coarse data cases:")
    try:
        coarse_base = Path(coarse_data_path)
        if coarse_base.exists():
            for case_dir in sorted(coarse_base.iterdir()):
                if case_dir.is_dir():
                    vtp_files = list(case_dir.glob("*.vtp"))
                    stl_files = list(case_dir.glob("*.stl"))
                    print(f"  📁 {case_dir.name}: {len(vtp_files)} VTP, {len(stl_files)} STL")
        else:
            print(f"  ❌ Directory not found: {coarse_base}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    inspect_vtp_files()
