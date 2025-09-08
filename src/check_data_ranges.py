import numpy as np
import pyvista as pv

# Check coarse data
coarse_vtp = "/data/ahmed_data/organized/test/coarse/run_451/boundary_451.vtp"
mesh_coarse = pv.read(coarse_vtp)

print("COARSE DATA (run_451):")
if 'p' in mesh_coarse.point_data:
    p_data = mesh_coarse.point_data['p']
    print(f"  Pressure (p): min={p_data.min():.4f}, max={p_data.max():.4f}, mean={p_data.mean():.4f}")
if 'wallShearStress' in mesh_coarse.point_data:
    ws_data = mesh_coarse.point_data['wallShearStress']
    print(f"  Wall shear: min={ws_data.min():.4f}, max={ws_data.max():.4f}, mean={ws_data.mean():.4f}")

# Check fine data
fine_vtp = "/data/ahmed_data/organized/test/fine/run_451/boundary_451.vtp"
mesh_fine = pv.read(fine_vtp)

print("\nFINE DATA (run_451):")
if 'pMean' in mesh_fine.cell_data:
    p_data = mesh_fine.cell_data['pMean']
    print(f"  Pressure (pMean): min={p_data.min():.4f}, max={p_data.max():.4f}, mean={p_data.mean():.4f}")
if 'wallShearStressMean' in mesh_fine.cell_data:
    ws_data = mesh_fine.cell_data['wallShearStressMean']
    print(f"  Wall shear: min={ws_data.min():.4f}, max={ws_data.max():.4f}, mean={ws_data.mean():.4f}")

# Check training data for comparison
train_npy = "/data/ahmed_data/processed/train/run_1.npy"
data = np.load(train_npy, allow_pickle=True).item()
surface_fields = data['surface_fields']
print(f"\nTRAINING DATA (run_1):")
print(f"  Surface fields shape: {surface_fields.shape}")
print(f"  First 4 features (fine): min={surface_fields[:, :4].min():.4f}, max={surface_fields[:, :4].max():.4f}, mean={surface_fields[:, :4].mean():.4f}")
print(f"  Last 4 features (coarse): min={surface_fields[:, 4:].min():.4f}, max={surface_fields[:, 4:].max():.4f}, mean={surface_fields[:, 4:].mean():.4f}")
