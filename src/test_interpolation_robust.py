import numpy as np
from pathlib import Path
import pyvista as pv
from scipy.spatial import cKDTree

def test_interpolation_quality(fine_path, coarse_path, num_samples=5):
    """Test interpolation quality between coarse and fine meshes"""
    
    results = []
    
    for i in range(1, num_samples + 1):
        case_dir = f"run_{i}"
        
        # Load fine mesh
        fine_vtp = Path(fine_path) / case_dir / f"boundary_{i}.vtp"
        fine_mesh = pv.read(str(fine_vtp))
        fine_coords = np.array(fine_mesh.cell_centers().points)
        
        # Extract fine fields
        fine_pressure = np.array(fine_mesh.cell_data.get('pMean', 
                                fine_mesh.cell_data.get('p', [])))
        
        # Load coarse mesh
        coarse_vtp = Path(coarse_path) / case_dir / f"boundary_{i}.vtp"
        coarse_mesh = pv.read(str(coarse_vtp))
        coarse_coords = np.array(coarse_mesh.cell_centers().points)
        
        # Extract coarse fields
        coarse_pressure = np.array(coarse_mesh.cell_data.get('p',
                                  coarse_mesh.cell_data.get('pressure', [])))
        
        # Interpolate coarse to fine
        tree = cKDTree(coarse_coords)
        distances, indices = tree.query(fine_coords, k=4)
        
        # Inverse distance weighting
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / distances
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Interpolate
        coarse_interp = np.zeros(len(fine_coords))
        for j in range(4):
            coarse_interp += weights[:, j] * coarse_pressure[indices[:, j]]
        
        # Calculate error metrics
        rmse = np.sqrt(np.mean((fine_pressure - coarse_interp)**2))
        rel_rmse = rmse / np.sqrt(np.mean(fine_pressure**2))
        
        # Calculate systematic bias
        bias = np.mean(coarse_interp - fine_pressure)
        rel_bias = bias / np.mean(np.abs(fine_pressure))
        
        results.append({
            'case': i,
            'rmse': rmse,
            'rel_rmse': rel_rmse,
            'bias': bias,
            'rel_bias': rel_bias
        })
        
        print(f"Case {i}: RMSE={rel_rmse:.2%}, Bias={rel_bias:.2%}")
    
    # Summary statistics
    avg_rmse = np.mean([r['rel_rmse'] for r in results])
    avg_bias = np.mean([r['rel_bias'] for r in results])
    
    print(f"\nSummary:")
    print(f"Average relative RMSE: {avg_rmse:.2%}")
    print(f"Average relative bias: {avg_bias:.2%}")
    
    if avg_rmse > 0.10:  # 10% threshold
        print("⚠️ WARNING: Interpolation error too high!")
        print("Consider using more neighbors or better interpolation method")
    else:
        print("✅ Interpolation quality acceptable")
    
    return results

if __name__ == "__main__":
    fine_path = "/data/ahmed_data/organized/train/fine/"
    coarse_path = "/data/ahmed_data/organized/train/coarse/"
    
    results = test_interpolation_quality(fine_path, coarse_path)
