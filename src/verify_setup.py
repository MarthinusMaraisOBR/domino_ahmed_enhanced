# verify_setup.py
#!/usr/bin/env python3
"""
Verification script to check if Enhanced DoMINO is properly configured
"""

import os
import sys
from pathlib import Path
import yaml
import torch

def check_mark(condition):
    return "✅" if condition else "❌"

def main():
    print("="*60)
    print("Enhanced DoMINO Setup Verification")
    print("="*60)
    
    errors = []
    warnings = []
    
    # 1. Check Python version
    print("\n1. Python Environment:")
    print(f"   Python version: {sys.version}")
    python_ok = sys.version_info >= (3, 8)
    print(f"   {check_mark(python_ok)} Python 3.8+ required")
    if not python_ok:
        errors.append("Python version too old")
    
    # 2. Check PyTorch and CUDA
    print("\n2. PyTorch and CUDA:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   {check_mark(cuda_available)} CUDA available")
        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            errors.append("CUDA not available")
    except ImportError:
        print(f"   {check_mark(False)} PyTorch not installed")
        errors.append("PyTorch not installed")
    
    # 3. Check required packages
    print("\n3. Required Packages:")
    packages = {
        'physicsnemo': 'PhysicsNeMo',
        'hydra': 'Hydra',
        'omegaconf': 'OmegaConf',
        'pyvista': 'PyVista',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'apex': 'APEX',
        'vtk': 'VTK',
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   {check_mark(True)} {name}")
        except ImportError:
            print(f"   {check_mark(False)} {name}")
            errors.append(f"{name} not installed")
    
    # 4. Check file structure
    print("\n4. File Structure:")
    required_files = [
        'train.py',
        'test_enhanced.py',
        'enhanced_domino_model.py',
        'openfoam_datapipe.py',
        'process_data.py',
        'conf/config.yaml',
        'compute_enhanced_scaling_factors.py',
        'test_interpolation_robust.py',
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        print(f"   {check_mark(exists)} {file}")
        if not exists:
            errors.append(f"Missing file: {file}")
    
    # 5. Check data directories
    print("\n5. Data Directories:")
    
    # Load config to get paths
    config_path = Path("conf/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check fine data path
        fine_path = config.get('data_processor', {}).get('input_dir', '')
        fine_exists = Path(fine_path).exists() if fine_path else False
        print(f"   {check_mark(fine_exists)} Fine data: {fine_path}")
        if not fine_exists:
            errors.append(f"Fine data directory not found: {fine_path}")
        
        # Check coarse data path
        coarse_path = config.get('data_processor', {}).get('coarse_input_dir', '')
        coarse_exists = Path(coarse_path).exists() if coarse_path else False
        print(f"   {check_mark(coarse_exists)} Coarse data: {coarse_path}")
        if not coarse_exists:
            errors.append(f"Coarse data directory not found: {coarse_path}")
        
        # Check if enhanced features are enabled
        enhanced_enabled = config.get('data_processor', {}).get('use_enhanced_features', False)
        print(f"   {check_mark(enhanced_enabled)} Enhanced features enabled in config")
        if not enhanced_enabled:
            warnings.append("Enhanced features not enabled in config.yaml")
        
        # Check critical model settings
        enhanced_model_config = config.get('model', {}).get('enhanced_model', {})
        if enhanced_model_config:
            surface_features = enhanced_model_config.get('surface_input_features', 4)
            print(f"   {check_mark(surface_features == 8)} Surface input features = 8 (got {surface_features})")
            
            c2f_config = enhanced_model_config.get('coarse_to_fine', {})
            use_residual = c2f_config.get('use_residual', True)
            print(f"   {check_mark(not use_residual)} Residual connection disabled")
            if use_residual:
                errors.append("CRITICAL: Residual connection must be disabled!")
            
            use_spectral = c2f_config.get('use_spectral', True)
            print(f"   {check_mark(not use_spectral)} Spectral features disabled")
            if use_spectral:
                warnings.append("Spectral features should be disabled for stability")
        
        # Count available data
        if fine_exists:
            fine_cases = len(list(Path(fine_path).iterdir()))
            print(f"   Found {fine_cases} fine resolution cases")
        
        if coarse_exists:
            coarse_cases = len(list(Path(coarse_path).iterdir()))
            print(f"   Found {coarse_cases} coarse resolution cases")
    else:
        errors.append("config.yaml not found")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
    else:
        print("\n✅ No errors found!")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if not errors and not warnings:
        print("\n🎉 Everything looks good! You're ready to proceed.")
    
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())