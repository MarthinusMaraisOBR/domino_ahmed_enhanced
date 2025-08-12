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
        'test.py',
        'enhanced_domino_model.py',
        'openfoam_datapipe.py',
        'process_data.py',
        'conf/config.yaml',
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
        
        # Count available data
        if fine_exists:
            fine_cases = len(list(Path(fine_path).iterdir()))
            print(f"   Found {fine_cases} fine resolution cases")
        
        if coarse_exists:
            coarse_cases = len(list(Path(coarse_path).iterdir()))
            print(f"   Found {coarse_cases} coarse resolution cases")
    else:
        errors.append("config.yaml not found")
    
    # 6. Check enhanced model configuration
    print("\n6. Enhanced Model Configuration:")
    if config_path.exists():
        model_config = config.get('model', {})
        enhanced_config = model_config.get('enhanced_model', {})
        
        surface_features = enhanced_config.get('surface_input_features', 4)
        print(f"   Surface input features: {surface_features}")
        print(f"   {check_mark(surface_features == 8)} Should be 8 for enhanced mode")
        if surface_features != 8:
            warnings.append("surface_input_features should be 8 for enhanced mode")
        
        # Check coarse-to-fine config
        c2f_config = enhanced_config.get('coarse_to_fine', {})
        use_spectral = c2f_config.get('use_spectral', True)
        use_residual = c2f_config.get('use_residual', True)
        print(f"   Use spectral features: {use_spectral}")
        print(f"   Use residual connections: {use_residual}")
    
    # 7. Test enhanced model import
    print("\n7. Enhanced Model Import Test:")
    try:
        from enhanced_domino_model import DoMINOEnhanced
        print(f"   {check_mark(True)} DoMINOEnhanced can be imported")
        
        # Try to create a small model
        try:
            test_model = DoMINOEnhanced(
                input_features=3,
                output_features_vol=None,
                output_features_surf=4,
                model_parameters={'enhanced_model': {'surface_input_features': 8}}
            )
            print(f"   {check_mark(True)} DoMINOEnhanced can be instantiated")
        except Exception as e:
            print(f"   {check_mark(False)} DoMINOEnhanced instantiation failed: {str(e)}")
            errors.append(f"Model instantiation failed: {str(e)}")
    except ImportError as e:
        print(f"   {check_mark(False)} Cannot import DoMINOEnhanced: {str(e)}")
        errors.append(f"Cannot import enhanced model: {str(e)}")
    
    # 8. Check for existing checkpoints
    print("\n8. Existing Checkpoints:")
    checkpoint_dir = Path("outputs/Ahmed_Dataset/1/models")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        print(f"   Found {len(checkpoints)} checkpoint(s)")
        for ckpt in checkpoints[:3]:  # Show first 3
            print(f"     - {ckpt.name}")
    else:
        print(f"   No checkpoints found (this is OK for first run)")
    
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
        print("\n🎉 Everything looks good! You're ready to train Enhanced DoMINO.")
        print("\nNext steps:")
        print("1. Run: python process_data.py")
        print("2. Run: python train.py")
    elif not errors:
        print("\n⚠️  Setup is functional but check the warnings above.")
    else:
        print("\n❌ Please fix the errors above before proceeding.")
    
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
