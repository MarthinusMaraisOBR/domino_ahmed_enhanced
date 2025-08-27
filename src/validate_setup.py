# validate_setup.py - Validate setup before training

import torch
from enhanced_domino_model import DoMINOEnhanced
from physics_loss import PhysicsAwareLoss
import yaml

def validate_setup():
    """Check that everything is configured correctly."""
    
    print("Pre-training Validation")
    print("=" * 60)
    
    # 1. Check model initialization
    print("\n1. Testing model initialization...")
    try:
        # Use your actual config file name
        with open('conf/config.yaml', 'r') as f:  # Changed from config_retrain.yaml
            config = yaml.safe_load(f)
        
        from omegaconf import DictConfig
        model_config = DictConfig(config['model'])
        
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters=model_config
        )
        print("   ✓ Model initialized successfully")
        
        # Check initial weights
        if hasattr(model, 'coarse_to_fine_model'):
            c2f = model.coarse_to_fine_model
            print(f"   ✓ Initial residual weight: {c2f.residual_weight.item():.3f}")
            print(f"   ✓ Initial correction weight: {c2f.correction_weight.item():.3f}")
            
            # Check if physics loss is enabled
            if config['model'].get('use_physics_loss', False):
                print("   ✓ Physics-aware loss: ENABLED")
            else:
                print("   ⚠ Physics-aware loss: DISABLED")
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        return False
    
    # 2. Check loss function
    print("\n2. Testing loss function...")
    try:
        if config['model'].get('use_physics_loss', False):
            loss_fn = PhysicsAwareLoss()
            
            # Create dummy data
            batch_size, n_points, n_features = 2, 1000, 4
            pred = torch.randn(batch_size, n_points, n_features)
            target = torch.randn(batch_size, n_points, n_features)
            coarse = torch.randn(batch_size, n_points, n_features)
            areas = torch.ones(batch_size, n_points, 1)
            normals = torch.randn(batch_size, n_points, 3)
            
            loss, components = loss_fn(pred, target, coarse, areas, normals, model)
            print(f"   ✓ Physics loss computed: {loss.item():.6f}")
            print(f"   ✓ Components: {list(components.keys())}")
        else:
            print("   ⚠ Physics loss not enabled in config")
    except Exception as e:
        print(f"   ✗ Loss function failed: {e}")
        return False
    
    # 3. Check data availability
    print("\n3. Checking data availability...")
    from pathlib import Path
    
    train_path = Path("/data/ahmed_data/processed/train")
    if train_path.exists():
        train_files = list(train_path.glob("*.npy"))
        print(f"   ✓ Found {len(train_files)} training files")
    else:
        print(f"   ✗ Training data not found at {train_path}")
        return False
    
    # 4. Check configuration settings
    print("\n4. Checking configuration...")
    try:
        enhanced_config = config['model'].get('enhanced_model', {})
        c2f_config = enhanced_config.get('coarse_to_fine', {})
        
        print(f"   Enhanced features: {config['data_processor'].get('use_enhanced_features', False)}")
        print(f"   Use residual: {c2f_config.get('use_residual', False)}")
        print(f"   Dropout rate: {c2f_config.get('dropout_rate', 0)}")
        print(f"   Hidden layers: {c2f_config.get('hidden_layers', [])}")
        
        if c2f_config.get('use_residual', False):
            print(f"   Residual weight range: [{c2f_config.get('residual_weight_min', 0.7)}, {c2f_config.get('residual_weight_max', 1.0)}]")
            print(f"   Max correction weight: {c2f_config.get('correction_weight_max', 0.3)}")
    except Exception as e:
        print(f"   ✗ Configuration check failed: {e}")
    
    print("\n✓ All checks passed! Ready to train.")
    return True

if __name__ == "__main__":
    validate_setup()