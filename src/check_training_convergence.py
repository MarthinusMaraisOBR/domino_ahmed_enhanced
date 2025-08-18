#!/usr/bin/env python3
"""
Step 2: Check if the model is actually learning during training
This script analyzes training logs and tests overfitting capability
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_tensorboard_logs():
    """Analyze training convergence from TensorBoard logs."""
    
    print("="*80)
    print("STEP 2: ANALYZING TRAINING CONVERGENCE")
    print("="*80)
    
    tb_path = Path("outputs/Ahmed_Dataset/enhanced_1/tensorboard")
    
    if not tb_path.exists():
        print(f"❌ TensorBoard logs not found: {tb_path}")
        return None
    
    print(f"\n📊 Loading TensorBoard logs from: {tb_path}")
    
    try:
        # Load tensorboard data
        ea = EventAccumulator(str(tb_path))
        ea.Reload()
        
        # Get available scalars
        scalars = ea.Tags()['scalars']
        print(f"Available metrics: {scalars}")
        
        # Extract loss data
        if 'Loss/train' in scalars:
            train_loss = ea.Scalars('Loss/train')
            steps = [s.step for s in train_loss]
            values = [s.value for s in train_loss]
            
            print(f"\nTraining Loss Analysis:")
            print(f"  Initial loss: {values[0]:.6f}")
            print(f"  Final loss: {values[-1]:.6f}")
            print(f"  Reduction: {(1 - values[-1]/values[0])*100:.1f}%")
            
            # Check if loss is decreasing
            if values[-1] < values[0]:
                print(f"  ✅ Loss is decreasing")
            else:
                print(f"  ❌ Loss is NOT decreasing - training failed!")
                
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Convergence')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig('training_convergence.png', dpi=150)
            print(f"  ✅ Saved plot to training_convergence.png")
            
            return values
            
    except Exception as e:
        print(f"  ❌ Error loading TensorBoard logs: {str(e)}")
        return None

def test_overfit_single_sample():
    """Test if model can overfit to a single training sample."""
    
    print("\n" + "="*80)
    print("TESTING OVERFITTING CAPABILITY")
    print("="*80)
    
    print("\nThis tests if the model architecture can learn at all...")
    
    try:
        from enhanced_domino_model import DoMINOEnhanced
        
        # Create a small model
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=None,
            output_features_surf=4,
            model_parameters={
                'enhanced_model': {
                    'surface_input_features': 8,
                    'coarse_to_fine': {
                        'hidden_layers': [64, 64],  # Smaller for quick test
                        'use_spectral': True,
                        'use_residual': True
                    }
                },
                'activation': 'relu',
                'model_type': 'surface',
                'interp_res': [16, 16, 16],  # Small grid
            }
        )
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create synthetic data
        batch_size = 1
        num_points = 100
        
        # Simulated coarse and fine features
        coarse_features = torch.randn(batch_size, num_points, 4) * 0.5
        fine_features = coarse_features * 2 + torch.randn(batch_size, num_points, 4) * 0.1
        
        # Combine for training
        surface_fields = torch.cat([fine_features, coarse_features], dim=-1)
        
        # Dummy geometry data
        data = {
            'surface_fields': surface_fields,
            'geometry_coordinates': torch.randn(batch_size, 100, 3),
            'surface_mesh_centers': torch.randn(batch_size, num_points, 3),
            'surf_grid': torch.randn(batch_size, 16, 16, 16, 3),
            'sdf_surf_grid': torch.randn(batch_size, 16, 16, 16),
        }
        
        print(f"\nTraining on single sample for 100 iterations...")
        losses = []
        
        for i in range(100):
            optimizer.zero_grad()
            
            # Forward pass
            _, predictions = model(data)
            
            # Simple MSE loss against fine features
            loss = nn.MSELoss()(predictions, fine_features)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 20 == 0:
                print(f"  Iteration {i:3d}: Loss = {loss.item():.6f}")
        
        # Check if model learned
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (1 - final_loss/initial_loss) * 100
        
        print(f"\nOverfitting Test Results:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        if improvement > 50:
            print(f"  ✅ Model CAN learn (architecture is viable)")
        else:
            print(f"  ❌ Model CANNOT learn (architecture problem!)")
            
        # Plot overfitting curve
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Single Sample Overfitting Test')
        plt.grid(True)
        plt.savefig('overfitting_test.png', dpi=150)
        print(f"  ✅ Saved plot to overfitting_test.png")
        
        return improvement > 50
        
    except Exception as e:
        print(f"  ❌ Error in overfitting test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_checkpoint_weights():
    """Analyze if model weights are changing during training."""
    
    print("\n" + "="*80)
    print("CHECKING MODEL WEIGHT EVOLUTION")
    print("="*80)
    
    model_dir = Path("outputs/Ahmed_Dataset/enhanced_1/models")
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    checkpoints = sorted(model_dir.glob("DoMINOEnhanced.*.pt"))
    
    if len(checkpoints) < 2:
        print(f"❌ Need at least 2 checkpoints, found {len(checkpoints)}")
        return False
    
    print(f"\n📊 Found {len(checkpoints)} checkpoints")
    
    # Load first and last checkpoint
    early_ckpt = torch.load(checkpoints[0], map_location='cpu')
    late_ckpt = torch.load(checkpoints[-1], map_location='cpu')
    
    print(f"  Early: {checkpoints[0].name}")
    print(f"  Late:  {checkpoints[-1].name}")
    
    # Compare weights
    weight_changes = []
    
    for key in early_ckpt.keys():
        if 'coarse_to_fine_model' in key:  # Focus on enhanced part
            early_weight = early_ckpt[key]
            late_weight = late_ckpt[key]
            
            # Calculate change
            diff = (late_weight - early_weight).abs().mean().item()
            rel_change = diff / (early_weight.abs().mean().item() + 1e-8)
            
            weight_changes.append(rel_change)
            
            if len(weight_changes) <= 5:  # Show first few
                print(f"  {key[-30:]:30s}: {rel_change:.2%} change")
    
    avg_change = np.mean(weight_changes) if weight_changes else 0
    
    print(f"\nAverage weight change: {avg_change:.2%}")
    
    if avg_change > 0.01:
        print(f"  ✅ Weights ARE changing during training")
    else:
        print(f"  ❌ Weights NOT changing - optimizer issue!")
        
    return avg_change > 0.01

def analyze_predictions_vs_baseline():
    """Compare model predictions against simple baselines."""
    
    print("\n" + "="*80)
    print("ANALYZING PREDICTIONS VS BASELINES")
    print("="*80)
    
    # Load a test prediction
    pred_dir = Path("/data/ahmed_data/predictions")
    
    if not pred_dir.exists():
        print(f"❌ Predictions directory not found: {pred_dir}")
        return
    
    vtp_files = list(pred_dir.glob("*.vtp"))
    
    if not vtp_files:
        print("❌ No prediction files found")
        return
    
    print(f"\n📊 Analyzing {vtp_files[0].name}")
    
    try:
        import pyvista as pv
        mesh = pv.read(str(vtp_files[0]))
        
        # Check available arrays
        print(f"Available arrays: {mesh.array_names}")
        
        # Analyze predicted values
        if 'Predicted_Pressure' in mesh.array_names:
            pred_pressure = mesh['Predicted_Pressure']
            print(f"\nPredicted Pressure Statistics:")
            print(f"  Min: {pred_pressure.min():.6f}")
            print(f"  Max: {pred_pressure.max():.6f}")
            print(f"  Mean: {pred_pressure.mean():.6f}")
            print(f"  Std: {pred_pressure.std():.6f}")
            
            # Check for issues
            if pred_pressure.min() > 0:
                print("  ⚠️ All pressures positive - physically unlikely!")
            if pred_pressure.max() < -5:
                print("  ⚠️ Pressures too negative - check scaling!")
            if pred_pressure.std() < 0.01:
                print("  ⚠️ Very low variance - model outputting constants?")
                
        if 'Coarse_Pressure' in mesh.array_names and 'Fine_Pressure_GroundTruth_Interpolated' in mesh.array_names:
            coarse = mesh['Coarse_Pressure']
            fine = mesh['Fine_Pressure_GroundTruth_Interpolated']
            pred = mesh['Predicted_Pressure'] if 'Predicted_Pressure' in mesh.array_names else coarse
            
            # Calculate errors
            baseline_error = np.mean((coarse - fine)**2)
            model_error = np.mean((pred - fine)**2)
            
            print(f"\nError Comparison:")
            print(f"  Baseline (coarse) MSE: {baseline_error:.6f}")
            print(f"  Model MSE: {model_error:.6f}")
            print(f"  Improvement: {(1 - model_error/baseline_error)*100:.1f}%")
            
            # Simple baseline: average of coarse
            avg_baseline = np.ones_like(fine) * coarse.mean()
            avg_error = np.mean((avg_baseline - fine)**2)
            
            print(f"  Constant avg MSE: {avg_error:.6f}")
            
            if model_error > baseline_error:
                print("  ❌ Model is WORSE than using coarse directly!")
            if model_error > avg_error:
                print("  ❌ Model is WORSE than predicting the mean!")
                
    except Exception as e:
        print(f"  ❌ Error analyzing predictions: {str(e)}")

def diagnose_architecture_issues():
    """Diagnose potential architecture problems."""
    
    print("\n" + "="*80)
    print("ARCHITECTURE DIAGNOSIS")
    print("="*80)
    
    print("\n🔍 Checking CoarseToFineModel architecture...")
    
    from enhanced_domino_model import CoarseToFineModel
    
    # Test the coarse-to-fine model in isolation
    model = CoarseToFineModel(
        input_dim=4,
        output_dim=4,
        encoding_dim=448,
        hidden_layers=[512, 512, 512],
        use_spectral=True,
        use_residual=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test with dummy data
    batch = 1
    points = 100
    
    coarse = torch.randn(batch, points, 4) * 0.5
    geometry = torch.randn(batch, points, 448)
    
    with torch.no_grad():
        output = model(coarse, geometry)
    
    print(f"  Input shape: {coarse.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Check if residual connection dominates
    if hasattr(model, 'residual_projection'):
        residual = model.residual_projection(coarse)
        print(f"\n  Residual projection exists")
        print(f"  Residual norm: {residual.norm():.4f}")
        print(f"  Output norm: {output.norm():.4f}")
        
        if output.norm() < residual.norm() * 1.5:
            print("  ⚠️ WARNING: Residual might be dominating output!")
            print("     Consider reducing residual weight or removing it")
    
    print("\n💡 POTENTIAL ISSUES:")
    print("1. Residual connection may pass through coarse unchanged")
    print("2. Geometry encoding dimension (448) might be too large")
    print("3. Spectral features might not be learning useful patterns")
    print("4. Network might be too deep/shallow for the task")
    
    return True

def main():
    """Run all diagnostic steps."""
    
    print("\n🔍 DEEP DIVE: TRAINING CONVERGENCE ANALYSIS\n")
    
    # Step 1: Analyze training logs
    train_losses = analyze_tensorboard_logs()
    
    # Step 2: Test overfitting capability
    can_overfit = test_overfit_single_sample()
    
    # Step 3: Check weight evolution
    weights_change = check_checkpoint_weights()
    
    # Step 4: Analyze predictions
    analyze_predictions_vs_baseline()
    
    # Step 5: Diagnose architecture
    diagnose_architecture_issues()
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    issues = []
    
    if train_losses is None:
        issues.append("Cannot verify training convergence")
    elif train_losses and train_losses[-1] >= train_losses[0]:
        issues.append("Training loss not decreasing!")
        
    if not can_overfit:
        issues.append("Model cannot overfit - architecture problem!")
        
    if not weights_change:
        issues.append("Model weights not updating during training")
    
    if issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
            
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Check if loss function is correct")
        print("2. Verify optimizer is working")
        print("3. Try removing residual connections")
        print("4. Reduce model complexity")
        print("5. Check data preprocessing pipeline")
    else:
        print("\n✅ Training appears to be working")
        print("\n🔍 But predictions are still wrong, check:")
        print("1. Scaling/normalization consistency")
        print("2. Feature ordering during inference")
        print("3. Model architecture effectiveness")
    
    return len(issues) == 0
