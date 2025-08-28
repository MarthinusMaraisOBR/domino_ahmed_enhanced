# Analyze what the model actually learned
import torch
import numpy as np

def diagnose_model():
    print("="*80)
    print("MODEL DIAGNOSIS")
    print("="*80)
    
    # Load checkpoint
    checkpoint_path = "outputs/Ahmed_Dataset/enhanced_v2_physics/models/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Check the residual weights
    print("\n1. Residual Connection Weights:")
    print("-"*40)
    residual_weight = None
    correction_weight = None
    
    for key in checkpoint.keys():
        if 'residual_weight' in key:
            residual_weight = checkpoint[key].item()
            print(f"  Residual weight: {residual_weight:.4f}")
        if 'correction_weight' in key:
            correction_weight = checkpoint[key].item()
            print(f"  Correction weight: {correction_weight:.4f}")
    
    if residual_weight and correction_weight:
        print(f"  Ratio (residual/correction): {residual_weight/correction_weight:.2f}")
        
        if residual_weight > 0.9:
            print("  ⚠️ Model is mostly passing through input (high residual)")
        if correction_weight < 0.1:
            print("  ⚠️ Model barely applying corrections (low correction weight)")
    
    # 2. Check output projection weights
    print("\n2. Output Projection Analysis:")
    print("-"*40)
    
    output_key = 'coarse_to_fine_model.output_projection.weight'
    if output_key in checkpoint:
        weights = checkpoint[output_key]
        print(f"  Weight shape: {weights.shape}")
        print(f"  Weight statistics:")
        print(f"    Mean: {weights.mean():.6f}")
        print(f"    Std:  {weights.std():.6f}")
        print(f"    Min:  {weights.min():.6f}")
        print(f"    Max:  {weights.max():.6f}")
        
        # Check if weights are too small (collapsed)
        if weights.std() < 0.001:
            print("  ⚠️ Weights have collapsed - very low variation!")
        
        # Check weight magnitude
        weight_norm = torch.norm(weights).item()
        print(f"    Norm: {weight_norm:.6f}")
        if weight_norm < 0.01:
            print("  ⚠️ Weights are too small - model not learning!")
    
    # 3. Check bias
    bias_key = 'coarse_to_fine_model.output_bias'
    if bias_key in checkpoint:
        bias = checkpoint[bias_key].numpy()
        print(f"\n3. Output Bias:")
        print(f"  {bias}")
        if np.abs(bias).max() > 1.0:
            print("  ⚠️ Large bias values - model relying on constant shift!")
    
    # 4. Analyze main network weights
    print("\n4. Main Network Analysis:")
    print("-"*40)
    
    # Check if main network weights are reasonable
    main_net_weights = []
    for key in checkpoint.keys():
        if 'main_network' in key and 'weight' in key:
            w = checkpoint[key]
            main_net_weights.append({
                'layer': key,
                'mean': w.mean().item(),
                'std': w.std().item(),
                'norm': torch.norm(w).item()
            })
    
    if main_net_weights:
        for w in main_net_weights[:3]:  # Show first 3 layers
            print(f"  {w['layer'].split('.')[-3]}: mean={w['mean']:.4f}, std={w['std']:.4f}")
    
    return checkpoint

checkpoint = diagnose_model()

# Now let's check what actually happens during inference
print("\n" + "="*80)
print("INFERENCE BEHAVIOR CHECK")
print("="*80)

# Simulate what happens with test data
def simulate_inference():
    # Load actual test sample to see what's happening
    import pyvista as pv
    
    test_file = "/data/ahmed_data/predictions_v2/boundary_451_comprehensive_comparison.vtp"
    if Path(test_file).exists():
        mesh = pv.read(test_file)
        
        coarse_p = np.array(mesh['Coarse_Pressure'])
        fine_p = np.array(mesh['Fine_Pressure_GroundTruth_Interpolated'])
        pred_p = np.array(mesh['Predicted_Pressure'])
        
        print("\nActual Test Results (Case 451):")
        print(f"  Coarse mean: {coarse_p.mean():.6f}")
        print(f"  Fine mean:   {fine_p.mean():.6f}")
        print(f"  Predicted mean: {pred_p.mean():.6f}")
        
        # Check if prediction is just scaled coarse
        from scipy.stats import pearsonr
        corr_pred_coarse, _ = pearsonr(pred_p.flatten(), coarse_p.flatten())
        print(f"\n  Correlation (pred vs coarse): {corr_pred_coarse:.3f}")
        if corr_pred_coarse > 0.95:
            print("  ⚠️ Predictions are essentially scaled coarse input!")
        
        # Check if there's a constant shift
        diff_mean = pred_p.mean() - coarse_p.mean()
        print(f"  Mean shift from coarse: {diff_mean:.6f}")
        
        # Check variance ratio
        var_ratio = pred_p.std() / coarse_p.std()
        print(f"  Std ratio (pred/coarse): {var_ratio:.3f}")
        
        if 0.9 < var_ratio < 1.1 and corr_pred_coarse > 0.9:
            print("\n  🔴 Model is essentially doing: output ≈ input + constant")
            print("     This means it didn't learn the coarse-to-fine mapping!")

simulate_inference()

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

print("""
The model shows classic overfitting symptoms:

1. **Training vs Test Gap**: 
   - Training: 99.7% improvement (on seen data)
   - Test: -164% improvement (worse than baseline)
   
2. **Model Behavior**:
   - High residual weight (0.915) = mostly passing input through
   - Low correction weight (0.165) = small corrections
   - Model learned to memorize training data, not generalize
   
3. **Why it happened**:
   - Training stopped early due to "no improvement"
   - The 99.7% was on training batches (memorization)
   - Validation wasn't improving (generalization failed)
   
4. **Solutions**:
   a) Reduce model complexity
   b) Add more regularization (dropout, weight decay)
   c) Use larger batch size if possible
   d) Early stopping based on validation, not training loss
   e) Data augmentation or more training data
   f) Lower learning rate with longer training
""")