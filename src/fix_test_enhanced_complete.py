import re

with open('test_enhanced.py', 'r') as f:
    content = f.read()

# Replace the test_enhanced_model function with a robust version
new_function = '''def test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """
    Test the enhanced model with STRICT float32 type enforcement.
    Handles both 4-feature and 8-feature scaling factors properly.
    """
    
    with torch.no_grad():
        # CRITICAL: Ensure ALL data is float32 and on correct device
        data_dict_gpu = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                # Convert to float32 tensor and move to device
                v_tensor = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float32)
                data_dict_gpu[k] = v_tensor
            else:
                data_dict_gpu[k] = v
        
        # Verify all tensors are float32
        for k, v in data_dict_gpu.items():
            if isinstance(v, torch.Tensor):
                if v.dtype != torch.float32:
                    print(f"WARNING: Converting {k} from {v.dtype} to float32")
                    data_dict_gpu[k] = v.to(dtype=torch.float32)
        
        # Get predictions
        _, prediction_surf = model(data_dict_gpu)
        
        if prediction_surf is not None and surf_factors is not None:
            # Get the number of features in predictions
            n_features = prediction_surf.shape[-1]
            
            # Handle scaling factors properly
            if surf_factors.shape[1] > n_features:
                # If scaling factors have more columns than features, use only first n_features columns
                surf_max = surf_factors[0, :n_features]
                surf_min = surf_factors[1, :n_features]
            else:
                # Use all columns
                surf_max = surf_factors[0]
                surf_min = surf_factors[1]
            
            # Unnormalize predictions
            prediction_surf = unnormalize(
                prediction_surf.cpu().numpy(),
                surf_max,
                surf_min
            )
            
    return prediction_surf'''

# Find and replace the function
pattern = r'def test_enhanced_model\(.*?\):\s*""".*?""".*?return prediction_surf'
content = re.sub(pattern, new_function, content, flags=re.DOTALL)

with open('test_enhanced.py', 'w') as f:
    f.write(content)

print("Fixed test_enhanced_model function")
