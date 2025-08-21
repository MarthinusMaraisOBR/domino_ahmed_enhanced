#!/usr/bin/env python3
import torch
import numpy as np
import subprocess
import sys

# Set environment
import os
os.environ['PYTHONPATH'] = '/workspace/PhysicsNeMo:' + os.environ.get('PYTHONPATH', '')

# Import and patch the test module
import test_enhanced

# Monkey-patch the test function to fix tensor conversion
original_test_fn = test_enhanced.test_enhanced_model

def fixed_test_enhanced_model(data_dict, model, device, cfg, surf_factors):
    """Fixed version that ensures CPU conversion."""
    result = original_test_fn(data_dict, model, device, cfg, surf_factors)
    
    # Convert to CPU numpy if needed
    if result is not None:
        if torch.is_tensor(result):
            result = result.cpu().numpy()
        elif isinstance(result, (list, tuple)):
            result = type(result)(
                r.cpu().numpy() if torch.is_tensor(r) else r 
                for r in result
            )
    
    return result

# Apply the patch
test_enhanced.test_enhanced_model = fixed_test_enhanced_model

# Run the main function
if __name__ == "__main__":
    sys.exit(test_enhanced.main())
