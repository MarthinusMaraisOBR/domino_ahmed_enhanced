import sys
import os

# Read the train.py file
with open('train.py', 'r') as f:
    content = f.read()

# Find and update the scaling factor loading section
old_surf_path = 'surf_save_path = os.path.join(\n        "outputs", cfg.project.name, "surface_scaling_factors.npy"\n    )'
new_surf_path = '''# Use enhanced scaling factors if enhanced mode is enabled
    if use_enhanced_features:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, f"{cfg.exp_tag}", "surface_scaling_factors.npy"
        )
        # Fallback to computing if not exists
        if not os.path.exists(surf_save_path):
            print("Computing enhanced scaling factors...")
            from physicsnemo.datapipes.cae.domino_datapipe import compute_scaling_factors
            compute_scaling_factors(cfg, cfg.data_processor.output_dir, use_cache=cfg.data_processor.use_cache)
            surf_save_path = os.path.join(
                "outputs", cfg.project.name, f"{cfg.exp_tag}", "surface_scaling_factors.npy"
            )
    else:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )'''

# Check if we can find and replace
if 'surf_save_path = os.path.join(' in content:
    print("✅ Found scaling factor loading code")
    # For now, just print instructions since the exact replacement is complex
    print("\nManual fix needed in train.py around line 880:")
    print("Change the surf_save_path to use the enhanced scaling factors")
else:
    print("❌ Could not find the exact string to replace")

print("\nQuick fix: Copy standard scaling factors and duplicate for 8 features:")
print("python -c \"import numpy as np; sf = np.load('outputs/Ahmed_Dataset/surface_scaling_factors.npy'); sf_8 = np.concatenate([sf, sf], axis=1); np.save('outputs/Ahmed_Dataset/enhanced_1/surface_scaling_factors.npy', sf_8); print('Created 8-feature scaling factors')\"")
