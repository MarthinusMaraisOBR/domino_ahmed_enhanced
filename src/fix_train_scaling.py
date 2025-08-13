#!/usr/bin/env python3
"""Fix train.py to use enhanced scaling factors correctly"""

import re

# Read train.py
with open('train.py', 'r') as f:
    lines = f.readlines()

# Find the line with surf_save_path
modified = False
for i, line in enumerate(lines):
    # Look for the surface scaling factor loading section
    if 'surf_save_path = os.path.join(' in line and 'surface_scaling_factors.npy' in lines[i+1] if i+1 < len(lines) else False:
        print(f"Found scaling factor loading at line {i+1}")
        
        # Replace the entire block with enhanced-aware code
        # Find the end of this statement (next line with a closing parenthesis)
        end_idx = i + 1
        while end_idx < len(lines) and ')' not in lines[end_idx]:
            end_idx += 1
        
        # Create the new code block
        new_code = '''    # Load surface scaling factors - use enhanced version if in enhanced mode
    if use_enhanced_features:
        # Look for enhanced scaling factors in the experiment directory
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, cfg.exp_tag, "surface_scaling_factors.npy"
        )
        if not os.path.exists(surf_save_path):
            # Fallback to main directory enhanced version
            surf_save_path = os.path.join(
                "outputs", cfg.project.name, "surface_scaling_factors_enhanced.npy"
            )
        logger.info(f"Using enhanced surface scaling factors from {surf_save_path}")
    else:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )
'''
        # Replace the lines
        lines[i:end_idx+1] = [new_code]
        modified = True
        break

if modified:
    # Write the modified file
    with open('train.py', 'w') as f:
        f.writelines(lines)
    print("✅ Successfully updated train.py!")
else:
    print("⚠️  Could not find the exact location. Manual update needed.")
    print("\nManual fix: Around line 885, replace the surf_save_path loading with:")
    print("""
    # Load surface scaling factors - use enhanced version if in enhanced mode
    if use_enhanced_features:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, cfg.exp_tag, "surface_scaling_factors.npy"
        )
        logger.info(f"Using enhanced surface scaling factors from {surf_save_path}")
    else:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )
    """)
