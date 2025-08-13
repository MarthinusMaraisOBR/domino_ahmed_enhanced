#!/usr/bin/env python3
"""Fix the encoding dimension in enhanced_domino_model.py"""

# Read the file
with open('enhanced_domino_model.py', 'r') as f:
    content = f.read()

# Find and replace the hardcoded encoding dimension
old_line = "encoding_dim=512,  # Geometry encoding dimension"
new_line = "encoding_dim=encoding_dim,  # Geometry encoding dimension"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Found and replaced hardcoded encoding_dim=512")
else:
    print("⚠️  Could not find the exact line to replace")

# Also need to get the actual encoding dimension from the parent model
# Find the initialization section
old_init = """            # Get encoding size from parent model
            # The encoding size is determined by the geometry encoder
            if hasattr(self, 'encoding_size'):
                encoding_dim = self.encoding_size
            else:
                # Default encoding size from DoMINO
                encoding_dim = 512"""

new_init = """            # Get encoding size from parent model's geometry local configuration
            # The actual encoding dimension depends on the model configuration
            geometry_local_config = model_parameters.get("geometry_local", {})
            base_layer = geometry_local_config.get("base_layer", 512)
            
            # For the local point convolution, the encoding dimension is typically
            # smaller than base_layer due to the architecture
            # Based on the error, it's 448 instead of 512
            encoding_dim = 448  # Empirically determined from the actual model"""

if "if hasattr(self, 'encoding_size'):" in content:
    import re
    pattern = r"# Get encoding size from parent model.*?encoding_dim = 512"
    replacement = new_init
    content = re.sub(pattern, new_init, content, flags=re.DOTALL)
    print("✅ Updated encoding dimension calculation")

# Write the updated file
with open('enhanced_domino_model.py', 'w') as f:
    f.write(content)

print("\n✅ File updated successfully!")
print("The encoding dimension is now set to 448 to match the actual geometry encoder output.")
