with open('test_enhanced.py', 'r') as f:
    content = f.read()

# Replace the scaling factor filename
old_filename = '"surface_scaling_factors.npy"'
new_filename = '"surface_scaling_factors_inference.npy"'

content = content.replace(old_filename, new_filename)

with open('test_enhanced.py', 'w') as f:
    f.write(content)
print("Fixed test script to use inference scaling factors")
