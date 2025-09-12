with open('test_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find and fix the normalization line
for i, line in enumerate(lines):
    if 'coarse_fields_normalized = normalize(coarse_data["fields"]' in line:
        # Replace with robust version
        lines[i-1] = '            # Handle scaling factors with different shapes\n'
        lines[i-1] += '            n_features = coarse_data["fields"].shape[-1]\n'
        lines[i-1] += '            if surf_factors.shape[1] > n_features:\n'
        lines[i-1] += '                # Use only the columns matching the data features\n'
        lines[i-1] += '                coarse_fields_normalized = normalize(coarse_data["fields"], surf_factors[0, :n_features], surf_factors[1, :n_features])\n'
        lines[i-1] += '            else:\n'
        lines[i] = '                coarse_fields_normalized = normalize(coarse_data["fields"], surf_factors[0], surf_factors[1])\n'
        break

with open('test_enhanced.py', 'w') as f:
    f.writelines(lines)

print("Fixed normalization to handle different shapes")
