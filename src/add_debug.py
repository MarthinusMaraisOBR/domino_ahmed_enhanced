with open('test_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find where we normalize and add debug print
for i, line in enumerate(lines):
    if 'coarse_fields_normalized = (coarse_data' in line:
        # Add debug prints after this line
        lines.insert(i+1, '                print(f"  DEBUG - Original coarse mean: {coarse_data[\"fields\"].mean():.4f}, std: {coarse_data[\"fields\"].std():.4f}")\n')
        lines.insert(i+2, '                print(f"  DEBUG - Normalized coarse mean: {coarse_fields_normalized.mean():.4f}, std: {coarse_fields_normalized.std():.4f}")\n')
        lines.insert(i+3, '                print(f"  DEBUG - Normalized range: [{coarse_fields_normalized.min():.4f}, {coarse_fields_normalized.max():.4f}]")\n')
        break

with open('test_enhanced.py', 'w') as f:
    f.writelines(lines)
print("Debug prints added")
