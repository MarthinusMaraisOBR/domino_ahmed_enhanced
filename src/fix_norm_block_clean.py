with open('test_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find and rewrite the problematic section
for i in range(len(lines)):
    if i >= 689 and 'else:' in lines[i] and 'surf_factors[0], surf_factors[1]' in lines[i+1]:
        # Remove the orphaned line
        if 'coarse_fields_normalized = coarse_data["fields"]' in lines[i+4]:
            lines[i+4] = ''
        break

with open('test_enhanced.py', 'w') as f:
    f.writelines(lines)

print("Cleaned up normalization block")
