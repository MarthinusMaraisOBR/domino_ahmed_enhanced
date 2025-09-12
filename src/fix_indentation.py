with open('test_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find the problematic area around line 685-687
for i in range(len(lines)):
    if 'else:' in lines[i] and 'coarse_fields_normalized' not in lines[i]:
        # Check if the next non-empty line needs indentation
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() and not lines[j].startswith(' '):
                # This line needs indentation
                lines[j] = '                ' + lines[j]
                break

with open('test_enhanced.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")
