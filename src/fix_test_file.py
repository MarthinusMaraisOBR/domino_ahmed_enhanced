with open('test_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find the data_dict creation section
for i in range(len(lines)):
    if '"surface_fields": coarse_data[\'fields\']' in lines[i]:
        # Replace this line with normalized version
        indent = '                '
        lines[i] = f'{indent}"surface_fields": coarse_fields_normalized,  # Normalized coarse features\n'
        
        # Add normalization code before data_dict
        normalization_code = f'''            # Normalize coarse fields using inference scaling factors
            if surf_factors is not None:
                coarse_fields_normalized = (coarse_data['fields'] - surf_factors[1]) / (surf_factors[0] - surf_factors[1])
            else:
                coarse_fields_normalized = coarse_data['fields']
            
'''
        # Find where to insert (before data_dict = {)
        for j in range(i-10, i):
            if 'data_dict = {' in lines[j]:
                lines.insert(j, normalization_code)
                break
        break

with open('test_enhanced.py', 'w') as f:
    f.writelines(lines)

print("Fixed normalization in test_enhanced.py")
