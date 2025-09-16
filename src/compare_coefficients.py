import pandas as pd
import numpy as np
import re
import sys

# ============= CONFIGURATION SECTION =============
# Change these variables for different test scenarios

# Input/Output configuration
TEST_OUTPUT_FILE = 'test_output_enhanced_500_test.txt'  # Change to your test output file
OUTPUT_CSV_NAME = 'coefficient_comparison_enhanced_500_test.csv'  # Change output name
DATA_TYPE = 'test'  # Options: 'train', 'test', or 'test_on_train'

# ============= END CONFIGURATION =============

# Read force predictions from enhanced model output
forces = []
with open(TEST_OUTPUT_FILE, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if 'Processing: run_' in lines[i]:
        match = re.search(r'run_(\d+)', lines[i])
        if match:
            run_num = int(match.group(1))
            current_run = {'run': run_num}
            
            # Look for Drag Forces section
            j = i
            while j < min(i + 100, len(lines)):
                if 'Drag Forces:' in lines[j]:
                    current_run['drag_coarse'] = float(lines[j+1].split()[1])
                    current_run['drag_true'] = float(lines[j+2].split()[2])  # Fine (Interp)
                    current_run['drag_pred'] = float(lines[j+3].split()[1])
                    
                    # Find Lift Forces
                    k = j + 4
                    while k < min(j + 20, len(lines)):
                        if 'Lift Forces:' in lines[k]:
                            current_run['lift_coarse'] = float(lines[k+1].split()[1])
                            current_run['lift_true'] = float(lines[k+2].split()[2])  # Fine (Interp)
                            current_run['lift_pred'] = float(lines[k+3].split()[1])
                            forces.append(current_run)
                            break
                        k += 1
                    break
                j += 1
    i += 1

df_forces = pd.DataFrame(forces)
print(f"Extracted {len(df_forces)} force measurements from {DATA_TYPE.upper().replace('_', ' ')} data")
print(f"Input file: {TEST_OUTPUT_FILE}")

# Get frontal area and coefficients for each case
frontal_areas = []
body_heights = []
body_widths = []
cd_true_file = []
cl_true_file = []
cd_coarse_file = []
cl_coarse_file = []

for run in df_forces['run']:
    # Read geometry file for this specific run
    geo_file = f'/data/ahmed_data/organized/coefficients/fine/geo_parameters_{run}.csv'
    geo_df = pd.read_csv(geo_file)
    
    # Extract dimensions (in mm)
    body_height = geo_df['body-height'].iloc[0]
    body_width = geo_df['body-width'].iloc[0]
    
    # Calculate frontal area (convert mm² to m²)
    frontal_area = (body_height * body_width) / 1e6
    
    body_heights.append(body_height)
    body_widths.append(body_width)
    frontal_areas.append(frontal_area)
    
    # Read FINE coefficient file
    fine_coeff_file = f'/data/ahmed_data/organized/coefficients/fine/force_mom_varref_{run}.csv'
    fine_coeff_df = pd.read_csv(fine_coeff_file)
    cd_true_file.append(fine_coeff_df['cd'].iloc[0])
    cl_true_file.append(fine_coeff_df['cl'].iloc[0])
    
    # Read COARSE coefficient file - path depends on data type
    if DATA_TYPE == 'test_on_train':
        coarse_coeff_file = f'/data/ahmed_data/organized/test_on_train/coarse/run_{run}/coefficients.csv'
    else:
        coarse_coeff_file = f'/data/ahmed_data/organized/{DATA_TYPE}/coarse/run_{run}/coefficients.csv'
    
    coarse_coeff_df = pd.read_csv(coarse_coeff_file)
    cd_coarse_file.append(coarse_coeff_df['cd'].iloc[0])
    cl_coarse_file.append(coarse_coeff_df['cl'].iloc[0])

# Add to dataframe
df_forces['body_height_mm'] = body_heights
df_forces['body_width_mm'] = body_widths
df_forces['frontal_area_m2'] = frontal_areas
df_forces['Cd_true_file'] = cd_true_file
df_forces['Cl_true_file'] = cl_true_file
df_forces['Cd_coarse_file'] = cd_coarse_file
df_forces['Cl_coarse_file'] = cl_coarse_file

# Calculate coefficients using actual frontal areas
# For non-dimensionalized: rho=1, V=1, so Cd = 2*F/A
df_forces['Cd_pred_calc'] = 2 * df_forces['drag_pred'] / df_forces['frontal_area_m2']
df_forces['Cd_true_calc'] = 2 * df_forces['drag_true'] / df_forces['frontal_area_m2']
df_forces['Cl_pred_calc'] = 2 * df_forces['lift_pred'] / df_forces['frontal_area_m2']
df_forces['Cl_true_calc'] = 2 * df_forces['lift_true'] / df_forces['frontal_area_m2']

# Calculate errors
# Predicted vs True (file)
df_forces['Cd_pred_error_%'] = 100 * (df_forces['Cd_pred_calc'] - df_forces['Cd_true_file']) / df_forces['Cd_true_file']
df_forces['Cl_pred_error_%'] = 100 * (df_forces['Cl_pred_calc'] - df_forces['Cl_true_file']) / df_forces['Cl_true_file']

# Coarse (file) vs True (file)
df_forces['Cd_coarse_error_%'] = 100 * (df_forces['Cd_coarse_file'] - df_forces['Cd_true_file']) / df_forces['Cd_true_file']
df_forces['Cl_coarse_error_%'] = 100 * (df_forces['Cl_coarse_file'] - df_forces['Cl_true_file']) / df_forces['Cl_true_file']

print(f"\n=== {DATA_TYPE.upper().replace('_', ' ')} DATA COEFFICIENT COMPARISON RESULTS ===")
print(f"\nFrontal area statistics:")
print(f"  Min: {df_forces['frontal_area_m2'].min():.6f} m²")
print(f"  Max: {df_forces['frontal_area_m2'].max():.6f} m²")
print(f"  Mean: {df_forces['frontal_area_m2'].mean():.6f} m²")

print("\nFirst 10 cases comparison:")
print("   Run  | Cd_pred_calc | Cd_true_file | Cd_coarse_file | Pred_Error% | Coarse_Error%")
print("-" * 85)
for i in range(min(10, len(df_forces))):
    row = df_forces.iloc[i]
    print(f"   {int(row['run']):3d}  | {row['Cd_pred_calc']:11.4f} | {row['Cd_true_file']:11.4f} | "
          f"{row['Cd_coarse_file']:13.4f} | {row['Cd_pred_error_%']:10.1f}% | {row['Cd_coarse_error_%']:12.1f}%")

print(f"\n=== OVERALL STATISTICS ({DATA_TYPE.upper().replace('_', ' ')} DATA) ===")
print(f"DRAG COEFFICIENT (Cd):")
print(f"  Predicted Mean Absolute Error: {df_forces['Cd_pred_error_%'].abs().mean():.2f}%")
print(f"  Coarse Mean Absolute Error:    {df_forces['Cd_coarse_error_%'].abs().mean():.2f}%")
print(f"  Predicted RMSE: {np.sqrt(np.mean((df_forces['Cd_pred_calc'] - df_forces['Cd_true_file'])**2)):.4f}")
print(f"  Coarse RMSE:    {np.sqrt(np.mean((df_forces['Cd_coarse_file'] - df_forces['Cd_true_file'])**2)):.4f}")

print(f"\nLIFT COEFFICIENT (Cl):")
print(f"  Predicted Mean Absolute Error: {df_forces['Cl_pred_error_%'].abs().mean():.2f}%")
print(f"  Coarse Mean Absolute Error:    {df_forces['Cl_coarse_error_%'].abs().mean():.2f}%")
print(f"  Predicted RMSE: {np.sqrt(np.mean((df_forces['Cl_pred_calc'] - df_forces['Cl_true_file'])**2)):.4f}")
print(f"  Coarse RMSE:    {np.sqrt(np.mean((df_forces['Cl_coarse_file'] - df_forces['Cl_true_file'])**2)):.4f}")

# Check consistency of fine coefficients
print(f"\n=== FINE COEFFICIENT CONSISTENCY CHECK ===")
print(f"Comparing Cd_true_calc vs Cd_true_file (should be very close):")
df_forces['Cd_fine_consistency_%'] = 100 * abs(df_forces['Cd_true_calc'] - df_forces['Cd_true_file']) / df_forces['Cd_true_file']
print(f"  Mean difference: {df_forces['Cd_fine_consistency_%'].mean():.2f}%")
print(f"  Max difference:  {df_forces['Cd_fine_consistency_%'].max():.2f}%")

# Save complete results
df_forces.to_csv(OUTPUT_CSV_NAME, index=False)
print(f"\n✅ Full results saved to {OUTPUT_CSV_NAME}")
print(f"✅ CSV contains all force and coefficient data with proper frontal areas for {DATA_TYPE.upper().replace('_', ' ')} data")