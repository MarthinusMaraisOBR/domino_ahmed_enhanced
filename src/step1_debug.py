# Add this debug code to your training data
import numpy as np

# Check a training sample
train_sample = train_dataset[0]
surface_fields = train_sample['surface_fields']

print(f"Training data shape: {surface_fields.shape}")
print(f"Enhanced features enabled: {cfg.data_processor.get('use_enhanced_features', False)}")

if surface_fields.shape[1] == 8:
    fine_fields = surface_fields[:4]
    coarse_fields = surface_fields[4:]
    print(f"Fine pressure range: [{fine_fields[0].min():.3f}, {fine_fields[0].max():.3f}]")
    print(f"Coarse pressure range: [{coarse_fields[0].min():.3f}, {coarse_fields[0].max():.3f}]")
else:
    print(f"Standard training - only {surface_fields.shape[1]} features")
