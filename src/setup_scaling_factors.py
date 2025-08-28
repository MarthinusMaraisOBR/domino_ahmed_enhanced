# Run this to set up the correct scaling factors for inference
import numpy as np
import os

# Load the correct scaling factors identified by the investigation
correct_scales = np.load('outputs/Ahmed_Dataset/surface_scaling_factors_enhanced.npy')

print(f"Loaded scaling factors with shape: {correct_scales.shape}")

# Create inference version (fine features only for denormalization)
scales_inference = correct_scales[:, :4]

print(f"Created inference scaling factors with shape: {scales_inference.shape}")

# Save in the expected location
os.makedirs("outputs/Ahmed_Dataset/enhanced_v2_physics", exist_ok=True)

# Save inference version
inf_path = "outputs/Ahmed_Dataset/enhanced_v2_physics/surface_scaling_factors_inference.npy"
np.save(inf_path, scales_inference)
print(f"✅ Saved inference scaling factors to: {inf_path}")

# Save full version for reference
full_path = "outputs/Ahmed_Dataset/enhanced_v2_physics/surface_scaling_factors.npy"
np.save(full_path, correct_scales)
print(f"✅ Saved full scaling factors to: {full_path}")

# Verify the values match what we found
print("\nVerification:")
print("Fine Features (for output denormalization):")
for i in range(4):
    print(f"  Feature {i}: min={scales_inference[1,i]:.6f}, max={scales_inference[0,i]:.6f}")

print("\n✅ Scaling factors set up for inference!")
