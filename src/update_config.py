import yaml
import os

# Update config.yaml with proper paths
config_path = 'conf/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set correct paths
config['eval']['scaling_param_path'] = 'outputs/Ahmed_Dataset'
config['exp_tag'] = '111'
config['eval']['checkpoint_name'] = 'DoMINOEnhanced.0.299.pt'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Ensure scaling factors are in the right place
source_path = 'outputs/Ahmed_Dataset/111/surface_scaling_factors.npy'
dest_path = 'outputs/Ahmed_Dataset/surface_scaling_factors_inference.npy'

if os.path.exists(source_path) and not os.path.exists(dest_path):
    import shutil
    shutil.copy(source_path, dest_path)
    print(f"Copied scaling factors to {dest_path}")

print("Configuration updated")
