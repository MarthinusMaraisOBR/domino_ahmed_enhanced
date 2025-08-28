#!/bin/bash
# quick_retrain.sh - Apply all fixes and restart training

echo "=========================================="
echo "ENHANCED DOMINO - APPLYING ALL FIXES"
echo "=========================================="

# Step 1: Backup current files
echo "1. Backing up current files..."
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp src/physics_loss.py backups/$(date +%Y%m%d_%H%M%S)/
cp src/train.py backups/$(date +%Y%m%d_%H%M%S)/
echo "   ✓ Backups created"

# Step 2: Apply fixes
echo "2. Applying code fixes..."
cp physics_loss_fixed.py src/physics_loss_fixed.py
cp train_fixed.py src/train_fixed.py
cp debug_training.py src/debug_training.py
echo "   ✓ Fixed files copied"

# Step 3: Run debugging first
echo "3. Running diagnostic checks..."
cd src
python debug_training.py
if [ $? -ne 0 ]; then
    echo "   ✗ Diagnostic checks failed - fix issues before training"
    exit 1
fi
echo "   ✓ Diagnostic checks passed"

# Step 4: Clean up old checkpoints (optional)
echo "4. Cleaning up failed training..."
read -p "   Remove old checkpoints? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p outputs/Ahmed_Dataset/old_runs
    mv outputs/Ahmed_Dataset/enhanced_* outputs/Ahmed_Dataset/old_runs/ 2>/dev/null
    echo "   ✓ Old runs moved to old_runs/"
fi

# Step 5: Create new output directory
echo "5. Creating fresh output directory..."
mkdir -p outputs/Ahmed_Dataset/enhanced_fixed_v2/models
mkdir -p outputs/Ahmed_Dataset/enhanced_fixed_v2/tensorboard
echo "   ✓ Output directories created"

# Step 6: Update config to use fixed training
echo "6. Updating configuration..."
cat > conf/config_fixed.yaml << 'EOF'
# Fixed configuration for Enhanced DoMINO
defaults:
  - config
  - _self_

exp_tag: enhanced_fixed_v2

# Enable physics-aware training
model:
  use_physics_loss: true
  enhanced_model:
    surface_input_features: 8
    debug: true
    monitor_training: true
    coarse_to_fine:
      hidden_layers: [128, 64]  # Simplified
      use_spectral: false
      use_residual: true
      dropout_rate: 0.3
      activation: gelu
      residual_weight_min: 0.8
      residual_weight_max: 1.0
      correction_weight_max: 0.2

train:
  epochs: 200
  checkpoint_interval: 10
  learning_rate: 0.00005  # Lower for stability

data_processor:
  use_enhanced_features: true
  coarse_input_dir: /data/ahmed_data/organized/train/coarse/
EOF
echo "   ✓ Configuration updated"

# Step 7: Start training with monitoring
echo "7. Starting fixed training..."
echo ""
echo "=========================================="
echo "TRAINING WILL START WITH:"
echo "  - Physics loss scale: 1e6"
echo "  - Learning rate: 5e-5"
echo "  - Gradient clipping: 1.0"
echo "  - Simplified architecture: [128, 64]"
echo "  - Weight constraints enabled"
echo "=========================================="
echo ""
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Use the fixed training script
    python train_fixed.py --config-name=config_fixed
else
    echo "Training cancelled. Run manually with:"
    echo "  cd src && python train_fixed.py --config-name=config_fixed"
fi
