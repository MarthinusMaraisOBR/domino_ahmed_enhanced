#!/bin/bash
# Script to start training with the fixed Enhanced DoMINO model

echo "=========================================="
echo "STARTING FIXED ENHANCED DOMINO TRAINING"
echo "=========================================="

# Step 1: Backup existing model
if [ -f "enhanced_domino_model.py" ]; then
    echo "Backing up existing model..."
    cp enhanced_domino_model.py enhanced_domino_model_original.py
    echo "✅ Backed up to enhanced_domino_model_original.py"
fi

# Step 2: Deploy fixed model
if [ -f "enhanced_domino_model_fixed.py" ]; then
    echo "Deploying fixed model..."
    cp enhanced_domino_model_fixed.py enhanced_domino_model.py
    echo "✅ Fixed model deployed"
else
    echo "❌ Fixed model file not found!"
    echo "Please ensure enhanced_domino_model_fixed.py exists"
    exit 1
fi

# Step 3: Backup existing config
if [ -f "conf/config.yaml" ]; then
    echo "Backing up existing config..."
    cp conf/config.yaml conf/config_original.yaml
    echo "✅ Config backed up to conf/config_original.yaml"
fi

# Step 4: Deploy fixed config
if [ -f "conf/config_enhanced_fixed.yaml" ]; then
    echo "Deploying fixed config..."
    cp conf/config_enhanced_fixed.yaml conf/config.yaml
    echo "✅ Fixed config deployed"
else
    echo "⚠️ Fixed config not found, using existing config"
    echo "Make sure to update config.yaml with the fixes!"
fi

# Step 5: Clean up old experiment (optional)
echo ""
echo "Do you want to clean old experiment data? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Cleaning old experiment data..."
    rm -rf outputs/Ahmed_Dataset/enhanced_fixed/
    echo "✅ Old data cleaned"
fi

# Step 6: Create necessary directories
echo "Creating output directories..."
mkdir -p outputs/Ahmed_Dataset/enhanced_fixed/models
mkdir -p outputs/Ahmed_Dataset/enhanced_fixed/tensorboard
echo "✅ Directories created"

# Step 7: Start training with monitoring
echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""
echo "Training will start in 5 seconds..."
echo "Press Ctrl+C to cancel"
echo ""
echo "Monitor training with:"
echo "  tensorboard --logdir=outputs/Ahmed_Dataset/enhanced_fixed/tensorboard"
echo ""
sleep 5

# Start training with output logging
echo "Starting training at $(date)"
echo "Logging to: train_fixed.log"
echo ""

# Run training with tee to see output and save to log
python train.py 2>&1 | tee train_fixed.log

echo ""
echo "Training completed at $(date)"
echo "Check train_fixed.log for full output"
