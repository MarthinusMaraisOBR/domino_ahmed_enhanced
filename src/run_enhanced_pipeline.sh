#!/bin/bash
# Complete pipeline for Enhanced DoMINO training
# Run this script from the src/ directory

echo "================================================"
echo "Enhanced DoMINO Training Pipeline"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the src/ directory"
    exit 1
fi

# Step 1: Verify data structure
echo ""
echo "Step 1: Verifying data structure..."
echo "------------------------------------"
python inspect_vtp.py

echo ""
echo "Press Enter to continue or Ctrl+C to stop..."
read

# Step 2: Test interpolation quality
echo ""
echo "Step 2: Testing interpolation quality..."
echo "----------------------------------------"
python test_interpolation_robust.py

echo ""
echo "Press Enter to continue or Ctrl+C to stop..."
read

# Step 3: Process data with enhanced features
echo ""
echo "Step 3: Processing data with enhanced features..."
echo "-------------------------------------------------"
python process_data.py

# Check if processing was successful
if [ $? -ne 0 ]; then
    echo "Error: Data processing failed!"
    exit 1
fi

echo ""
echo "Data processing complete. Check the output directory for .npy files."
echo "Press Enter to continue or Ctrl+C to stop..."
read

# Step 4: Optional - Cache data for faster training
echo ""
echo "Step 4: Would you like to cache data for faster training? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Caching data..."
    python cache_data.py
fi

# Step 5: Start training
echo ""
echo "Step 5: Starting Enhanced DoMINO training..."
echo "--------------------------------------------"
echo "Training will run for many epochs. You can monitor progress with:"
echo "  tensorboard --logdir=outputs/Ahmed_Dataset/1/tensorboard"
echo ""
echo "Press Enter to start training or Ctrl+C to stop..."
read

# Run training with enhanced features
python train.py

echo ""
echo "================================================"
echo "Training Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Check tensorboard for training metrics"
echo "2. Run testing with: python test_enhanced.py"
echo "3. Visualize results in ParaView"
