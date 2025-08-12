# Enhanced DoMINO Quick Start Guide

## 🚀 What You're Building

You're creating a physics-informed neural network that learns to predict high-fidelity CFD results from coarse RANS simulations. Think of it as a "super-resolution" model for fluid dynamics.

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [x] Fine resolution CFD data in `/data/ahmed_data/raw/`
- [x] Coarse resolution RANS data in `/data/ahmed_data_rans/raw/`
- [x] At least one GPU with 16GB+ memory
- [x] Docker container with PhysicsNeMo installed

## 🔧 Step-by-Step Instructions

### Step 1: Verify Your Setup
```bash
cd /workspace/PhysicsNeMo/examples/cfd/external_aerodynamics/domino/src
python verify_setup.py
```
This checks if everything is properly configured.

### Step 2: Inspect Your Data
```bash
python inspect_vtp.py
```
This shows what variables are in your VTP files. Note the exact names!

### Step 3: Update Configuration
Edit `conf/config.yaml`:
```yaml
data_processor:
  use_enhanced_features: true  # MUST be true
  input_dir: /data/ahmed_data/raw/
  coarse_input_dir: /data/ahmed_data_rans/raw/

variables:
  surface:
    solution:
      # Use the EXACT names from inspect_vtp.py
      pMean: scalar  # or "p" if that's what you see
      wallShearStressMean: vector  # or "wallShearStress"
    
    enhanced_features:
      input_feature_count: 8  # MUST be 8
      coarse_variable_mapping:
        pMean: p  # Map fine name to coarse name
        wallShearStressMean: wallShearStress

model:
  enhanced_model:
    surface_input_features: 8  # MUST be 8
```

### Step 4: Test Interpolation Quality
```bash
python test_interpolation_robust.py
```
This checks if coarse-to-fine interpolation works well. Look for <10% error.

### Step 5: Process Data
```bash
python process_data.py
```
This creates NPY files with 8 features (4 fine + 4 coarse interpolated).

### Step 6: Start Training
```bash
python train.py
```
Or use multiple GPUs:
```bash
torchrun --nproc_per_node=4 train.py
```

### Step 7: Monitor Training
In a new terminal:
```bash
tensorboard --logdir=outputs/Ahmed_Dataset/1/tensorboard
```
Open browser to `http://localhost:6006`

### Step 8: Test the Model
After training (or using a checkpoint):
```bash
python test_enhanced.py
```

## 📊 What to Expect

### During Training
- **Loss should decrease** from ~1e-2 to ~1e-4
- **Improvement metric** should be positive (shows beating baseline)
- **Training time**: ~2-4 hours for 500 epochs on 1 GPU

### Good Signs
- ✅ Loss decreasing steadily
- ✅ Improvement > 20% over coarse baseline
- ✅ Validation loss following training loss

### Warning Signs
- ⚠️ Loss exploding or NaN
- ⚠️ Improvement negative (worse than interpolation)
- ⚠️ Validation loss increasing (overfitting)

## 🔍 Understanding the Output

The model predicts:
1. **Pressure coefficient** (scalar)
2. **Wall shear stress** (3D vector)

These are used to calculate:
- **Drag force**: Fx = ∫(p·nx - τx)dA
- **Lift force**: Fz = ∫(p·nz - τz)dA

## 🐛 Troubleshooting

### "Module not found" errors
```bash
export PYTHONPATH=/workspace/PhysicsNeMo:$PYTHONPATH
```

### "Out of memory" errors
Reduce batch size in `config.yaml`:
```yaml
train:
  dataloader:
    batch_size: 1  # Was 2 or 4
```

### "NaN loss" errors
Reduce learning rate:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Was 0.001
```

### Data not found
Check paths match exactly:
```bash
ls -la /data/ahmed_data/raw/
ls -la /data/ahmed_data_rans/raw/
```

## 📈 Expected Performance

After successful training:
- **Prediction accuracy**: 85-95% of fine CFD
- **Speed**: 100-1000x faster than CFD simulation
- **Force prediction error**: <5% for drag, <10% for lift

## 🎯 Next Steps

1. **Hyperparameter tuning**: Try different learning rates, architectures
2. **More data**: Add more training cases for better generalization
3. **Transfer learning**: Fine-tune on specific geometries
4. **Production deployment**: Export model for inference server

## 💡 Tips for CFD Experts

- The model learns the "missing physics" between RANS and LES/DNS
- Geometry encoding via SDF captures near-wall behavior
- Loss functions enforce conservation (integral losses)
- Think of it as learning universal wall functions

## 📚 Files You Modified

1. `enhanced_domino_model.py` - The coarse-to-fine neural network
2. `train.py` - Training script with enhanced support
3. `test_enhanced.py` - Testing script for coarse input only
4. `config.yaml` - Configuration with enhanced features enabled

## ✅ Success Criteria

You know it's working when:
1. Training completes without errors
2. Loss decreases over epochs
3. Test predictions show smooth fields (no noise)
4. Force coefficients match fine CFD within 10%
5. Inference uses only coarse data

Good luck with your Enhanced DoMINO training! 🚀
