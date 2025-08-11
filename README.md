# DoMINO: Decomposable Multi-scale Iterative Neural Operator for External Aerodynamics

DoMINO is a local, multi-scale, point-cloud based model architecture to model large-scale
physics problems such as external aerodynamics. The DoMINO model architecture takes STL
geometries as input and evaluates flow quantities such as pressure and
wall shear stress on the surface of the car as well as velocity fields and pressure
in the volume around it. The DoMINO architecture is designed to be a fast, accurate
and scalable surrogate model for large-scale industrial simulations.

A preprint describing additional details about the model architecture can be found here
[paper](https://arxiv.org/abs/2501.13350).

## Enhanced DoMINO: Coarse-to-Fine Prediction

This repository implements an **Enhanced DoMINO** variant that enables high-fidelity predictions
using only coarse resolution input data:

### Standard vs Enhanced DoMINO

| Aspect | Standard DoMINO | Enhanced DoMINO |
|--------|-----------------|-----------------|
| **Training Input** | Fine CFD data | Fine CFD + Coarse RANS data |
| **Inference Input** | Fine CFD data | **Only** Coarse RANS data |
| **Output** | Fine predictions | Fine predictions |
| **Use Case** | Fast surrogate for CFD | Super-resolution from cheap RANS |

The Enhanced DoMINO learns a physics-informed mapping from coarse to fine resolution,
enabling accurate predictions without expensive CFD simulations at inference time.

Getting Started with Enhanced DoMINO on Ahmed Body Dataset
Configuration
DoMINO uses YAML configuration files powered by Hydra. The base configuration
config.yaml is located in src/conf directory.
Key configuration settings for Enhanced DoMINO:
yamldata_processor:
  use_enhanced_features: true  # Enable dual-resolution mode
  coarse_input_dir: /data/ahmed_data_rans/raw/  # Coarse RANS data

variables:
  surface:
    enhanced_features:
      input_feature_count: 8  # 4 fine + 4 coarse
      coarse_variable_mapping:  # Map coarse to fine variable names
        pMean: p
        wallShearStressMean: wallShearStress

model:
  enhanced_model:
    surface_input_features: 8
Data Processing

Prepare dual-resolution data:

Fine resolution CFD data: /data/ahmed_data/raw/
Coarse resolution RANS data: /data/ahmed_data_rans/raw/


Process the data:
bashpython process_data.py  # Creates 8-feature surface fields
The enhanced dataset concatenates fine and interpolated coarse features.

Training the Enhanced Model

Verify configuration:

Ensure use_enhanced_features: true in config.yaml
Check data paths for both fine and coarse data


Run training:
bashpython train.py
The model learns to predict fine features from coarse features during training.
Monitor the "improvement over coarse baseline" metric in tensorboard.
Training characteristics:

Input: 8 features (4 fine + 4 coarse)
Target: 4 fine features
Loss: Computed between predicted and actual fine features



Testing and Inference
For inference, the trained model requires only coarse data:
bashpython test.py  # Uses only coarse resolution input
The model will predict fine resolution surface fields from coarse input,
enabling fast, accurate predictions without expensive CFD simulations.
Visualizing Results
Download the predictions (.vtp format) and visualize in ParaView to compare:

Coarse input data
Fine predictions from Enhanced DoMINO
Ground truth fine CFD (if available)

Key Implementation Files

enhanced_domino_model.py: Coarse-to-fine model architecture
openfoam_datapipe.py: Enhanced dataset with interpolation
train.py: Modified training loop for enhanced features
test_enhanced_pipeline.py: Validation script
