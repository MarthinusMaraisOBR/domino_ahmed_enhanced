# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DoMINO Enhanced model for coarse-to-fine flow prediction.
This model learns to predict high-fidelity surface fields using only
low-fidelity (coarse) surface data and geometry as input.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.layers.weight_norm import WeightNormLinear


class FullyConnected(nn.Module):
    """Simple fully connected network"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int,
        layer_size: int,
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, layer_size))
        layers.append(nn.LayerNorm(layer_size))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "silu":
            layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LayerNorm(layer_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(layer_size, out_features))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SpectralFeatureExtractor(nn.Module):
    """
    Simple spectral feature extractor using sinusoidal embeddings.
    """
    
    def __init__(self, in_features: int, out_features: int, num_frequencies: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.randn(num_frequencies, in_features))
        
        # Output projection
        self.output_proj = nn.Linear(num_frequencies * 2, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral transformation."""
        # x shape: (B, N, in_features)
        # Compute frequencies
        freqs = torch.matmul(x, self.freq_weights.T)  # (B, N, num_frequencies)
        
        # Apply sin and cos
        sin_features = torch.sin(freqs)
        cos_features = torch.cos(freqs)
        
        # Concatenate
        spectral_features = torch.cat([sin_features, cos_features], dim=-1)
        
        # Project to output dimension
        return self.output_proj(spectral_features)


class CoarseToFineModel(nn.Module):
    """
    Neural network that learns to map coarse resolution flow fields
    to fine resolution predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # Coarse flow features
        output_dim: int = 4,  # Fine flow features
        encoding_dim: int = 448,  # Geometry encoding dimension (actual output from DoMINO)
        hidden_layers: list = [512, 512, 512],
        activation: str = "relu",
        use_spectral: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # Feature extractor for coarse flow data
        self.coarse_feature_extractor = FullyConnected(
            in_features=input_dim,
            out_features=256,
            num_layers=3,
            layer_size=256,
            activation=activation,
        )
        
        # Spectral feature extraction (helps capture multi-scale patterns)
        if use_spectral:
            self.spectral_layer = SpectralFeatureExtractor(
                in_features=input_dim,
                out_features=128,
                num_frequencies=16,
            )
            feature_dim = 256 + 128  # Regular + spectral features
        else:
            self.spectral_layer = None
            feature_dim = 256
        
        # Combine with geometry encoding
        fusion_input_dim = feature_dim + encoding_dim
        
        # Main processing network
        layers = []
        in_features = fusion_input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(WeightNormLinear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            
            in_features = hidden_dim
        
        self.main_network = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            WeightNormLinear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.LayerNorm(hidden_layers[-1] // 2),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            WeightNormLinear(hidden_layers[-1] // 2, output_dim),
        )
        
        # Residual connection from coarse to fine
        if use_residual:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        
    def forward(
        self, 
        coarse_features: torch.Tensor,
        geometry_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map coarse features to fine features using geometry information.
        
        Args:
            coarse_features: Coarse resolution features (B, N, 4)
            geometry_encoding: Geometry encoding (B, N, encoding_dim)
            
        Returns:
            Fine resolution predictions (B, N, 4)
        """
        # Extract features from coarse data
        coarse_processed = self.coarse_feature_extractor(coarse_features)
        
        # Apply spectral decomposition if enabled
        if self.spectral_layer is not None:
            spectral_features = self.spectral_layer(coarse_features)
            coarse_processed = torch.cat([coarse_processed, spectral_features], dim=-1)
        
        # Combine with geometry
        combined_features = torch.cat([coarse_processed, geometry_encoding], dim=-1)
        
        # Process through main network
        processed = self.main_network(combined_features)
        
        # Generate fine resolution output
        fine_prediction = self.output_projection(processed)
        
        # Add residual if enabled (helps preserve coarse structure)
        if self.use_residual:
            residual = self.residual_projection(coarse_features)
            fine_prediction = fine_prediction + residual
        
        return fine_prediction


class DoMINOEnhanced(DoMINO):
    """
    Enhanced DoMINO model that predicts fine resolution surface fields
    from coarse resolution input data and geometry.
    
    Training mode: Uses coarse features as input, fine features as target
    Inference mode: Takes only coarse data and geometry to predict fine resolution
    """
    
    def __init__(
        self,
        input_features: int = 3,
        output_features_vol: Optional[int] = None,
        output_features_surf: Optional[int] = None,
        model_parameters: Optional[Dict] = None,
    ):
        # Initialize base DoMINO model
        super().__init__(
            input_features=input_features,
            output_features_vol=output_features_vol,
            output_features_surf=output_features_surf,
            model_parameters=model_parameters,
        )
        
        # Check if enhanced features are enabled
        self.use_enhanced_features = model_parameters.get("enhanced_model", {}).get(
            "surface_input_features", 4
        ) > 4
        
        if self.use_enhanced_features and output_features_surf is not None:
            print("Initializing DoMINOEnhanced for coarse-to-fine prediction")
            
            # Coarse-to-fine model configuration
            coarse_to_fine_config = model_parameters.get("enhanced_model", {}).get(
                "coarse_to_fine", {}
            )
            
            # The actual geometry encoding dimension from DoMINO's local encoder
            # Based on the error message, this is 448, not 512
            # The encoding comes from geo_encoding_local which outputs 448 dimensions
            # for the current model configuration
            encoding_dim = 448
            
            # Initialize coarse-to-fine model
            self.coarse_to_fine_model = CoarseToFineModel(
                input_dim=4,  # Coarse surface features
                output_dim=output_features_surf,  # Fine surface features
                encoding_dim=encoding_dim,  # Actual geometry encoding size from DoMINO
                hidden_layers=coarse_to_fine_config.get("hidden_layers", [512, 512, 512]),
                activation=model_parameters.get("activation", "relu"),
                use_spectral=coarse_to_fine_config.get("use_spectral", True),
                use_residual=coarse_to_fine_config.get("use_residual", True),
            )
            
            # Training mode flag
            self.enhanced_training_mode = True
            
            print(f"DoMINOEnhanced initialized:")
            print(f"  - Input: 4 coarse features + geometry")
            print(f"  - Output: {output_features_surf} fine features")
            print(f"  - Geometry encoding: {encoding_dim} dimensions")
            print(f"  - Use spectral: {coarse_to_fine_config.get('use_spectral', True)}")
            print(f"  - Use residual: {coarse_to_fine_config.get('use_residual', True)}")
            
    def forward(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for enhanced model.
        
        In training mode with enhanced features:
        - surface_fields contains 8 features: [fine_features, coarse_features]
        - Use coarse_features as input to predict fine_features
        
        In inference mode:
        - surface_fields contains 4 features: [coarse_features]
        - Predict fine features from coarse
        
        Args:
            inputs_dict: Dictionary containing model inputs
            
        Returns:
            Tuple of (volume_predictions, surface_predictions)
        """
        
        # Handle volume predictions normally
        if self.output_features_vol is not None:
            # Standard volume processing - use parent class method
            vol_predictions = super().forward(inputs_dict)[0]
        else:
            vol_predictions = None
        
        # Handle surface predictions
        if self.output_features_surf is not None and self.use_enhanced_features:
            # Enhanced surface processing for coarse-to-fine prediction
            surf_predictions = self._forward_surface_enhanced(inputs_dict)
        elif self.output_features_surf is not None:
            # Standard surface processing - use parent class method
            surf_predictions = super().forward(inputs_dict)[1]
        else:
            surf_predictions = None
            
        return vol_predictions, surf_predictions
    
    def _forward_surface_enhanced(self, inputs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Enhanced surface forward pass for coarse-to-fine prediction.
        """
        
        # Get surface fields
        surface_fields = inputs_dict["surface_fields"]
        
        # Determine if we're in training or inference mode
        if surface_fields.shape[-1] == 8:
            # Training mode: we have both fine and coarse features
            fine_features = surface_fields[..., :4]
            coarse_features = surface_fields[..., 4:8]
            self.enhanced_training_mode = True
        else:
            # Inference mode: we only have coarse features
            coarse_features = surface_fields
            fine_features = None
            self.enhanced_training_mode = False
        
        # Get geometry encoding using parent class methods
        geometry_stl = inputs_dict["geometry_coordinates"]
        
        # Process geometry through encoder (from parent class)
        if "grid" in inputs_dict and "sdf_grid" in inputs_dict:
            vol_min = inputs_dict.get("volume_min_max", torch.zeros((1, 2, 3)))[:, 0]
            vol_max = inputs_dict.get("volume_min_max", torch.ones((1, 2, 3)))[:, 1]
            geometry_stl_normalized = 2.0 * (geometry_stl - vol_min) / (vol_max - vol_min) - 1.0
            
            encoding_g_vol = self.geo_rep_volume(
                geometry_stl_normalized, 
                inputs_dict["grid"], 
                inputs_dict["sdf_grid"]
            )
        else:
            encoding_g_vol = None
            
        if "surf_grid" in inputs_dict and "sdf_surf_grid" in inputs_dict:
            surf_min = inputs_dict.get("surface_min_max", torch.zeros((1, 2, 3)))[:, 0]
            surf_max = inputs_dict.get("surface_min_max", torch.ones((1, 2, 3)))[:, 1]
            geometry_stl_normalized_surf = 2.0 * (geometry_stl - surf_min) / (surf_max - surf_min) - 1.0
            
            encoding_g_surf = self.geo_rep_surface(
                geometry_stl_normalized_surf,
                inputs_dict["surf_grid"],
                inputs_dict["sdf_surf_grid"]
            )
        else:
            encoding_g_surf = None
        
        # Combine geometry encodings
        if encoding_g_vol is not None and encoding_g_surf is not None:
            geometry_encoding = 0.5 * encoding_g_vol + 0.5 * encoding_g_surf
        elif encoding_g_surf is not None:
            geometry_encoding = encoding_g_surf
        else:
            geometry_encoding = encoding_g_vol
        
        # Get local geometry encoding for surface points
        surface_centers = inputs_dict["surface_mesh_centers"]
        local_encoding = self.geo_encoding_local(
            geometry_encoding,
            surface_centers,
            inputs_dict.get("surf_grid", inputs_dict.get("grid")),
            mode="surface"
        )
        
        # Predict fine features from coarse
        predictions = self.coarse_to_fine_model(
            coarse_features,
            local_encoding
        )
        
        # Scale by inlet parameters if needed
        if self.encode_parameters:
            inlet_velocity = inputs_dict.get("stream_velocity")
            air_density = inputs_dict.get("air_density")
            if inlet_velocity is not None:
                param_encoding = self.encode_params_func(inlet_velocity, air_density)
                predictions = predictions * param_encoding
        
        return predictions