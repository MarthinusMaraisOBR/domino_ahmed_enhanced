
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
FIXED DoMINO Enhanced model for coarse-to-fine flow prediction.
Key fixes applied:
1. Disabled residual connection (was causing wrong predictions)
2. Added debug logging to track data flow
3. Fixed feature extraction logic
4. Simplified architecture for better learning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

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
        dropout_rate: float = 0.0,  # Added dropout for regularization
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
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
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
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (no activation)
        layers.append(nn.Linear(layer_size, out_features))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SpectralFeatureExtractor(nn.Module):
    """
    Simple spectral feature extractor using sinusoidal embeddings.
    FIXED: Reduced complexity for better learning
    """
    
    def __init__(self, in_features: int, out_features: int, num_frequencies: int = 8):  # Reduced from 16
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        # Learnable frequency weights with better initialization
        self.freq_weights = nn.Parameter(torch.randn(num_frequencies, in_features) * 0.5)
        
        # Output projection with layer norm
        self.output_proj = nn.Sequential(
            nn.Linear(num_frequencies * 2, out_features),
            nn.LayerNorm(out_features)
        )
        
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
    FIXED Neural network that learns to map coarse resolution flow fields
    to fine resolution predictions.
    
    Major changes:
    1. Disabled residual connection by default
    2. Reduced network complexity
    3. Added dropout for regularization
    4. Better weight initialization
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # Coarse flow features
        output_dim: int = 4,  # Fine flow features  
        encoding_dim: int = 448,  # Geometry encoding dimension
        hidden_layers: list = [256, 256],  # Network architecture
        activation: str = "relu",
        use_spectral: bool = False,  # Spectral features
        use_residual: bool = True,  # FIXED: Now enabled by default with proper implementation
        dropout_rate: float = 0.1,  # Regularization
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        print(f"[CoarseToFineModel] Initializing FIXED architecture:")
        print(f"  - Input dim: {input_dim}")
        print(f"  - Output dim: {output_dim}")
        print(f"  - Encoding dim: {encoding_dim}")
        print(f"  - Hidden layers: {hidden_layers}")
        print(f"  - Use spectral: {use_spectral}")
        print(f"  - Use residual: {use_residual} {'✅ ENABLED with learnable weighting' if use_residual else '⚠️ DISABLED'}")
        
        # Feature extractor for coarse flow data
        self.coarse_feature_extractor = FullyConnected(
            in_features=input_dim,
            out_features=128,
            num_layers=2,
            layer_size=128,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        
        # Spectral feature extraction (optional)
        if use_spectral:
            self.spectral_layer = SpectralFeatureExtractor(
                in_features=input_dim,
                out_features=64,
                num_frequencies=8,
            )
            feature_dim = 128 + 64
        else:
            self.spectral_layer = None
            feature_dim = 128
        
        # Combine with geometry encoding
        fusion_input_dim = feature_dim + encoding_dim
        
        # Main processing network
        layers = []
        in_features = fusion_input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(WeightNormLinear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            
            # Add dropout to middle layers
            if i < len(hidden_layers) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim
        
        self.main_network = nn.Sequential(*layers)
        
        # CRITICAL FIX: Output projection WITHOUT Tanh activation
        # This predicts the CORRECTION/REFINEMENT, not absolute values
        self.output_projection = nn.Linear(hidden_layers[-1], output_dim)
        
        # FIXED RESIDUAL IMPLEMENTATION
        if use_residual:
            # Learnable residual weight - starts small to ensure stability
            # This will be learned during training to find optimal blend
            self.residual_weight = nn.Parameter(torch.tensor(0.9))  # Start with strong input contribution
            self.correction_weight = nn.Parameter(torch.tensor(0.1))  # Start with weak correction
            
            # Optional: Learn a projection for the residual path if dimensions don't match
            if input_dim != output_dim:
                self.residual_projection = nn.Linear(input_dim, output_dim)
                nn.init.eye_(self.residual_projection.weight)  # Initialize as identity-like
                nn.init.zeros_(self.residual_projection.bias)
            else:
                self.residual_projection = nn.Identity()
            
            print(f"  - Residual weights: input={0.9:.2f}, correction={0.1:.2f} (learnable)")
        
        # Initialize weights properly for refinement learning
        self._initialize_weights_for_refinement()
        
    def _initialize_weights_for_refinement(self):
        """Initialize weights specifically for learning refinements, not absolute predictions."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, WeightNormLinear)):
                # Skip residual projection (already initialized as identity)
                if module == getattr(self, 'residual_projection', None):
                    continue
                    
                # Initialize output projection to produce small corrections initially
                if module == self.output_projection:
                    if hasattr(module, 'weight'):
                        nn.init.xavier_uniform_(module.weight, gain=0.01)  # Very small initial corrections
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Standard initialization for other layers
                    if hasattr(module, 'weight_v'):
                        nn.init.xavier_uniform_(module.weight_v, gain=0.5)
                    elif hasattr(module, 'weight'):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, 
        coarse_features: torch.Tensor,
        geometry_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map coarse features to fine features using geometry information.
        
        FIXED APPROACH: Learn refinements/corrections rather than absolute values.
        
        Args:
            coarse_features: Coarse resolution features (B, N, 4)
            geometry_encoding: Geometry encoding (B, N, encoding_dim)
            
        Returns:
            Fine resolution predictions (B, N, 4)
        """
        # Debug logging (only first batch to avoid spam)
        if self.training and torch.rand(1).item() < 0.01:  # Log 1% of batches
            print(f"[C2F Forward] Coarse input stats: mean={coarse_features.mean():.4f}, std={coarse_features.std():.4f}")
        
        # Extract features from coarse data
        coarse_processed = self.coarse_feature_extractor(coarse_features)
        
        # Apply spectral decomposition if enabled
        if self.spectral_layer is not None:
            spectral_features = self.spectral_layer(coarse_features)
            coarse_processed = torch.cat([coarse_processed, spectral_features], dim=-1)
        
        # Combine with geometry encoding
        combined_features = torch.cat([coarse_processed, geometry_encoding], dim=-1)
        
        # Process through main network
        processed = self.main_network(combined_features)
        
        # Generate CORRECTION/REFINEMENT (not absolute prediction)
        correction = self.output_projection(processed)
        
        # CRITICAL FIX: Proper residual connection with learnable weights
        if self.use_residual:
            # Project input if needed (usually identity)
            residual_features = self.residual_projection(coarse_features)
            
            # Learnable weighted combination:
            # output = α * input + β * correction
            # where α and β are learned to balance input preservation vs refinement
            fine_prediction = (
                self.residual_weight * residual_features + 
                self.correction_weight * correction
            )
            
            # Optional: Add sigmoid gating to ensure weights stay reasonable
            # This prevents the pathological 2x scaling issue
            if self.training:
                # Clamp weights during training to prevent instability
                self.residual_weight.data = torch.clamp(self.residual_weight.data, 0.5, 1.5)
                self.correction_weight.data = torch.clamp(self.correction_weight.data, -0.5, 0.5)
        else:
            # Without residual, just use the correction as absolute prediction
            # NOT RECOMMENDED - this is what caused your 2x scaling issue
            fine_prediction = correction
            print("⚠️ WARNING: Running without residual connection - may lead to scaling issues!")
        
        # Debug logging for monitoring
        if self.training and torch.rand(1).item() < 0.01:
            print(f"  Residual weight: {self.residual_weight.item():.3f}")
            print(f"  Correction weight: {self.correction_weight.item():.3f}")
            print(f"  Correction magnitude: {correction.abs().mean():.4f}")
            print(f"  Output stats: mean={fine_prediction.mean():.4f}, std={fine_prediction.std():.4f}")
        
        return fine_prediction


class DoMINOEnhanced(DoMINO):
    """
    FIXED Enhanced DoMINO model that predicts fine resolution surface fields
    from coarse resolution input data and geometry.
    
    Key fixes:
    1. Proper feature extraction with validation
    2. Disabled residual by default
    3. Added extensive debug logging
    4. Fixed inference mode detection
    """
    
    def __init__(
        self,
        input_features: int = 3,
        output_features_vol: Optional[int] = None,
        output_features_surf: Optional[int] = None,
        model_parameters: Optional[Dict] = None,
    ):
        # Convert dict to DictConfig if needed for parent class
        if isinstance(model_parameters, dict):
            from omegaconf import DictConfig
            model_parameters = DictConfig(model_parameters)
        
        # Initialize base DoMINO model
        super().__init__(
            input_features=input_features,
            output_features_vol=output_features_vol,
            output_features_surf=output_features_surf,
            model_parameters=model_parameters,
        )
        
        # Check if enhanced features are enabled
        enhanced_config = model_parameters.get("enhanced_model", {})
        self.use_enhanced_features = enhanced_config.get("surface_input_features", 4) > 4
        
        if self.use_enhanced_features and output_features_surf is not None:
            print("\n" + "="*60)
            print("Initializing DoMINOEnhanced for coarse-to-fine prediction")
            print("="*60)
            
            # Coarse-to-fine model configuration
            coarse_to_fine_config = enhanced_config.get("coarse_to_fine", {})
            
            # Get actual geometry encoding dimension
            # This depends on the model configuration
            encoding_dim = 448  # Based on your error messages
            
            # Initialize coarse-to-fine model with FIXES
            self.coarse_to_fine_model = CoarseToFineModel(
                input_dim=4,  # Coarse surface features
                output_dim=output_features_surf,  # Fine surface features
                encoding_dim=encoding_dim,
                hidden_layers=coarse_to_fine_config.get("hidden_layers", [256, 256]),  # Reduced
                activation=model_parameters.get("activation", "relu"),
                use_spectral=coarse_to_fine_config.get("use_spectral", False),  # Disabled
                use_residual=coarse_to_fine_config.get("use_residual", False),  # CRITICAL: Disabled
                dropout_rate=coarse_to_fine_config.get("dropout_rate", 0.1),
            )
            
            # Debug flag
            self.debug_mode = enhanced_config.get("debug", False)
            self.log_counter = 0
            
            print(f"\nDoMINOEnhanced initialized:")
            print(f"  - Input: 4 coarse features + {encoding_dim}D geometry")
            print(f"  - Output: {output_features_surf} fine features")
            print(f"  - Debug mode: {self.debug_mode}")
            print("="*60 + "\n")
            
    def forward(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for enhanced model with proper feature handling.
        """
        # Handle volume predictions normally
        if self.output_features_vol is not None:
            vol_predictions = super().forward(inputs_dict)[0]
        else:
            vol_predictions = None
        
        # Handle surface predictions
        if self.output_features_surf is not None and self.use_enhanced_features:
            surf_predictions = self._forward_surface_enhanced(inputs_dict)
        elif self.output_features_surf is not None:
            surf_predictions = super().forward(inputs_dict)[1]
        else:
            surf_predictions = None
            
        return vol_predictions, surf_predictions
    
    def _forward_surface_enhanced(self, inputs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        FIXED enhanced surface forward pass for coarse-to-fine prediction.
        """
        
        # Get surface fields
        surface_fields = inputs_dict["surface_fields"]
        
        # CRITICAL: Determine mode and extract correct features
        if surface_fields.shape[-1] == 8:
            # Training mode: 8 features [fine(0:4), coarse(4:8)]
            fine_features = surface_fields[..., :4]
            coarse_features = surface_fields[..., 4:8]
            
            # Debug logging
            if self.debug_mode and self.log_counter % 100 == 0:
                print(f"\n[Training Mode]")
                print(f"  Fine features (target) - mean: {fine_features.mean():.4f}, std: {fine_features.std():.4f}")
                print(f"  Coarse features (input) - mean: {coarse_features.mean():.4f}, std: {coarse_features.std():.4f}")
            
            self.log_counter += 1
            
        elif surface_fields.shape[-1] == 4:
            # Inference mode: 4 features (coarse only)
            coarse_features = surface_fields
            fine_features = None
            
            if self.debug_mode:
                print(f"\n[Inference Mode]")
                print(f"  Coarse features (input) - mean: {coarse_features.mean():.4f}, std: {coarse_features.std():.4f}")
        else:
            raise ValueError(f"Unexpected surface_fields shape: {surface_fields.shape}")
        
        # Get geometry encoding using parent class methods
        geometry_stl = inputs_dict["geometry_coordinates"]
        
        # Process geometry through encoder
        if "surf_grid" in inputs_dict and "sdf_surf_grid" in inputs_dict:
            surf_min = inputs_dict.get("surface_min_max", torch.zeros((1, 2, 3)))[:, 0]
            surf_max = inputs_dict.get("surface_min_max", torch.ones((1, 2, 3)))[:, 1]
            geometry_stl_normalized = 2.0 * (geometry_stl - surf_min) / (surf_max - surf_min) - 1.0
            
            encoding_g_surf = self.geo_rep_surface(
                geometry_stl_normalized,
                inputs_dict["surf_grid"],
                inputs_dict["sdf_surf_grid"]
            )
        else:
            # Fallback if surface grid not available
            if "grid" in inputs_dict and "sdf_grid" in inputs_dict:
                vol_min = inputs_dict.get("volume_min_max", torch.zeros((1, 2, 3)))[:, 0]
                vol_max = inputs_dict.get("volume_min_max", torch.ones((1, 2, 3)))[:, 1]
                geometry_stl_normalized = 2.0 * (geometry_stl - vol_min) / (vol_max - vol_min) - 1.0
                
                encoding_g_surf = self.geo_rep_volume(
                    geometry_stl_normalized,
                    inputs_dict["grid"],
                    inputs_dict["sdf_grid"]
                )
            else:
                raise ValueError("No grid data available for geometry encoding")
        
        # Get local geometry encoding for surface points
        surface_centers = inputs_dict["surface_mesh_centers"]
        local_encoding = self.geo_encoding_local(
            encoding_g_surf,
            surface_centers,
            inputs_dict.get("surf_grid", inputs_dict.get("grid")),
            mode="surface"
        )
        
        # Predict fine features from coarse using FIXED model
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
        
        # Debug: log prediction statistics
        if self.debug_mode and self.log_counter % 100 == 0:
            print(f"  Predictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        
        return predictions