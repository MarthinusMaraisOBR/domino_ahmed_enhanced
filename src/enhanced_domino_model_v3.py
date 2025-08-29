# enhanced_domino_model_v3.py
# Enhanced DoMINO V3 with multi-task learning for coefficient prediction

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.layers.weight_norm import WeightNormLinear


class CoefficientPredictionHead(nn.Module):
    """
    Dedicated head for predicting aerodynamic coefficients from surface fields.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        output_dim: int = 2,  # Cd and Cl
        dropout_rate: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, surface_features: torch.Tensor, surface_encoding: torch.Tensor) -> torch.Tensor:
        """
        Predict coefficients from surface features.
        
        Args:
            surface_features: [batch, points, features] - Surface field predictions
            surface_encoding: [batch, points, encoding_dim] - Geometry encoding
            
        Returns:
            coefficients: [batch, 2] - Predicted Cd and Cl
        """
        # Global pooling over surface points
        surface_mean = torch.mean(surface_features, dim=1)  # [batch, features]
        surface_max = torch.max(surface_features, dim=1)[0]  # [batch, features]
        encoding_mean = torch.mean(surface_encoding, dim=1)  # [batch, encoding_dim]
        
        # Concatenate global features
        global_features = torch.cat([surface_mean, surface_max, encoding_mean], dim=-1)
        
        # Predict coefficients
        coefficients = self.network(global_features)
        
        return coefficients


class CoarseToFineModelV3(nn.Module):
    """
    V3: Enhanced neural network with coefficient prediction capability.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        encoding_dim: int = 448,
        hidden_layers: list = [256, 128],
        activation: str = "gelu",
        use_residual: bool = True,
        dropout_rate: float = 0.2,
        residual_weight_min: float = 0.7,
        residual_weight_max: float = 1.0,
        correction_weight_max: float = 0.3,
        predict_coefficients: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.predict_coefficients = predict_coefficients
        self.residual_weight_min = residual_weight_min
        self.residual_weight_max = residual_weight_max
        self.correction_weight_max = correction_weight_max
        
        print(f"[CoarseToFineModelV3] Initializing multi-task architecture:")
        print(f"  - Surface prediction: {input_dim} -> {output_dim}")
        print(f"  - Coefficient prediction: {'ENABLED' if predict_coefficients else 'DISABLED'}")
        
        # Feature extractor for coarse flow data
        self.coarse_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # Main processing network for surface fields
        fusion_input_dim = 128 + encoding_dim
        layers = []
        in_features = fusion_input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(WeightNormLinear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim
        
        self.main_network = nn.Sequential(*layers)
        
        # Output projection for surface corrections
        self.output_projection = nn.Linear(hidden_layers[-1], output_dim)
        
        # Coefficient prediction head
        # Coefficient prediction head
        if predict_coefficients:
            # Fixed: Input dimension should match what we actually concatenate
            # surface_mean (4) + surface_max (4) + encoding_mean (448) = 456
            coeff_input_dim = output_dim * 2 + encoding_dim  # 4*2 + 448 = 456
            self.coefficient_head = CoefficientPredictionHead(
                input_dim=coeff_input_dim,
                hidden_dims=[256, 128, 64],
                output_dim=2,  # Cd, Cl
                dropout_rate=dropout_rate,
                activation=activation
            )
        
        # Residual connection parameters
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.9))
            self.correction_weight = nn.Parameter(torch.tensor(0.1))
            self.output_bias = nn.Parameter(torch.zeros(output_dim))
            
            if input_dim != output_dim:
                self.residual_projection = nn.Linear(input_dim, output_dim)
                nn.init.eye_(self.residual_projection.weight)
                nn.init.zeros_(self.residual_projection.bias)
            else:
                self.residual_projection = nn.Identity()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module == getattr(self, 'residual_projection', None):
                    continue
                    
                if module == self.output_projection:
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, 
        coarse_features: torch.Tensor,
        geometry_encoding: torch.Tensor,
        return_coefficients: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with surface field and coefficient prediction.
        
        Returns:
            (fine_prediction, coefficients) if return_coefficients else (fine_prediction, None)
        """
        # Extract features from coarse data
        coarse_processed = self.coarse_feature_extractor(coarse_features)
        
        # Combine with geometry encoding
        combined_features = torch.cat([coarse_processed, geometry_encoding], dim=-1)
        
        # Process through main network
        processed = self.main_network(combined_features)
        
        # Generate correction
        correction = self.output_projection(processed)
        
        # Apply residual connection
        if self.use_residual:
            # Constrain weights
            residual_w = torch.sigmoid(self.residual_weight) * (self.residual_weight_max - self.residual_weight_min) + self.residual_weight_min
            correction_w = torch.sigmoid(self.correction_weight) * self.correction_weight_max
            
            # Project input if needed
            residual_features = self.residual_projection(coarse_features)
            
            # Weighted combination
            fine_prediction = (
                residual_w * residual_features + 
                correction_w * correction +
                self.output_bias
            )
            
            self.actual_residual_weight = residual_w.detach()
            self.actual_correction_weight = correction_w.detach()
        else:
            fine_prediction = correction
            self.actual_residual_weight = torch.tensor(0.0)
            self.actual_correction_weight = torch.tensor(1.0)
        
        # Predict coefficients if requested
        coefficients = None
        if self.predict_coefficients and return_coefficients:
            # Concatenate features for coefficient prediction
            coeff_features = torch.cat([
                fine_prediction,  # Predicted surface fields
                coarse_processed,  # Processed coarse features
                geometry_encoding  # Geometry encoding
            ], dim=-1)
            
            coefficients = self.coefficient_head(fine_prediction, geometry_encoding)
        
        return fine_prediction, coefficients


class DoMINOEnhancedV3(DoMINO):
    """
    V3: Enhanced DoMINO with multi-task learning for surface fields and coefficients.
    """
    
    def __init__(
        self,
        input_features: int = 3,
        output_features_vol: Optional[int] = None,
        output_features_surf: Optional[int] = None,
        model_parameters: Optional[Dict] = None,
    ):
        # Convert dict to DictConfig if needed
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
        self.predict_coefficients = enhanced_config.get("predict_coefficients", True)
        
        if self.use_enhanced_features and output_features_surf is not None:
            print("\n" + "="*60)
            print("Initializing DoMINOEnhancedV3 with Multi-Task Learning")
            print("="*60)
            
            # Coarse-to-fine model configuration
            coarse_to_fine_config = enhanced_config.get("coarse_to_fine", {})
            encoding_dim = 448  # Based on your configuration
            
            # Initialize V3 model with coefficient prediction
            self.coarse_to_fine_model = CoarseToFineModelV3(
                input_dim=4,
                output_dim=output_features_surf,
                encoding_dim=encoding_dim,
                hidden_layers=coarse_to_fine_config.get("hidden_layers", [256, 128]),
                activation=model_parameters.get("activation", "gelu"),
                use_residual=coarse_to_fine_config.get("use_residual", True),
                dropout_rate=coarse_to_fine_config.get("dropout_rate", 0.2),
                residual_weight_min=coarse_to_fine_config.get("residual_weight_min", 0.7),
                residual_weight_max=coarse_to_fine_config.get("residual_weight_max", 1.0),
                correction_weight_max=coarse_to_fine_config.get("correction_weight_max", 0.3),
                predict_coefficients=self.predict_coefficients,
            )
            
            # Store coefficient predictions
            self.last_coefficients = None
            
            # Debug and monitoring flags
            self.debug_mode = enhanced_config.get("debug", False)
            self.monitor_training = enhanced_config.get("monitor_training", True)
            
            print(f"\nDoMINOEnhancedV3 initialized:")
            print(f"  - Surface field prediction: ENABLED")
            print(f"  - Coefficient prediction: {'ENABLED' if self.predict_coefficients else 'DISABLED'}")
            print("="*60 + "\n")
            
    def forward(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with surface and coefficient predictions."""
        # Handle volume predictions
        if self.output_features_vol is not None:
            vol_predictions = super().forward(inputs_dict)[0]
        else:
            vol_predictions = None
        
        # Handle surface predictions
        if self.output_features_surf is not None and self.use_enhanced_features:
            surf_predictions = self._forward_surface_enhanced_v3(inputs_dict)
        elif self.output_features_surf is not None:
            surf_predictions = super().forward(inputs_dict)[1]
        else:
            surf_predictions = None
            
        return vol_predictions, surf_predictions
    
    def _forward_surface_enhanced_v3(self, inputs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced surface forward pass with coefficient prediction."""
        
        # Get surface fields
        surface_fields = inputs_dict["surface_fields"]
        
        # Determine mode and extract features
        if surface_fields.shape[-1] == 8:
            # Training mode: 8 features [fine(0:4), coarse(4:8)]
            fine_features = surface_fields[..., :4]
            coarse_features = surface_fields[..., 4:8]
        elif surface_fields.shape[-1] == 4:
            # Inference mode: 4 features (coarse only)
            coarse_features = surface_fields
            fine_features = None
        else:
            raise ValueError(f"Unexpected surface_fields shape: {surface_fields.shape}")
        
        # Get geometry encoding (same as V2)
        geometry_stl = inputs_dict["geometry_coordinates"]
        
        # Process geometry through encoder
        if "surf_grid" in inputs_dict and "sdf_surf_grid" in inputs_dict:
            surf_min_max = inputs_dict.get("surface_min_max")
            if surf_min_max is not None:
                surf_min = surf_min_max[:, 0:1, :]
                surf_max = surf_min_max[:, 1:2, :]
            else:
                batch_size = geometry_stl.shape[0]
                surf_min = torch.zeros((batch_size, 1, 3), device=geometry_stl.device)
                surf_max = torch.ones((batch_size, 1, 3), device=geometry_stl.device)
            
            geometry_stl_normalized = 2.0 * (geometry_stl - surf_min) / (surf_max - surf_min + 1e-8) - 1.0
            
            encoding_g_surf = self.geo_rep_surface(
                geometry_stl_normalized,
                inputs_dict["surf_grid"],
                inputs_dict["sdf_surf_grid"]
            )
        else:
            if "grid" in inputs_dict and "sdf_grid" in inputs_dict:
                vol_min_max = inputs_dict.get("volume_min_max")
                if vol_min_max is not None:
                    vol_min = vol_min_max[:, 0:1, :]
                    vol_max = vol_min_max[:, 1:2, :]
                else:
                    batch_size = geometry_stl.shape[0]
                    vol_min = torch.zeros((batch_size, 1, 3), device=geometry_stl.device)
                    vol_max = torch.ones((batch_size, 1, 3), device=geometry_stl.device)
                
                geometry_stl_normalized = 2.0 * (geometry_stl - vol_min) / (vol_max - vol_min + 1e-8) - 1.0
                
                encoding_g_surf = self.geo_rep_volume(
                    geometry_stl_normalized,
                    inputs_dict["grid"],
                    inputs_dict["sdf_grid"]
                )
            else:
                raise ValueError("No grid data available for geometry encoding")
        
        # Get local geometry encoding
        surface_centers = inputs_dict["surface_mesh_centers"]
        local_encoding = self.geo_encoding_local(
            encoding_g_surf,
            surface_centers,
            inputs_dict.get("surf_grid", inputs_dict.get("grid")),
            mode="surface"
        )
        
        # Predict fine features and coefficients
        predictions, coefficients = self.coarse_to_fine_model(
            coarse_features,
            local_encoding,
            return_coefficients=self.training or self.predict_coefficients
        )
        
        # Store coefficients for loss calculation
        if coefficients is not None:
            self.last_coefficients = coefficients
        
        # Scale by inlet parameters if needed
        if self.encode_parameters:
            inlet_velocity = inputs_dict.get("stream_velocity")
            air_density = inputs_dict.get("air_density")
            if inlet_velocity is not None:
                param_encoding = self.encode_params_func(inlet_velocity, air_density)
                predictions = predictions * param_encoding
        
        return predictions
    
    def get_coefficients(self) -> Optional[torch.Tensor]:
        """Get the last predicted coefficients."""
        return self.last_coefficients
    
    def get_regularization_loss(self):
        """Get regularization loss from coarse-to-fine model."""
        if hasattr(self, 'coarse_to_fine_model'):
            reg_loss = 0.0
            c2f = self.coarse_to_fine_model
            
            # Penalize if residual weight goes too low
            if c2f.actual_residual_weight < c2f.residual_weight_min:
                reg_loss += (c2f.residual_weight_min - c2f.actual_residual_weight) ** 2
            
            # Penalize if correction weight goes too high
            if c2f.actual_correction_weight > c2f.correction_weight_max:
                reg_loss += (c2f.actual_correction_weight - c2f.correction_weight_max) ** 2
            
            # Penalize large bias values
            if hasattr(c2f, 'output_bias'):
                reg_loss += 0.01 * torch.mean(c2f.output_bias ** 2)
            
            return reg_loss
        return torch.tensor(0.0)
