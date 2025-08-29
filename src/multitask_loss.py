# multitask_loss.py
# Multi-task loss function for surface field and coefficient prediction

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining surface field prediction and coefficient prediction.
    """
    
    def __init__(
        self,
        surface_weight: float = 0.7,
        coefficient_weight: float = 0.3,
        physics_weight: float = 0.1,
        use_physics_constraints: bool = True,
    ):
        super().__init__()
        
        self.surface_weight = surface_weight
        self.coefficient_weight = coefficient_weight
        self.physics_weight = physics_weight
        self.use_physics_constraints = use_physics_constraints
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print(f"MultiTaskLoss initialized:")
        print(f"  Surface weight: {surface_weight}")
        print(f"  Coefficient weight: {coefficient_weight}")
        print(f"  Physics weight: {physics_weight}")
    
    def forward(
        self,
        pred_surface: torch.Tensor,
        target_surface: torch.Tensor,
        pred_coefficients: Optional[torch.Tensor],
        target_coefficients: Optional[torch.Tensor],
        coarse_surface: Optional[torch.Tensor] = None,
        surface_areas: Optional[torch.Tensor] = None,
        surface_normals: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss.
        
        Args:
            pred_surface: Predicted surface fields [batch, points, 4]
            target_surface: Target surface fields [batch, points, 4]
            pred_coefficients: Predicted coefficients [batch, 2]
            target_coefficients: Target coefficients [batch, 2]
            coarse_surface: Coarse surface fields for baseline [batch, points, 4]
            surface_areas: Surface area weights [batch, points, 1]
            surface_normals: Surface normals [batch, points, 3]
            model: Model for regularization
            
        Returns:
            (total_loss, loss_components_dict)
        """
        batch_size = pred_surface.shape[0]
        device = pred_surface.device
        
        loss_components = {}
        total_loss = 0.0
        
        # 1. Surface field loss
        surface_loss = self._compute_surface_loss(
            pred_surface, target_surface, coarse_surface
        )
        loss_components['surface'] = surface_loss.item()
        total_loss += self.surface_weight * surface_loss
        
        # 2. Coefficient loss
        if pred_coefficients is not None and target_coefficients is not None:
            coeff_loss = self._compute_coefficient_loss(
                pred_coefficients, target_coefficients
            )
            loss_components['coefficient'] = coeff_loss.item()
            total_loss += self.coefficient_weight * coeff_loss
        
        # 3. Physics-based consistency loss
        if self.use_physics_constraints and surface_areas is not None and surface_normals is not None:
            physics_loss = self._compute_physics_consistency(
                pred_surface, pred_coefficients, surface_areas, surface_normals
            )
            loss_components['physics'] = physics_loss.item()
            total_loss += self.physics_weight * physics_loss
        
        # 4. Model regularization
        if model is not None and hasattr(model, 'get_regularization_loss'):
            reg_loss = model.get_regularization_loss()
            if reg_loss > 0:
                loss_components['regularization'] = reg_loss.item()
                total_loss += 0.01 * reg_loss
        
        # 5. Track improvement over baseline
        if coarse_surface is not None:
            baseline_error = torch.mean((coarse_surface - target_surface) ** 2)
            prediction_error = torch.mean((pred_surface - target_surface) ** 2)
            improvement = (baseline_error - prediction_error) / (baseline_error + 1e-8)
            loss_components['improvement'] = improvement.item()
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components
    
    def _compute_surface_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute surface field prediction loss."""
        
        # Split pressure and wall shear stress
        pred_pressure = pred[..., 0:1]
        pred_shear = pred[..., 1:4]
        
        target_pressure = target[..., 0:1]
        target_shear = target[..., 1:4]
        
        # Pressure loss (MSE)
        pressure_loss = self.mse_loss(pred_pressure, target_pressure)
        
        # Wall shear stress loss (MSE per component)
        shear_loss = self.mse_loss(pred_shear, target_shear)
        
        # Distribution matching loss
        pred_mean = pred.mean(dim=1)
        target_mean = target.mean(dim=1)
        mean_loss = self.mse_loss(pred_mean, target_mean)
        
        pred_std = pred.std(dim=1)
        target_std = target.std(dim=1)
        std_loss = self.mse_loss(pred_std, target_std)
        
        # Combine losses
        surface_loss = 0.3 * pressure_loss + 0.3 * shear_loss + 0.2 * mean_loss + 0.2 * std_loss
        
        return surface_loss
    
    def _compute_coefficient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute coefficient prediction loss."""
        
        # Use L1 loss for coefficients (more robust to outliers)
        cd_loss = self.l1_loss(pred[:, 0], target[:, 0])
        cl_loss = self.l1_loss(pred[:, 1], target[:, 1])
        
        # Relative error penalty for large errors
        cd_rel_error = torch.abs(pred[:, 0] - target[:, 0]) / (torch.abs(target[:, 0]) + 1e-4)
        cl_rel_error = torch.abs(pred[:, 1] - target[:, 1]) / (torch.abs(target[:, 1]) + 1e-4)
        
        rel_penalty = torch.mean(cd_rel_error + cl_rel_error)
        
        # Combine losses
        coeff_loss = 0.4 * cd_loss + 0.4 * cl_loss + 0.2 * rel_penalty
        
        return coeff_loss
    
    def _compute_physics_consistency(
        self,
        surface_fields: torch.Tensor,
        predicted_coeffs: Optional[torch.Tensor],
        areas: torch.Tensor,
        normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics consistency between surface fields and coefficients.
        """
        
        if predicted_coeffs is None:
            return torch.tensor(0.0, device=surface_fields.device)
        
        # Ensure proper shapes
        if areas.dim() == 2:
            areas = areas.unsqueeze(-1)
        
        # Extract components
        pressure = surface_fields[..., 0:1]
        wall_shear_x = surface_fields[..., 1:2]
        wall_shear_z = surface_fields[..., 3:4]
        
        normal_x = normals[..., 0:1]
        normal_z = normals[..., 2:3]
        
        # Integrate forces from surface fields
        integrated_cd = torch.sum(
            pressure * normal_x * areas - wall_shear_x * areas,
            dim=1
        ).squeeze()
        
        integrated_cl = torch.sum(
            pressure * normal_z * areas - wall_shear_z * areas,
            dim=1
        ).squeeze()
        
        # Normalize integrated values (approximate scaling)
        integrated_cd = integrated_cd / 1000.0  # Approximate normalization
        integrated_cl = integrated_cl / 1000.0
        
        # Compare with predicted coefficients
        cd_consistency = torch.abs(integrated_cd - predicted_coeffs[:, 0])
        cl_consistency = torch.abs(integrated_cl - predicted_coeffs[:, 1])
        
        physics_loss = torch.mean(cd_consistency + cl_consistency)
        
        return physics_loss
    
    def update_weights(self, epoch: int):
        """
        Dynamically adjust loss weights during training.
        
        Args:
            epoch: Current training epoch
        """
        if epoch < 10:
            # Focus on surface fields initially
            self.surface_weight = 0.8
            self.coefficient_weight = 0.1
            self.physics_weight = 0.1
        elif epoch < 50:
            # Gradually increase coefficient importance
            self.surface_weight = 0.6
            self.coefficient_weight = 0.3
            self.physics_weight = 0.1
        else:
            # Final balanced weights
            self.surface_weight = 0.5
            self.coefficient_weight = 0.35
            self.physics_weight = 0.15
