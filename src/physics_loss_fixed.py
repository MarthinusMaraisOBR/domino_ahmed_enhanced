# physics_loss_fixed.py - Physics-aware loss with proper scaling and force calculations

import torch
import torch.nn as nn
import numpy as np

class PhysicsAwareLossFixed(nn.Module):
    def __init__(self, force_scale=1e6, debug=False):
        super().__init__()
        # Adjusted weights for better balance
        self.mse_weight = 0.3  # Reduced from 0.4
        self.distribution_weight = 0.2  # Reduced from 0.3
        self.physics_weight = 0.4  # Increased from 0.2
        self.regularization_weight = 0.1
        
        # Force scaling factor to make physics loss comparable to MSE
        self.force_scale = force_scale
        self.debug = debug
        
        # Track statistics for monitoring
        self.running_stats = {
            'mse_scale': [],
            'physics_scale': [],
            'force_errors': []
        }
        
    def forward(self, pred, target, coarse_input, areas, normals, model):
        batch_size = pred.shape[0]
        device = pred.device
        
        # Ensure all inputs are on same device and dtype
        pred = pred.to(device)
        target = target.to(device)
        coarse_input = coarse_input.to(device)
        areas = areas.to(device)
        normals = normals.to(device)
        
        # 1. Standard MSE loss (normalized)
        mse_loss = torch.mean((pred - target) ** 2)
        
        # 2. Distribution matching loss
        pred_mean = pred.mean(dim=1)
        target_mean = target.mean(dim=1)
        mean_loss = torch.mean((pred_mean - target_mean) ** 2)
        
        pred_std = pred.std(dim=1) + 1e-8  # Avoid division by zero
        target_std = target.std(dim=1) + 1e-8
        std_loss = torch.mean(((pred_std - target_std) / target_std) ** 2)
        
        distribution_loss = mean_loss + 0.5 * std_loss
        
        # 3. Physics-based loss (properly scaled force coefficients)
        # Calculate forces for each field
        drag_pred, lift_pred = self._compute_forces(pred, areas, normals)
        drag_target, lift_target = self._compute_forces(target, areas, normals)
        drag_coarse, lift_coarse = self._compute_forces(coarse_input, areas, normals)
        
        # Calculate force errors (normalized by baseline)
        baseline_drag_error = torch.abs(drag_coarse - drag_target) + 1e-8
        baseline_lift_error = torch.abs(lift_coarse - lift_target) + 1e-8
        
        drag_error = torch.abs(drag_pred - drag_target) / baseline_drag_error
        lift_error = torch.abs(lift_pred - lift_target) / baseline_lift_error
        
        # Scale up physics loss to make it comparable to MSE
        physics_loss = (drag_error + lift_error) * self.force_scale
        
        # 4. Regularization on correction magnitude
        reg_loss = 0.0
        if hasattr(model, 'coarse_to_fine_model'):
            c2f = model.coarse_to_fine_model
            
            # Penalize if residual weight goes below threshold
            if hasattr(c2f, 'actual_residual_weight'):
                residual_w = c2f.actual_residual_weight
                if residual_w < 0.8:
                    reg_loss += ((0.8 - residual_w) ** 2) * 10.0
            
            # Penalize if correction weight goes above threshold
            if hasattr(c2f, 'actual_correction_weight'):
                correction_w = c2f.actual_correction_weight
                if correction_w > 0.25:
                    reg_loss += ((correction_w - 0.25) ** 2) * 10.0
            
            # Penalize large corrections relative to input
            correction = pred - coarse_input
            correction_magnitude = torch.mean(torch.abs(correction))
            input_magnitude = torch.mean(torch.abs(coarse_input)) + 1e-8
            
            # Corrections should be small relative to input (typically < 20%)
            relative_correction = correction_magnitude / input_magnitude
            if relative_correction > 0.2:
                reg_loss += ((relative_correction - 0.2) ** 2) * 5.0
                
        # Add regularization as tensor if it's scalar
        if not isinstance(reg_loss, torch.Tensor):
            reg_loss = torch.tensor(reg_loss, device=device)
        
        # Combine all losses with adaptive weighting
        total_loss = (
            self.mse_weight * mse_loss +
            self.distribution_weight * distribution_loss +
            self.physics_weight * physics_loss +
            self.regularization_weight * reg_loss
        )
        
        # Debug output
        if self.debug:
            self.running_stats['mse_scale'].append(mse_loss.item())
            self.running_stats['physics_scale'].append(physics_loss.item())
            self.running_stats['force_errors'].append({
                'drag': drag_error.item(),
                'lift': lift_error.item()
            })
            
            if len(self.running_stats['mse_scale']) % 10 == 0:
                avg_mse = np.mean(self.running_stats['mse_scale'][-10:])
                avg_physics = np.mean(self.running_stats['physics_scale'][-10:])
                print(f"\n[Physics Loss Debug]")
                print(f"  Average MSE scale: {avg_mse:.2e}")
                print(f"  Average Physics scale: {avg_physics:.2e}")
                print(f"  Scale ratio: {avg_physics/avg_mse:.2f}")
                
                if avg_physics < avg_mse * 0.1:
                    print(f"  ⚠️ Physics loss too small - increase force_scale")
                elif avg_physics > avg_mse * 10:
                    print(f"  ⚠️ Physics loss too large - decrease force_scale")
        
        # Calculate improvement metric
        baseline_error = torch.mean((coarse_input - target) ** 2)
        prediction_error = torch.mean((pred - target) ** 2)
        improvement = (baseline_error - prediction_error) / (baseline_error + 1e-8)
        
        # Return loss components for monitoring
        return total_loss, {
            'mse': mse_loss.item(),
            'distribution': distribution_loss.item(),
            'mean_error': mean_loss.item(),
            'std_error': std_loss.item(),
            'physics': physics_loss.item(),
            'drag_error': drag_error.item(),
            'lift_error': lift_error.item(),
            'regularization': reg_loss.item(),
            'improvement': improvement.item(),
            'relative_correction': (correction_magnitude / input_magnitude).item() if 'correction' in locals() else 0.0
        }
    
    def _compute_forces(self, fields, areas, normals):
        """
        Compute drag and lift forces from surface fields.
        
        Args:
            fields: [batch, points, 4] - pressure and wall shear stress
            areas: [batch, points, 1] - cell areas
            normals: [batch, points, 3] - surface normals
        
        Returns:
            drag: [batch] - integrated drag coefficient
            lift: [batch] - integrated lift coefficient
        """
        # Ensure proper shapes
        if areas.dim() == 2:
            areas = areas.unsqueeze(-1)
        
        # Extract pressure and wall shear components
        pressure = fields[:, :, 0:1]  # [batch, points, 1]
        wall_shear_x = fields[:, :, 1:2]  # [batch, points, 1]
        wall_shear_z = fields[:, :, 3:4]  # [batch, points, 1]
        
        # Extract normal components
        normal_x = normals[:, :, 0:1]  # [batch, points, 1]
        normal_z = normals[:, :, 2:3]  # [batch, points, 1]
        
        # Calculate pressure forces
        pressure_force_x = pressure * normal_x * areas  # [batch, points, 1]
        pressure_force_z = pressure * normal_z * areas  # [batch, points, 1]
        
        # Calculate shear forces
        shear_force_x = wall_shear_x * areas  # [batch, points, 1]
        shear_force_z = wall_shear_z * areas  # [batch, points, 1]
        
        # Integrate forces (sum over points, average over batch)
        drag = torch.mean(torch.sum(pressure_force_x - shear_force_x, dim=1))
        lift = torch.mean(torch.sum(pressure_force_z - shear_force_z, dim=1))
        
        return drag, lift
    
    def adjust_force_scale(self, epoch):
        """
        Dynamically adjust force scale based on training progress.
        
        Args:
            epoch: Current epoch number
        """
        if epoch < 10:
            # Start with smaller physics weight to stabilize training
            self.force_scale = 1e5
        elif epoch < 50:
            # Gradually increase physics importance
            self.force_scale = 1e6
        else:
            # Full physics enforcement
            self.force_scale = 5e6
            
    def get_loss_weights(self):
        """Return current loss weights for logging."""
        return {
            'mse': self.mse_weight,
            'distribution': self.distribution_weight,
            'physics': self.physics_weight,
            'regularization': self.regularization_weight,
            'force_scale': self.force_scale
        }
