# physics_loss.py - New loss function that prevents wrong physics

import torch
import torch.nn as nn

class PhysicsAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_weight = 0.4
        self.distribution_weight = 0.3
        self.physics_weight = 0.2
        self.regularization_weight = 0.1
        
    def forward(self, pred, target, coarse_input, areas, normals, model):
        batch_size = pred.shape[0]
        
        # 1. Standard MSE loss
        mse_loss = torch.mean((pred - target) ** 2)
        
        # 2. Distribution matching loss (prevents mean shift)
        pred_mean = pred.mean(dim=1)
        target_mean = target.mean(dim=1)
        mean_loss = torch.mean((pred_mean - target_mean) ** 2)
        
        pred_std = pred.std(dim=1)
        target_std = target.std(dim=1)
        std_loss = torch.mean((pred_std - target_std) ** 2)
        
        distribution_loss = mean_loss + 0.5 * std_loss
        
        # 3. Physics-based loss (force coefficients)
        drag_pred = self._compute_drag(pred, areas, normals)
        drag_target = self._compute_drag(target, areas, normals)
        physics_loss = (drag_pred - drag_target) ** 2
        
        # 4. Regularization on weights and correction magnitude
        reg_loss = 0.0
        if hasattr(model, 'coarse_to_fine_model'):
            c2f = model.coarse_to_fine_model
            
            # Penalize if residual weight goes below 0.7
            if c2f.residual_weight < 0.7:
                reg_loss += (0.7 - c2f.residual_weight) ** 2
            
            # Penalize if correction weight goes above 0.3
            if c2f.correction_weight > 0.3:
                reg_loss += (c2f.correction_weight - 0.3) ** 2
            
            # Penalize large corrections
            correction = pred - coarse_input
            correction_magnitude = torch.mean(correction ** 2)
            if correction_magnitude > 0.1:  # Corrections shouldn't be huge
                reg_loss += correction_magnitude
        
        # Combine all losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.distribution_weight * distribution_loss +
            self.physics_weight * physics_loss +
            self.regularization_weight * reg_loss
        )
        
        # Return loss components for monitoring
        return total_loss, {
            'mse': mse_loss.item(),
            'distribution': distribution_loss.item(),
            'physics': physics_loss.item(),
            'regularization': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'mean_error': mean_loss.item(),
            'std_error': std_loss.item()
        }
    
    def _compute_drag(self, fields, areas, normals):
        pressure = fields[:, :, 0:1]
        wall_shear_x = fields[:, :, 1:2]
        
        pressure_force = torch.sum(pressure * normals[:, :, 0:1] * areas, dim=1)
        shear_force = torch.sum(wall_shear_x * areas, dim=1)
        
        return torch.mean(pressure_force - shear_force)
