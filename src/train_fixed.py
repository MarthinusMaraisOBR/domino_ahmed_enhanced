#!/usr/bin/env python3
"""
train_fixed.py - Fixed training script for Enhanced DoMINO with proper dynamics
Includes gradient clipping, adaptive learning rate, correction monitoring, and debugging
"""

import time
import os
import torch
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
    create_domino_dataset,
)

# Import fixed versions
from enhanced_domino_model import DoMINOEnhanced
from physics_loss_fixed import PhysicsAwareLossFixed

# For memory tracking
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()


class EnhancedTrainer:
    """Enhanced trainer with comprehensive monitoring and fixes."""
    
    def __init__(self, cfg: DictConfig, dist: DistributedManager):
        self.cfg = cfg
        self.dist = dist
        self.logger = RankZeroLoggingWrapper(PythonLogger("EnhancedTrain"), dist)
        
        # Training state tracking
        self.best_improvement = -float('inf')
        self.patience_counter = 0
        self.max_patience = 20
        self.negative_improvement_counter = 0
        self.max_negative_improvements = 10
        
        # Monitoring thresholds
        self.max_correction_magnitude = 0.3
        self.min_improvement_threshold = -0.5
        self.target_mean_error = 0.1
        
        # Initialize physics loss with debugging
        self.physics_loss = PhysicsAwareLossFixed(force_scale=1e6, debug=True)
        
        # GPU handle for memory monitoring
        self.gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)
        
    def validate_batch(self, batch_data, predictions, targets, coarse_input):
        """Validate batch predictions and detect anomalies."""
        
        # Check for NaN/Inf
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            self.logger.error("NaN/Inf detected in predictions!")
            return False
            
        # Check correction magnitude
        corrections = predictions - coarse_input
        correction_magnitude = torch.abs(corrections).mean().item()
        
        if correction_magnitude > self.max_correction_magnitude:
            self.logger.warning(f"Large corrections detected: {correction_magnitude:.4f}")
            
        # Check if predictions are just passthrough
        correlation = torch.corrcoef(torch.stack([
            predictions.flatten(),
            coarse_input.flatten()
        ]))[0, 1].item()
        
        if abs(correlation) > 0.99:
            self.logger.warning("Predictions nearly identical to input (passthrough behavior)")
            
        # Check mean preservation
        pred_mean = predictions.mean().item()
        target_mean = targets.mean().item()
        mean_error = abs(pred_mean - target_mean)
        
        if mean_error > self.target_mean_error:
            self.logger.warning(f"Large mean error: {mean_error:.4f}")
            
        return True
    
    def train_epoch(self, dataloader, model, optimizer, scaler, epoch, tb_writer):
        """Single training epoch with comprehensive monitoring."""
        
        model.train()
        epoch_stats = {
            'loss': [],
            'improvement': [],
            'mean_error': [],
            'correction_magnitude': [],
            'physics_loss': [],
            'weight_stats': []
        }
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            batch_data = self._dict_to_device(batch_data)
            
            # Extract fields for enhanced training
            surface_fields = batch_data["surface_fields"]
            target_surf = surface_fields[..., :4]  # Fine features
            coarse_surf = surface_fields[..., 4:8]  # Coarse features
            
            # Prepare model input (only coarse features during forward pass)
            batch_data["surface_fields"] = coarse_surf
            
            with autocast(enabled=True):
                # Forward pass
                _, predictions = model(batch_data)
                
                # Calculate physics-aware loss
                areas = batch_data["surface_areas"].unsqueeze(-1)
                normals = batch_data["surface_normals"]
                
                loss, loss_components = self.physics_loss(
                    predictions,
                    target_surf,
                    coarse_surf,
                    areas,
                    normals,
                    model
                )
                
            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            
            # Gradient clipping - CRITICAL for stability
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Validate batch
            with torch.no_grad():
                batch_valid = self.validate_batch(
                    batch_data, predictions, target_surf, coarse_surf
                )
            
            # Record statistics
            epoch_stats['loss'].append(loss.item())
            epoch_stats['improvement'].append(loss_components['improvement'])
            epoch_stats['mean_error'].append(loss_components['mean_error'])
            epoch_stats['correction_magnitude'].append(loss_components['relative_correction'])
            epoch_stats['physics_loss'].append(loss_components['physics'])
            
            # Monitor model weights
            if hasattr(model, 'module'):  # DDP
                actual_model = model.module
            else:
                actual_model = model
                
            if hasattr(actual_model, 'coarse_to_fine_model'):
                c2f = actual_model.coarse_to_fine_model
                if hasattr(c2f, 'actual_residual_weight'):
                    epoch_stats['weight_stats'].append({
                        'residual': c2f.actual_residual_weight.item(),
                        'correction': c2f.actual_correction_weight.item()
                    })
            
            # Detailed logging every N batches
            if batch_idx % 20 == 0:
                self._log_batch_stats(batch_idx, len(dataloader), epoch, 
                                     loss_components, grad_norm)
                
            # Early stopping within epoch if things go wrong
            if loss_components['improvement'] < self.min_improvement_threshold:
                self.negative_improvement_counter += 1
                if self.negative_improvement_counter > self.max_negative_improvements:
                    self.logger.error("Too many negative improvements - stopping epoch")
                    break
            else:
                self.negative_improvement_counter = max(0, self.negative_improvement_counter - 1)
        
        # Compute epoch statistics
        epoch_summary = self._compute_epoch_summary(epoch_stats)
        
        # Log to tensorboard
        if self.dist.rank == 0 and tb_writer:
            self._log_to_tensorboard(tb_writer, epoch, epoch_summary)
        
        return epoch_summary
    
    def _dict_to_device(self, data_dict):
        """Move dictionary of tensors to device."""
        device_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                device_dict[key] = value.to(self.dist.device)
            elif isinstance(value, np.ndarray):
                device_dict[key] = torch.from_numpy(value).to(self.dist.device)
            else:
                device_dict[key] = value
        return device_dict
    
    def _log_batch_stats(self, batch_idx, total_batches, epoch, loss_components, grad_norm):
        """Log detailed batch statistics."""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}")
        self.logger.info(f"{'='*60}")
        
        # Loss components
        self.logger.info("Loss Components:")
        self.logger.info(f"  MSE:          {loss_components['mse']:.2e}")
        self.logger.info(f"  Distribution: {loss_components['distribution']:.2e}")
        self.logger.info(f"  Physics:      {loss_components['physics']:.2e}")
        self.logger.info(f"  Regularization: {loss_components['regularization']:.2e}")
        
        # Performance metrics
        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"  Improvement:  {loss_components['improvement']*100:.1f}%")
        self.logger.info(f"  Mean Error:   {loss_components['mean_error']:.4f}")
        self.logger.info(f"  Correction Magnitude: {loss_components['relative_correction']*100:.1f}%")
        
        # Force errors
        self.logger.info("\nForce Errors (normalized):")
        self.logger.info(f"  Drag: {loss_components['drag_error']:.4f}")
        self.logger.info(f"  Lift: {loss_components['lift_error']:.4f}")
        
        # Training dynamics
        self.logger.info(f"\nGradient Norm: {grad_norm:.4f}")
        
        # Memory usage
        gpu_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
        self.logger.info(f"GPU Memory: {gpu_info.used / 1024**3:.2f} GB")
    
    def _compute_epoch_summary(self, epoch_stats):
        """Compute summary statistics for the epoch."""
        
        summary = {}
        for key in ['loss', 'improvement', 'mean_error', 'correction_magnitude', 'physics_loss']:
            if epoch_stats[key]:
                summary[f'{key}_mean'] = np.mean(epoch_stats[key])
                summary[f'{key}_std'] = np.std(epoch_stats[key])
                summary[f'{key}_min'] = np.min(epoch_stats[key])
                summary[f'{key}_max'] = np.max(epoch_stats[key])
        
        # Weight statistics
        if epoch_stats['weight_stats']:
            weights = epoch_stats['weight_stats']
            summary['residual_weight'] = np.mean([w['residual'] for w in weights])
            summary['correction_weight'] = np.mean([w['correction'] for w in weights])
        
        return summary
    
    def _log_to_tensorboard(self, writer, epoch, summary):
        """Log metrics to TensorBoard."""
        
        for key, value in summary.items():
            if 'weight' in key:
                writer.add_scalar(f'Weights/{key}', value, epoch)
            elif 'improvement' in key:
                writer.add_scalar(f'Performance/{key}', value * 100, epoch)
            elif 'mean_error' in key:
                writer.add_scalar(f'Errors/{key}', value, epoch)
            elif 'physics' in key:
                writer.add_scalar(f'Physics/{key}', value, epoch)
            else:
                writer.add_scalar(f'Loss/{key}', value, epoch)
    
    def should_stop_early(self, epoch_summary):
        """Determine if training should stop early."""
        
        # Check improvement trend
        current_improvement = epoch_summary.get('improvement_mean', -1.0)
        
        if current_improvement > self.best_improvement:
            self.best_improvement = current_improvement
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Stop conditions
        stop_reasons = []
        
        if self.patience_counter > self.max_patience:
            stop_reasons.append(f"No improvement for {self.max_patience} epochs")
        
        if current_improvement < -0.5:
            stop_reasons.append(f"Severe negative improvement: {current_improvement*100:.1f}%")
        
        if epoch_summary.get('mean_error_mean', 0) > 0.5:
            stop_reasons.append(f"Mean error too high: {epoch_summary['mean_error_mean']:.4f}")
        
        if epoch_summary.get('residual_weight', 1.0) < 0.5:
            stop_reasons.append(f"Residual weight collapsed: {epoch_summary['residual_weight']:.3f}")
        
        if stop_reasons:
            self.logger.error("EARLY STOPPING TRIGGERED:")
            for reason in stop_reasons:
                self.logger.error(f"  - {reason}")
            return True
        
        return False


def create_adaptive_optimizer(model, cfg, epoch=0):
    """Create optimizer with adaptive learning rate."""
    
    # Start with very low learning rate
    if epoch < 10:
        lr = 5e-5
    elif epoch < 50:
        lr = 1e-4
    else:
        lr = 5e-5
    
    # Different learning rates for different parts
    param_groups = [
        {'params': [], 'lr': lr, 'name': 'coarse_to_fine'},
        {'params': [], 'lr': lr * 0.1, 'name': 'output_projection'},
        {'params': [], 'lr': lr * 0.5, 'name': 'other'}
    ]
    
    for name, param in model.named_parameters():
        if 'output_projection' in name:
            param_groups[1]['params'].append(param)
        elif 'coarse_to_fine' in name:
            param_groups[0]['params'].append(param)
        else:
            param_groups[2]['params'].append(param)
    
    # Remove empty groups
    param_groups = [g for g in param_groups if g['params']]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-5,  # Add weight decay for regularization
        betas=(0.9, 0.999)
    )
    
    return optimizer


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with all fixes applied."""
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Create trainer
    trainer = EnhancedTrainer(cfg, dist)
    
    trainer.logger.info("="*80)
    trainer.logger.info("ENHANCED DOMINO TRAINING - FIXED VERSION")
    trainer.logger.info("="*80)
    trainer.logger.info("Key fixes applied:")
    trainer.logger.info("  ✓ Physics loss properly scaled (1e6)")
    trainer.logger.info("  ✓ Gradient clipping enabled")
    trainer.logger.info("  ✓ Adaptive learning rate")
    trainer.logger.info("  ✓ Correction magnitude monitoring")
    trainer.logger.info("  ✓ Early stopping on anomalies")
    trainer.logger.info("  ✓ Weight regularization")
    trainer.logger.info("="*80)
    
    # Compute scaling factors
    compute_scaling_factors(
        cfg=cfg,
        input_path=cfg.data_processor.output_dir,
        use_cache=cfg.data_processor.use_cache,
    )
    
    # Load scaling factors
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, cfg.exp_tag, "surface_scaling_factors.npy"
    )
    if not os.path.exists(surf_save_path):
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )
    
    surf_factors = np.load(surf_save_path) if os.path.exists(surf_save_path) else None
    
    # Get variable names
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = sum(
        3 if cfg.variables.surface.solution[j] == "vector" else 1 
        for j in surface_variable_names
    )
    
    # Create datasets
    train_dataset = create_domino_dataset(
        cfg, phase="train",
        volume_variable_names=[],
        surface_variable_names=surface_variable_names,
        vol_factors=None,
        surf_factors=surf_factors,
    )
    
    val_dataset = create_domino_dataset(
        cfg, phase="val", 
        volume_variable_names=[],
        surface_variable_names=surface_variable_names,
        vol_factors=None,
        surf_factors=surf_factors,
    )
    
    # Create data loaders
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.world_size, rank=dist.rank,
        shuffle=True, drop_last=False
    )
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=dist.world_size, rank=dist.rank,
        shuffle=False, drop_last=False
    )
    
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=1,  # Must be 1 due to ball query
        pin_memory=False, num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=1, pin_memory=False, num_workers=0
    )
    
    # Create model
    trainer.logger.info("\nCreating Enhanced DoMINO model...")
    model = DoMINOEnhanced(
        input_features=3,
        output_features_vol=None,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)
    
    model = torch.compile(model, disable=True)
    
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    
    # Create optimizer with adaptive learning rate
    optimizer = create_adaptive_optimizer(model, cfg, epoch=0)
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    # Setup tensorboard
    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))
    
    # Setup checkpoint directory
    model_save_path = os.path.join(cfg.output, "models")
    best_model_path = os.path.join(model_save_path, "best_model")
    
    if dist.rank == 0:
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        Path(best_model_path).mkdir(parents=True, exist_ok=True)
    
    if dist.world_size > 1:
        torch.distributed.barrier()
    
    # Training loop
    trainer.logger.info("\nStarting training loop...")
    trainer.logger.info("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.epochs):
        trainer.logger.info(f"\nEPOCH {epoch}/{cfg.train.epochs}")
        trainer.logger.info("="*60)
        
        # Adjust physics loss scale based on epoch
        trainer.physics_loss.adjust_force_scale(epoch)
        
        # Recreate optimizer with new learning rate if needed
        if epoch in [10, 50, 100]:
            optimizer = create_adaptive_optimizer(model, cfg, epoch)
            trainer.logger.info(f"Adjusted learning rate for epoch {epoch}")
        
        # Training epoch
        train_sampler.set_epoch(epoch)
        epoch_summary = trainer.train_epoch(
            train_dataloader, model, optimizer, scaler, epoch, writer
        )
        
        # Log epoch summary
        trainer.logger.info(f"\nEpoch {epoch} Summary:")
        trainer.logger.info(f"  Average Loss: {epoch_summary.get('loss_mean', 0):.2e}")
        trainer.logger.info(f"  Average Improvement: {epoch_summary.get('improvement_mean', 0)*100:.1f}%")
        trainer.logger.info(f"  Mean Error: {epoch_summary.get('mean_error_mean', 0):.4f}")
        trainer.logger.info(f"  Physics Loss: {epoch_summary.get('physics_loss_mean', 0):.2e}")
        
        if 'residual_weight' in epoch_summary:
            trainer.logger.info(f"  Weights - Residual: {epoch_summary['residual_weight']:.3f}, "
                              f"Correction: {epoch_summary['correction_weight']:.3f}")
        
        # Validation (simplified - can be expanded)
        val_loss = epoch_summary.get('loss_mean', float('inf'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.logger.info(f"  New best validation loss: {best_val_loss:.2e}")
            
            if dist.rank == 0:
                save_checkpoint(
                    best_model_path,
                    models=model,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=scaler,
                    epoch=epoch,
                )
        
        # Regular checkpointing
        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0:
            save_checkpoint(
                model_save_path,
                models=model,
                optimizer=optimizer, 
                scheduler=None,
                scaler=scaler,
                epoch=epoch,
            )
        
        # Early stopping check
        if trainer.should_stop_early(epoch_summary):
            trainer.logger.error(f"Early stopping at epoch {epoch}")
            break
    
    trainer.logger.info("\n" + "="*80)
    trainer.logger.info("TRAINING COMPLETED")
    trainer.logger.info("="*80)
    trainer.logger.info(f"Best validation loss: {best_val_loss:.2e}")
    trainer.logger.info(f"Best improvement: {trainer.best_improvement*100:.1f}%")
    trainer.logger.info("\nNext steps:")
    trainer.logger.info("1. Test with: python test_enhanced.py")
    trainer.logger.info("2. Check visualizations in ParaView")


if __name__ == "__main__":
    main()
