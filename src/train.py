# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code defines a distributed pipeline for training the DoMINO model on
CFD datasets. It includes the computation of scaling factors, instantiating 
the DoMINO model and datapipe, automatically loading the most recent checkpoint, 
training the model in parallel using DistributedDataParallel across multiple 
GPUs, calculating the loss and updating model parameters using mixed precision. 
This is a common recipe that enables training of combined models for surface and 
volume as well either of them separately. Validation is also conducted every epoch, 
where predictions are compared against ground truth values. The code logs training
and validation metrics to TensorBoard. The train tab in config.yaml can be used to 
specify batch size, number of epochs and other training parameters.
"""

import time
import os
import re
import torch
import torchinfo

from typing import Literal

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time

# Import the enhanced model (only once!)
from enhanced_domino_model import DoMINOEnhanced

from physicsnemo.utils.profiling import profile, Profiler

# Initialize NVML
nvmlInit()


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
) -> torch.Tensor:
    """Calculate mean squared error or root mean squared error with masking for padded values.

    Args:
        output: Predicted values from the model
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor

    Returns:
        Calculated loss as a scalar tensor
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * target**2.0, dims)
    else:
        denom = torch.sum(mask)

    return torch.mean(num / denom)


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_scalar, output_vector = torch.split(output, [1, 3], dim=2)
    target_scalar, target_vector = torch.split(target, [1, 3], dim=2)

    numerator = torch.mean((output_scalar - target_scalar) ** 2.0)
    vector_diff_sq = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "mse":
        masked_loss_pres = numerator
        masked_loss_ws = torch.sum(vector_diff_sq)
    else:
        denom = torch.mean((target_scalar) ** 2.0)
        masked_loss_pres = numerator / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        masked_loss_ws_num = vector_diff_sq
        masked_loss_ws_denom = torch.mean((target_vector) ** 2.0, (0, 1))
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_ws

    return loss / 4.0


def loss_fn_area(
    output: torch.Tensor,
    target: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    area_scaling_factor: float,
    loss_type: Literal["mse", "rmse"],
) -> torch.Tensor:
    """Calculate area-weighted loss for surface data considering normal vectors.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        normals: Normal vectors for the surface
        area: Area values for surface elements
        area_scaling_factor: Scaling factor for area weighting
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Area-weighted loss as a scalar tensor
    """
    area = area * area_scaling_factor
    area_scale_factor = area

    # Separate the scalar and vector components.
    target_scalar, target_vector = torch.split(
        target * area_scale_factor, [1, 3], dim=2
    )
    output_scalar, output_vector = torch.split(
        output * area_scale_factor, [1, 3], dim=2
    )

    # Apply the normals to the scalar components (only [:,:,0]):
    normals, _ = torch.split(normals, [1, normals.shape[-1] - 1], dim=2)
    target_scalar = target_scalar * normals
    output_scalar = output_scalar * normals

    # Compute the mean diff**2 of the scalar component:
    masked_loss_pres = torch.mean(((output_scalar - target_scalar) ** 2.0), dim=(0, 1))
    if loss_type == "rmse":
        masked_loss_pres /= torch.mean(target_scalar**2.0, dim=(0, 1))

    # Compute the mean diff**2 of the vector component, leave the last dimension:
    masked_loss_ws = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))

    if loss_type == "rmse":
        masked_loss_ws /= torch.mean((target_vector) ** 2.0, (0, 1))

    # Combine the scalar and vector components:
    loss = 0.25 * (masked_loss_pres + torch.sum(masked_loss_ws))

    return loss


def integral_loss_fn(
    output, target, area, normals, stream_velocity=None, padded_value=-10
):
    drag_loss = drag_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    lift_loss = lift_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3

    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    normals = torch.select(normals, 2, 2)
    # output_true_0 = output_true[:, :, 0]
    output_true_0 = output_true.select(2, 0)
    output_pred_0 = output_pred.select(2, 0)

    pres_true = output_true_0 * normals
    pres_pred = output_pred_0 * normals

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.mean(pres_pred + wz_pred, (1))
    masked_truth = torch.mean(pres_true + wz_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.mean(pres_pred + wx_pred, (1))
    masked_truth = torch.mean(pres_true + wx_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def compute_loss_dict(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    integral_scaling_factor: float,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
) -> tuple[torch.Tensor, dict]:
    """Compute the loss terms for standard training."""
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    # Volume loss
    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]
        loss_vol = loss_fn(
            prediction_vol, target_vol, loss_fn_type.loss_type, padded_value=-10
        )
        if loss_fn_type.loss_type == "mse":
            loss_vol = loss_vol * vol_loss_scaling
        loss_dict["loss_vol"] = loss_vol
        total_loss_terms.append(loss_vol)

    # Surface loss
    if prediction_surf is not None:
        target_surf = batch_inputs["surface_fields"]
        surface_areas = batch_inputs["surface_areas"]
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_normals = batch_inputs["surface_normals"]
        stream_velocity = batch_inputs["stream_velocity"]

        loss_surf = loss_fn_surface(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        loss_surf_area = loss_fn_area(
            prediction_surf,
            target_surf,
            surface_normals,
            surface_areas,
            area_scaling_factor=loss_fn_type.area_weighing_factor,
            loss_type=loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling
            loss_surf_area = loss_surf_area * surf_loss_scaling

        total_loss_terms.append(0.5 * loss_surf)
        loss_dict["loss_surf"] = 0.5 * loss_surf
        total_loss_terms.append(0.5 * loss_surf_area)
        loss_dict["loss_surf_area"] = 0.5 * loss_surf_area

        loss_integral = (
            integral_loss_fn(
                prediction_surf,
                target_surf,
                surface_areas,
                surface_normals,
                stream_velocity,
                padded_value=-10,
            )
        ) * integral_scaling_factor
        loss_dict["loss_integral"] = loss_integral
        total_loss_terms.append(loss_integral)

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict


def compute_loss_dict_enhanced(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    integral_scaling_factor: float,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
    use_enhanced_features: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the loss terms for enhanced training.
    
    When use_enhanced_features=True:
    - prediction_surf contains predicted fine features (from coarse input)
    - batch_inputs["surface_fields"] contains 8 features [fine, coarse]
    - We compare predictions against the fine features only
    """
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    # Volume loss remains unchanged
    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]
        loss_vol = loss_fn(
            prediction_vol, target_vol, loss_fn_type.loss_type, padded_value=-10
        )
        if loss_fn_type.loss_type == "mse":
            loss_vol = loss_vol * vol_loss_scaling
        loss_dict["loss_vol"] = loss_vol
        total_loss_terms.append(loss_vol)

    # Surface loss with enhanced features
    if prediction_surf is not None:
        surface_areas = batch_inputs["surface_areas"]
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_normals = batch_inputs["surface_normals"]
        stream_velocity = batch_inputs["stream_velocity"]
        
        if use_enhanced_features:
            # Extract fine features as target (first 4 features)
            surface_fields_all = batch_inputs["surface_fields"]
            target_surf = surface_fields_all[..., :4]  # Fine features only
            
            # Log coarse-to-fine improvement
            coarse_features = surface_fields_all[..., 4:8]
            
            # Calculate baseline error (using coarse as prediction)
            baseline_error = torch.mean((coarse_features - target_surf) ** 2)
            prediction_error = torch.mean((prediction_surf - target_surf) ** 2)
            improvement = (baseline_error - prediction_error) / baseline_error
            loss_dict["improvement"] = improvement
        else:
            # Standard training
            target_surf = batch_inputs["surface_fields"]
        
        # Calculate losses
        loss_surf = loss_fn_surface(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        loss_surf_area = loss_fn_area(
            prediction_surf,
            target_surf,
            surface_normals,
            surface_areas,
            area_scaling_factor=loss_fn_type.area_weighing_factor,
            loss_type=loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling
            loss_surf_area = loss_surf_area * surf_loss_scaling

        total_loss_terms.append(0.5 * loss_surf)
        loss_dict["loss_surf"] = 0.5 * loss_surf
        total_loss_terms.append(0.5 * loss_surf_area)
        loss_dict["loss_surf_area"] = 0.5 * loss_surf_area
        
        # Integral loss
        loss_integral = (
            integral_loss_fn(
                prediction_surf,
                target_surf,
                surface_areas,
                surface_normals,
                stream_velocity,
                padded_value=-10,
            )
        ) * integral_scaling_factor
        loss_dict["loss_integral"] = loss_integral
        total_loss_terms.append(loss_integral)

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict


def validation_step(
    dataloader,
    model,
    device,
    logger,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
):
    running_vloss = 0.0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            with autocast(enabled=True):

                prediction_vol, prediction_surf = model(sampled_batched)
                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                )

            running_vloss += loss.item()

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss


def validation_step_enhanced(
    dataloader,
    model,
    device,
    logger,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    use_enhanced_features=False,
):
    """Enhanced validation handling coarse-to-fine prediction."""
    running_vloss = 0.0
    running_improvement = 0.0
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            with autocast(enabled=True):
                prediction_vol, prediction_surf = model(sampled_batched)
                
                if use_enhanced_features:
                    loss, loss_dict = compute_loss_dict_enhanced(
                        prediction_vol,
                        prediction_surf,
                        sampled_batched,
                        loss_fn_type,
                        integral_scaling_factor,
                        surf_loss_scaling,
                        vol_loss_scaling,
                        use_enhanced_features=True,
                    )
                    if "improvement" in loss_dict:
                        running_improvement += loss_dict["improvement"].item()
                else:
                    loss, loss_dict = compute_loss_dict(
                        prediction_vol,
                        prediction_surf,
                        sampled_batched,
                        loss_fn_type,
                        integral_scaling_factor,
                        surf_loss_scaling,
                        vol_loss_scaling,
                    )

            running_vloss += loss.item()

    avg_vloss = running_vloss / (i_batch + 1)
    
    if use_enhanced_features:
        avg_improvement = running_improvement / (i_batch + 1)
        logger.info(f"Validation - Average improvement over coarse: {avg_improvement:.1%}")

    return avg_vloss


@profile
def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
):

    dist = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    for i_batch, sample_batched in enumerate(dataloader):

        sampled_batched = dict_to_device(sample_batched, device)

        with autocast(enabled=True):
            with nvtx.range("Model Forward Pass"):
                prediction_vol, prediction_surf = model(sampled_batched)

            loss, loss_dict = compute_loss_dict(
                prediction_vol,
                prediction_surf,
                sampled_batched,
                loss_fn_type,
                integral_scaling_factor,
                surf_loss_scaling,
                vol_loss_scaling,
            )

        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Gather data and report
        running_loss += loss.item()
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        # Format the loss dict into a string:
        loss_string = (
            "  "
            + "\t".join([f"{key.replace('loss_',''):<10}" for key in loss_dict.keys()])
            + "\n"
        )
        loss_string += (
            "  " + f"\t".join([f"{l.item():<10.3e}" for l in loss_dict.values()]) + "\n"
        )

        logging_string += loss_string
        logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta:.3f} Gb\n"
        logging_string += f"  Time taken: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    if dist.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)

    return last_loss


@profile
def train_epoch_enhanced(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    use_enhanced_features=False,
):
    """Enhanced training epoch handling coarse-to-fine prediction."""
    
    dist = DistributedManager()
    running_loss = 0.0
    running_improvement = 0.0  # Track improvement over baseline
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    
    for i_batch, sample_batched in enumerate(dataloader):
        sampled_batched = dict_to_device(sample_batched, device)

        with autocast(enabled=True):
            with nvtx.range("Model Forward Pass"):
                prediction_vol, prediction_surf = model(sampled_batched)

            # Use enhanced loss calculation if needed
            if use_enhanced_features:
                loss, loss_dict = compute_loss_dict_enhanced(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    use_enhanced_features=True,
                )
                
                if "improvement" in loss_dict:
                    running_improvement += loss_dict["improvement"].item()
            else:
                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                )

        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Gather data and report
        running_loss += loss.item()
        
        # Enhanced logging
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        
        # Format the loss dict into a string
        loss_string = (
            "  "
            + "\t".join([f"{key.replace('loss_',''):<10}" for key in loss_dict.keys() if key != "improvement"])
            + "\n"
        )
        loss_string += (
            "  " + f"\t".join([f"{l.item():<10.3e}" for k, l in loss_dict.items() if k != "improvement"]) + "\n"
        )

        if use_enhanced_features and "improvement" in loss_dict:
            loss_string += f"  Improvement over coarse baseline: {loss_dict['improvement'].item():.1%}\n"

        logging_string += loss_string
        logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta:.3f} Gb\n"
        logging_string += f"  Time taken: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    last_loss = running_loss / (i_batch + 1)
    
    if use_enhanced_features:
        avg_improvement = running_improvement / (i_batch + 1)
        logger.info(f"Average improvement over coarse: {avg_improvement:.1%}")
    
    if dist.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)
        
        if use_enhanced_features and running_improvement > 0:
            avg_improvement = running_improvement / (i_batch + 1)
            tb_writer.add_scalar("Improvement/train", avg_improvement, tb_x)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize NVML
    nvmlInit()

    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    compute_scaling_factors(
        cfg=cfg,
        input_path=cfg.data_processor.output_dir,
        use_cache=cfg.data_processor.use_cache,
    )
    model_type = cfg.model.model_type

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    # Check if enhanced features are enabled
    use_enhanced_features = cfg.data_processor.get('use_enhanced_features', False)
    
    if use_enhanced_features:
        logger.info("="*60)
        logger.info("🚀 ENHANCED TRAINING MODE - Coarse-to-Fine Prediction")
        logger.info("="*60)
        logger.info("  Training to predict fine resolution from coarse input")
        logger.info(f"  Fine data path: {cfg.data_processor.input_dir}")
        logger.info(f"  Coarse data path: {cfg.data_processor.coarse_input_dir}")

    # Handle volume variables (check if they exist for surface-only datasets)
    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        if hasattr(cfg.variables, 'volume') and cfg.variables.volume is not None:
            volume_variable_names = list(cfg.variables.volume.solution.keys())
            for j in volume_variable_names:
                if cfg.variables.volume.solution[j] == "vector":
                    num_vol_vars += 3
                else:
                    num_vol_vars += 1
            logger.info(f"Volume variables: {volume_variable_names} (total: {num_vol_vars})")
        else:
            logger.info("No volume variables found - surface-only dataset")
    else:
        num_vol_vars = None

    # Surface variables
    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
        logger.info(f"Surface variables: {surface_variable_names} (total: {num_surf_vars})")
    else:
        num_surf_vars = None

    # Load scaling factors
    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    # Load surface scaling factors - use enhanced version if in enhanced mode
    if use_enhanced_features:
        # Look for enhanced scaling factors in the experiment directory
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, cfg.exp_tag, "surface_scaling_factors.npy"
        )
        if not os.path.exists(surf_save_path):
            # Fallback to main directory enhanced version
            surf_save_path = os.path.join(
                "outputs", cfg.project.name, "surface_scaling_factors_enhanced.npy"
            )
        logger.info(f"Using enhanced surface scaling factors from {surf_save_path}")
    else:
        surf_save_path = os.path.join(
            "outputs", cfg.project.name, "surface_scaling_factors.npy"
        )
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
        logger.info(f"Loaded volume scaling factors from {vol_save_path}")
    else:
        vol_factors = None

    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
        logger.info(f"Loaded surface scaling factors from {surf_save_path}")
    else:
        surf_factors = None

    # Create datasets
    train_dataset = create_domino_dataset(
        cfg,
        phase="train",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )
    val_dataset = create_domino_dataset(
        cfg,
        phase="val",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )

    # Create data loaders
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.train.sampler,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.val.sampler,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **cfg.train.dataloader,
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **cfg.val.dataloader,
    )

    # Model creation - use enhanced model if configured
    if use_enhanced_features:
        logger.info("\nCreating DoMINOEnhanced model for coarse-to-fine prediction...")
        model = DoMINOEnhanced(
            input_features=3,
            output_features_vol=num_vol_vars,
            output_features_surf=num_surf_vars,
            model_parameters=cfg.model,
        ).to(dist.device)
        logger.info("  ✅ Enhanced model created successfully")
        
        # Log enhanced configuration
        enhanced_config = cfg.model.get('enhanced_model', {})
        logger.info(f"  Surface input features: {enhanced_config.get('surface_input_features', 4)}")
        logger.info(f"  Use spectral: {enhanced_config.get('coarse_to_fine', {}).get('use_spectral', True)}")
        logger.info(f"  Use residual: {enhanced_config.get('coarse_to_fine', {}).get('use_residual', True)}")
    else:
        logger.info("\nCreating standard DoMINO model...")
        model = DoMINO(
            input_features=3,
            output_features_vol=num_vol_vars,
            output_features_surf=num_surf_vars,
            model_parameters=cfg.model,
        ).to(dist.device)
        logger.info("  ✅ Standard model created successfully")
    
    model = torch.compile(model, disable=True)

    # Print model summary
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

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

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.5
    )

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    epoch_number = 0

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")
    if dist.rank == 0:
        create_directory(model_save_path)
        create_directory(param_save_path)
        create_directory(best_model_path)

    if dist.world_size > 1:
        torch.distributed.barrier()

    init_epoch = load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=dist.device,
    )

    if init_epoch != 0:
        init_epoch += 1  # Start with the next epoch
    epoch_number = init_epoch

    # Retrieve the smallest validation loss if available
    numbers = []
    for filename in os.listdir(best_model_path):
        match = re.search(r"\d+\.\d*[1-9]\d*", filename)
        if match:
            number = float(match.group(0))
            numbers.append(number)

    best_vloss = min(numbers) if numbers else 1_000_000.0

    initial_integral_factor_orig = cfg.model.integral_loss_scaling_factor

    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting Training Loop")
    logger.info("="*60)

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch_number}")
        logger.info(f"{'='*40}")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        initial_integral_factor = initial_integral_factor_orig

        if epoch > 250:
            surface_scaling_loss = 1.0 * cfg.model.surf_loss_scaling
        else:
            surface_scaling_loss = cfg.model.surf_loss_scaling

        model.train(True)
        epoch_start_time = time.perf_counter()
        
        # Use enhanced training functions if enhanced features are enabled
        if use_enhanced_features:
            logger.info("Running enhanced training epoch...")
            avg_loss = train_epoch_enhanced(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                tb_writer=writer,
                logger=logger,
                gpu_handle=gpu_handle,
                epoch_index=epoch,
                device=dist.device,
                integral_scaling_factor=initial_integral_factor,
                loss_fn_type=cfg.model.loss_function,
                vol_loss_scaling=cfg.model.vol_loss_scaling,
                surf_loss_scaling=surface_scaling_loss,
                use_enhanced_features=use_enhanced_features,
            )
        else:
            logger.info("Running standard training epoch...")
            avg_loss = train_epoch(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                tb_writer=writer,
                logger=logger,
                gpu_handle=gpu_handle,
                epoch_index=epoch,
                device=dist.device,
                integral_scaling_factor=initial_integral_factor,
                loss_fn_type=cfg.model.loss_function,
                vol_loss_scaling=cfg.model.vol_loss_scaling,
                surf_loss_scaling=surface_scaling_loss,
            )

        epoch_end_time = time.perf_counter()
        logger.info(
            f"Training epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )

        model.eval()
        logger.info("Running validation...")
        
        # Use enhanced validation if enhanced features are enabled
        if use_enhanced_features:
            avg_vloss = validation_step_enhanced(
                dataloader=val_dataloader,
                model=model,
                device=dist.device,
                logger=logger,
                use_sdf_basis=cfg.model.use_sdf_in_basis_func,
                use_surface_normals=cfg.model.use_surface_normals,
                integral_scaling_factor=initial_integral_factor,
                loss_fn_type=cfg.model.loss_function,
                vol_loss_scaling=cfg.model.vol_loss_scaling,
                surf_loss_scaling=surface_scaling_loss,
                use_enhanced_features=use_enhanced_features,
            )
        else:
            avg_vloss = validation_step(
                dataloader=val_dataloader,
                model=model,
                device=dist.device,
                logger=logger,
                use_sdf_basis=cfg.model.use_sdf_in_basis_func,
                use_surface_normals=cfg.model.use_surface_normals,
                integral_scaling_factor=initial_integral_factor,
                loss_fn_type=cfg.model.loss_function,
                vol_loss_scaling=cfg.model.vol_loss_scaling,
                surf_loss_scaling=surface_scaling_loss,
            )

        scheduler.step()
        
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch_number} SUMMARY")
        logger.info(f"{'='*40}")
        logger.info(f"  Training loss: {avg_loss:.5f}")
        logger.info(f"  Validation loss: {avg_vloss:.5f}")
        logger.info(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"  Integral factor: {initial_integral_factor}")
        logger.info(f"  Total epoch time: {time.perf_counter() - start_time:.3f} seconds")

        if dist.rank == 0:
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number,
            )
            writer.flush()

        # Track best performance, and save the model's state
        if dist.world_size > 1:
            torch.distributed.barrier()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            logger.info(f"  🏆 NEW BEST VALIDATION LOSS: {best_vloss:.5f}")
            if dist.rank == 0:
                save_checkpoint(
                    to_absolute_path(best_model_path),
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=str(best_vloss.item()),  # Use loss as filename
                )
        else:
            logger.info(f"  Best validation loss so far: {best_vloss:.5f}")

        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0.0:
            logger.info(f"  💾 Saving checkpoint at epoch {epoch}")
            save_checkpoint(
                to_absolute_path(model_save_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
            )

        epoch_number += 1

        if scheduler.get_last_lr()[0] == 1e-6:
            logger.info("\n" + "="*60)
            logger.info("Training ended - Learning rate reached minimum")
            logger.info("="*60)
            exit()

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"Final best validation loss: {best_vloss:.5f}")
    logger.info(f"Checkpoints saved in: {model_save_path}")
    logger.info("\nNext steps:")
    logger.info("1. Test the model with: python test_enhanced.py")
    logger.info("2. View training curves in TensorBoard")


if __name__ == "__main__":
    main()