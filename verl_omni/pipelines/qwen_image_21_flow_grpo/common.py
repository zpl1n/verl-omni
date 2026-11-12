# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Shared Qwen-Image-2.1 geometry and schedule for rollout and replay."""

import numpy as np
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift


def image_shape(height: int, width: int) -> tuple[int, int, int]:
    """Return the unpatched latent grid; the VAE compresses each spatial axis 16x."""
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("Qwen-Image-2.1 height and width must be positive multiples of 32.")
    return (1, height // 16, width // 16)


def target_image_mask(prompt_embeds: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Build (B, L + N/4) slots; each image slot represents four latent tokens."""
    if num_latents <= 0 or num_latents % 4:
        raise ValueError("Qwen-Image-2.1 requires a positive multiple of four latent tokens.")
    batch, length = prompt_embeds.shape[:2]
    mask = torch.zeros((batch, length + num_latents // 4), dtype=torch.bool, device=prompt_embeds.device)
    mask[:, length:] = True
    return mask


def configure_scheduler(scheduler, *, height: int, width: int, num_inference_steps: int, device) -> None:
    """Use the base checkpoint's flow schedule on both sides of the RL boundary."""
    _, latent_height, latent_width = image_shape(height, width)
    if num_inference_steps < 2:
        raise ValueError("Qwen-Image-2.1 FlowGRPO requires at least two inference steps.")
    mu = calculate_shift(
        latent_height * latent_width,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=device,
        sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
        mu=mu,
    )


def validate_base_pipeline(true_cfg_scale: float) -> None:
    """Limit the first integration to the base T2I checkpoint without CFG."""
    if true_cfg_scale != 1.0:
        raise ValueError("Qwen-Image-2.1 FlowGRPO currently requires pipeline.true_cfg_scale=1.0.")
