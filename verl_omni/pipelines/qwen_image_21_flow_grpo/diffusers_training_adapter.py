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

"""Qwen-Image-2.1 FlowGRPO replay through the native VeOmni transformer."""

import torch
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

from .common import configure_scheduler, image_shape, target_image_mask, validate_base_pipeline

__all__ = ["QwenImage21FlowGRPO"]


@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="flow_grpo")
class QwenImage21FlowGRPO(QwenImage):
    """Reuse FlowGRPO likelihoods with 2.1's unpatched latent and joint-token layout."""

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device):
        """Configure the same latent-length-dependent schedule as rollout."""
        validate_base_pipeline(model_config.pipeline.true_cfg_scale)
        configure_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def validate_lora_config(cls, model_config):
        """Reject adapter-only updates until their rollout synchronization is validated."""
        raise ValueError("Qwen-Image-2.1 FlowGRPO currently supports full-weight training; set model.lora_rank=0.")

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    ):
        """Select a replay step from fp32 (B, T+1, N, C) trajectories."""
        validate_base_pipeline(model_config.pipeline.true_cfg_scale)
        if model_config.lora_rank or model_config.lora_adapter_path is not None:
            cls.validate_lora_config(model_config)
        height = tu.get_non_tensor_data(micro_batch, "height", default=model_config.pipeline.height)
        width = tu.get_non_tensor_data(micro_batch, "width", default=model_config.pipeline.width)
        shape = image_shape(height, width)
        hidden_states = latents[:, step]
        if hidden_states.shape[1] != shape[1] * shape[2]:
            raise ValueError("Qwen-Image-2.1 replay latent length does not match height/width at VAE scale 16.")
        if prompt_embeds_mask is None or prompt_embeds_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError("Qwen-Image-2.1 replay requires the rollout prompt embedding mask.")
        return {
            "hidden_states": hidden_states,
            "timestep": timesteps[:, step].float() / 1000.0,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "img_shapes": [[shape]] * hidden_states.shape[0],
        }, None

    @classmethod
    def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
        """Unpad each prompt before joint RoPE so replay matches single-request rollout."""
        predictions = []
        for row in range(model_inputs["hidden_states"].shape[0]):
            valid = model_inputs["encoder_hidden_states_mask"][row].bool()
            embeds = model_inputs["encoder_hidden_states"][row : row + 1, valid]
            if embeds.shape[1] == 0:
                raise ValueError("Qwen-Image-2.1 replay received an empty prompt embedding sequence.")
            latents = model_inputs["hidden_states"][row : row + 1]
            predictions.append(
                module(
                    hidden_states=latents,
                    timestep=model_inputs["timestep"][row : row + 1],
                    encoder_hidden_states=embeds,
                    img_shapes=[model_inputs["img_shapes"][row]],
                    img_mask=target_image_mask(embeds, latents.shape[1]),
                    return_dict=False,
                )[0][:, -latents.shape[1] :]
            )
        return torch.cat(predictions, dim=0)
