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

"""FlowGRPO rollout adapted from vllm-omni's Qwen-Image-2.1 pipeline.

https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/qwen_image_21
"""

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import (
    install_rollout_postprocessor,
    rollout_output,
    with_visual_artifacts,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import sample_per_sample_sde_windows
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import condition_images_from_payload, prompt_ids_from_payload
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import seeded_or_entropy_generator

from .common import configure_scheduler, image_shape, target_image_mask, validate_base_pipeline

__all__ = ["QwenImage21PipelineWithLogProb"]

install_rollout_postprocessor(pipeline_qwen_image_21, "get_qwen_image_21_post_process_func")


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="flow_grpo")
class QwenImage21PipelineWithLogProb(QwenImage21Pipeline):
    """Base T2I rollout with fp32 trajectories and request-local SDE randomness."""

    supports_request_batch = True
    supports_step_execution = False
    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
        }
    )

    def __init__(self, *, od_config, prefix=""):
        if (od_config.extras or {}).get("sample_sigmas") is not None:
            raise ValueError("Qwen-Image-2.1 FlowGRPO requires the base checkpoint, without Turbo sample_sigmas.")
        if od_config.parallel_config is not None and od_config.parallel_config.sequence_parallel_size != 1:
            raise ValueError("Qwen-Image-2.1 FlowGRPO currently requires rollout sequence parallel size 1.")
        super().__init__(od_config=od_config, prefix=prefix)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(od_config.model, subfolder="scheduler")
        template_ids = self.processor.tokenizer(self.prompt_template_t2i.format(""))["input_ids"]
        self._system_token_ids = template_ids[: self._drop_idx]

    def prepare_encode(self, state, **kwargs):
        """Reject the inherited inference-only step path, which does not record RL data."""
        raise NotImplementedError("Qwen-Image-2.1 FlowGRPO requires rollout.step_execution=false.")

    def encode_prompt(self, prompt_ids, attention_mask=None, *, max_sequence_length):
        """Encode agent-loop token IDs, using Qwen3-VL's pre-final-norm hidden states."""
        ids = torch.as_tensor(prompt_ids, device=self.device, dtype=torch.long)
        if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
            raise ValueError("Qwen-Image-2.1 encode_prompt expects one token sequence per request.")
        ids = ids.reshape(1, -1)
        mask = (
            torch.ones_like(ids, dtype=torch.bool)
            if attention_mask is None
            else torch.as_tensor(attention_mask, device=self.device, dtype=torch.bool).reshape(1, -1)
        )
        if mask.shape != ids.shape:
            raise ValueError("Qwen-Image-2.1 prompt_mask must match prompt_ids.")
        ids = ids[:, mask[0]]
        mask = torch.ones_like(ids)
        if not 0 < ids.shape[1] - self._drop_idx <= max_sequence_length <= self._max_length:
            raise ValueError("Qwen-Image-2.1 prompt length exceeds pipeline.max_sequence_length or is empty.")
        prefix = ids.new_tensor(self._system_token_ids)
        if not torch.equal(ids[0, : self._drop_idx], prefix):
            raise ValueError("Use the Qwen-Image-2.1 system prompt and chat template from the example recipe.")
        if (ids == self._img_token_id).any():
            raise ValueError("Qwen-Image-2.1 FlowGRPO currently supports text-to-image prompts only.")

        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = self.text_encoder(
                input_ids=ids,
                attention_mask=mask,
                mm_token_type_ids=torch.zeros_like(ids),
                output_hidden_states=True,
            )
        finally:
            handle.remove()
        embeds = outputs.hidden_states[-1][:, self._drop_idx :].to(self.text_encoder.dtype)
        return embeds, torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)

    def diffuse(
        self, latents, prompt_embeds, timesteps, *, shape, noise_level, sde_window, sde_type, generator, logprobs
    ):
        """Sample one trajectory; retain W+1 latents and W likelihoods for replay."""
        all_latents, all_log_probs, all_timesteps = [], [], []
        img_mask = target_image_mask(prompt_embeds, latents.shape[1])
        start, end = sde_window
        self.scheduler.set_begin_index(0)
        self.scheduler._init_step_index(timesteps[0])
        for index, timestep in enumerate(timesteps):
            if index == start:
                all_latents.append(latents.detach().float().clone())
            # Full recomputation matches the trainable transformer; no cross-step KV cache.
            noise_pred = self.transformer(
                hidden_states=latents.to(self.transformer.img_in.weight.dtype),
                encoder_hidden_states=prompt_embeds,
                timestep=timestep.expand(latents.shape[0]) / 1000.0,
                img_shapes=[[shape]] * latents.shape[0],
                img_mask=img_mask,
                kv_cache=None,
                return_dict=False,
            )[0][:, -latents.shape[1] :]
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep,
                latents.float(),
                generator=generator,
                noise_level=noise_level if start <= index < end else 0.0,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )
            if start <= index < end:
                all_latents.append(latents.detach().float().clone())
                all_timesteps.append(timestep.expand(latents.shape[0]))
                if logprobs:
                    all_log_probs.append(log_prob)
        return (
            latents,
            torch.stack(all_latents, dim=1),
            torch.stack(all_log_probs, dim=1) if logprobs else None,
            torch.stack(all_timesteps, dim=1),
        )

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> DiffusionOutput | list[DiffusionOutput]:
        """Process requests separately so different prompt lengths retain their own RoPE geometry."""
        if isinstance(req, DiffusionRequestBatch):
            return [self.forward(request) for request in req.requests]
        if OmniDiffusionRequest.is_dummy_run_request_id(req.request_id):
            return DiffusionOutput(output=None)
        if not isinstance(req.prompt, dict):
            raise ValueError("Qwen-Image-2.1 FlowGRPO expects one tokenized prompt per request.")
        prompt = req.prompt
        media = prompt.get("multi_modal_data") or {}
        if condition_images_from_payload(prompt) or any(media.get(key) is not None for key in ("audio", "video")):
            raise ValueError("Qwen-Image-2.1 FlowGRPO currently supports text-to-image requests only.")
        prompt_ids = prompt_ids_from_payload(prompt)
        if prompt_ids is None:
            raise ValueError("Qwen-Image-2.1 FlowGRPO requires prompt_ids from the diffusion agent loop.")
        sampling = req.sampling_params
        validate_base_pipeline(sampling.true_cfg_scale)
        if sampling.sigmas is not None:
            raise ValueError("Custom sigmas are not supported by Qwen-Image-2.1 FlowGRPO replay.")
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError("Use rollout.n for multiple samples; num_outputs_per_prompt must be 1.")
        shape = image_shape(sampling.height, sampling.width)
        prompt_embeds, prompt_mask = self.encode_prompt(
            prompt_ids, prompt.get("prompt_mask"), max_sequence_length=sampling.max_sequence_length
        )
        generator = sampling.generator
        if generator is None:
            generator = seeded_or_entropy_generator(sampling.seed, self.device)
        latents, _ = self.prepare_latents(
            None, 1, self.latent_channels, sampling.height, sampling.width, torch.float32, self.device, generator
        )
        configure_scheduler(
            self.scheduler,
            height=sampling.height,
            width=sampling.width,
            num_inference_steps=sampling.num_inference_steps,
            device=self.device,
        )
        extra = sampling.extra_args or {}
        window_size = extra.get("sde_window_size")
        window_range = extra["sde_window_range"]
        if window_size is not None and (
            window_size <= 0
            or window_range is None
            or len(window_range) != 2
            or not 0 <= window_range[0] < window_range[1] <= len(self.scheduler.timesteps) - 1
            or window_size > window_range[1] - window_range[0]
        ):
            raise ValueError("SDE window must be nonempty and end before the final deterministic denoising step.")
        if extra["logprobs"] and extra["noise_level"] <= 0:
            raise ValueError("Training log probabilities require a positive noise_level.")
        if extra["sde_type"] not in ("sde", "cps"):
            raise ValueError("Qwen-Image-2.1 FlowGRPO supports sde_type='sde' or 'cps'.")
        windows = sample_per_sample_sde_windows(
            sde_window_size=window_size,
            sde_window_range=window_range,
            num_timesteps=len(self.scheduler.timesteps),
            batch_size=1,
            generator=generator,
            device=self.device,
        )
        latents, trajectory, log_probs, timesteps = self.diffuse(
            latents,
            prompt_embeds,
            self.scheduler.timesteps,
            shape=shape,
            noise_level=extra["noise_level"],
            sde_window=windows[0],
            sde_type=extra["sde_type"],
            generator=generator,
            logprobs=extra["logprobs"],
        )
        output_type = sampling.output_type or "pil"
        requested = extra.get("requested_outputs")
        decode = output_type != "latent" or (requested is not None and "image_preview" in requested)
        image = self._decode_latents(latents, sampling.height, sampling.width).output if decode else None
        if image is not None:
            # The 2.1 VAE emits RGBA; the T2I reward/preview contract is RGB.
            image = image[:, :3]
        result = rollout_output(
            media=image,
            trajectory_latents=trajectory,
            trajectory_log_probs=log_probs,
            trajectory_timesteps=timesteps,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_mask,
                "negative_prompt_embeds": None,
                "negative_prompt_embeds_mask": None,
            },
            to_cpu=True,
        )
        return with_visual_artifacts(
            result,
            decoded=image,
            latents=latents,
            latent_layout="LC",
            output_type=output_type,
            context=f"pipeline={type(self).__name__}, request_id={req.request_id}",
            requested=requested,
        )
