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

"""Build a download-free 2.1 checkpoint for the VeOmni FlowGRPO smoke test."""

import argparse
import json
from pathlib import Path

import torch
from build_qwen_image_edit_plus_tiny_random import _build_tiny_chatml_tokenizer
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)


def build_checkpoint(output_dir: Path) -> None:
    """Write small real components with the production component names and 16x VAE geometry."""
    marker = output_dir / "qwen_image_21_smoke.json"
    if marker.exists() and json.loads(marker.read_text()) == {"version": 1}:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing checkpoint: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    tokenizer = _build_tiny_chatml_tokenizer()
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(output_dir / "processor")
    config = Qwen3VLConfig(
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [0, 1],
            "patch_size": 16,
        },
    )
    Qwen3VLForConditionalGeneration(config).save_pretrained(output_dir / "text_encoder")
    QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        context_in_dim=32,
        axes_dims_rope=(4, 6, 6),
    ).save_pretrained(output_dir / "transformer")
    AutoencoderKLQwenImage21(
        base_dim=16,
        decoder_base_dim=16,
        z_dim=4,
        dim_mult=[1, 2, 4, 4, 4],
        num_res_blocks=1,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    ).save_pretrained(output_dir / "vae")
    FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).save_pretrained(output_dir / "scheduler")
    index = {
        "_class_name": "QwenImage21Pipeline",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    (output_dir / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    marker.write_text(json.dumps({"version": 1}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build_checkpoint(parser.parse_args().output_dir)
