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

"""Check 2.1 rollout/replay likelihoods, token geometry and encoder boundaries without weights."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict
from transformers.utils.chat_template_utils import _compile_jinja_template

from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_21_flow_grpo.common import configure_scheduler, image_shape
from verl_omni.pipelines.qwen_image_21_flow_grpo.diffusers_training_adapter import QwenImage21FlowGRPO
from verl_omni.pipelines.qwen_image_21_flow_grpo.vllm_omni_rollout_adapter import QwenImage21PipelineWithLogProb
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


def _config(**pipeline_overrides):
    return SimpleNamespace(
        architecture="QwenImage21Pipeline",
        algorithm="flow_grpo",
        external_lib=None,
        lora_rank=0,
        lora_adapter_path=None,
        pipeline=SimpleNamespace(
            **(dict(height=64, width=64, num_inference_steps=4, true_cfg_scale=1.0) | pipeline_overrides)
        ),
        algo=SimpleNamespace(noise_level=0.7, sde_type="sde"),
    )


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = torch.nn.Linear(4, 4)
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, **kwargs):
        self.calls.append((encoder_hidden_states.shape[1], img_mask.clone(), kwargs))
        assert img_shapes == [[(1, 4, 4)]]
        assert not img_mask[:, : encoder_hidden_states.shape[1]].any()
        assert img_mask[:, encoder_hidden_states.shape[1] :].all()
        target = hidden_states * self.scale + encoder_hidden_states.mean() * 0.01
        prefix = target.new_zeros(target.shape[0], encoder_hidden_states.shape[1], target.shape[2])
        return (torch.cat([prefix, target], dim=1),)


def _pipeline():
    pipeline = QwenImage21PipelineWithLogProb.__new__(QwenImage21PipelineWithLogProb)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _Transformer()
    pipeline.scheduler = FlowMatchSDEDiscreteScheduler(use_dynamic_shifting=True)
    configure_scheduler(pipeline.scheduler, height=64, width=64, num_inference_steps=4, device="cpu")
    return pipeline


def test_registry_pair():
    config = _config()
    assert DiffusionModelBase.get_class(config) is QwenImage21FlowGRPO
    assert VllmOmniPipelineBase.get_class("QwenImage21Pipeline", "flow_grpo") is QwenImage21PipelineWithLogProb


@pytest.mark.parametrize("sde_type", ["sde", "cps"])
def test_rollout_replay_likelihood_and_gradient_with_ragged_prompts(sde_type):
    torch.manual_seed(5)
    pipeline = _pipeline()
    config = _config()
    config.algo.sde_type = sde_type
    rollout_results = []
    embeds = [torch.randn(1, length, 8) for length in (3, 7)]
    for row, embedding in enumerate(embeds):
        with torch.no_grad():
            rollout_results.append(
                pipeline.diffuse(
                    torch.randn(1, 16, 4),
                    embedding,
                    pipeline.scheduler.timesteps,
                    shape=(1, 4, 4),
                    noise_level=0.7,
                    sde_window=(1, 3),
                    sde_type=sde_type,
                    generator=torch.Generator().manual_seed(row),
                    logprobs=True,
                )
            )
    batch = TensorDict(
        {
            "all_latents": torch.cat([result[1] for result in rollout_results]),
            "all_timesteps": torch.cat([result[3] for result in rollout_results]),
        },
        batch_size=[2],
    )
    assert batch["all_latents"].dtype == torch.float32
    assert batch["all_latents"].shape == (2, 3, 16, 4)
    padded_embeds = torch.cat([torch.nn.functional.pad(embeds[0], (0, 0, 0, 4)), embeds[1]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    old_log_probs = torch.cat([result[2] for result in rollout_results])
    assert all(call[2]["kv_cache"] is None for call in pipeline.transformer.calls)

    replay_log_probs = []
    for step in range(2):
        inputs, negative = QwenImage21FlowGRPO.prepare_model_inputs(
            pipeline.transformer,
            config,
            batch["all_latents"],
            batch["all_timesteps"],
            padded_embeds,
            mask,
            None,
            None,
            batch,
            step,
        )
        log_prob, *_ = QwenImage21FlowGRPO.forward_and_sample_previous_step(
            pipeline.transformer, pipeline.scheduler, config, inputs, negative, batch, step
        )
        replay_log_probs.append(log_prob)
    replay = torch.stack(replay_log_probs, dim=1)
    torch.testing.assert_close(replay, old_log_probs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.exp(replay - old_log_probs), torch.ones_like(replay))
    (-replay.mean()).backward()
    assert torch.isfinite(pipeline.transformer.scale.grad)
    assert pipeline.transformer.scale.grad.abs() > 0
    assert [call[0] for call in pipeline.transformer.calls[-4:]] == [3, 7, 3, 7]


@pytest.mark.parametrize("height,width", [(63, 64), (64, 65), (0, 64)])
def test_invalid_geometry_rejected(height, width):
    with pytest.raises(ValueError, match="multiples of 32"):
        image_shape(height, width)


def test_replay_rejects_mismatched_latent_grid():
    config = _config()
    with pytest.raises(ValueError, match="latent length"):
        QwenImage21FlowGRPO.prepare_model_inputs(
            None,
            config,
            torch.zeros(1, 2, 4, 64),
            torch.zeros(1, 1),
            torch.zeros(1, 3, 8),
            torch.ones(1, 3),
            None,
            None,
            TensorDict({}, batch_size=[1]),
            0,
        )


def test_token_encoder_uses_pre_norm_features_and_restores_hook():
    pipeline = _pipeline()
    pipeline._drop_idx = 2
    pipeline._system_token_ids = [10, 11]
    pipeline._img_token_id = 99
    pipeline._max_length = 8192
    norm = torch.nn.Linear(4, 4, bias=False)
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    encoder = MagicMock(dtype=torch.float32)
    encoder.model.language_model.norm = norm
    encoder.side_effect = lambda **kwargs: SimpleNamespace(hidden_states=[norm(hidden)])
    pipeline.text_encoder = encoder
    embeds, mask = pipeline.encode_prompt([0, 10, 11, 12, 13], [0, 1, 1, 1, 1], max_sequence_length=8)
    torch.testing.assert_close(embeds, hidden[:, 2:])
    assert mask.tolist() == [[1, 1]]
    assert encoder.call_args.kwargs["input_ids"].tolist() == [[10, 11, 12, 13]]
    assert not encoder.call_args.kwargs["mm_token_type_ids"].any()
    assert not norm._forward_hooks
    encoder.side_effect = RuntimeError("encoder failure")
    with pytest.raises(RuntimeError, match="encoder failure"):
        pipeline.encode_prompt([10, 11, 12, 13], max_sequence_length=8)
    assert not norm._forward_hooks


@pytest.mark.parametrize("content", ["A red circle", [{"type": "text", "text": "A red circle"}]])
def test_recipe_template_matches_upstream_t2i(content):
    template_path = (
        Path(__file__).resolve().parents[2] / "examples/flowgrpo_trainer/qwen_image_21/prompt_template.jinja"
    )
    template = _compile_jinja_template(template_path.read_text())
    rendered = template.render(
        messages=[{"role": "system", "content": "old template"}, {"role": "user", "content": content}]
    )
    assert rendered == (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\nA red circle<|im_end|>\n<|im_start|>assistant\n"
    )


def test_step_execution_fails_instead_of_returning_inference_only_outputs():
    with pytest.raises(NotImplementedError, match="step_execution=false"):
        _pipeline().prepare_encode(None)


@pytest.mark.parametrize("output_type", ["pil", "latent"])
def test_request_forward_returns_native_trajectory_and_named_media(output_type):
    pipeline = _pipeline()
    pipeline.latent_channels = 4
    pipeline.encode_prompt = MagicMock(return_value=(torch.randn(1, 3, 8), torch.ones(1, 3)))
    pipeline.prepare_latents = MagicMock(return_value=(torch.randn(1, 16, 4), None))
    pipeline._decode_latents = MagicMock(return_value=SimpleNamespace(output=torch.zeros(1, 4, 64, 64)))
    sampling = SimpleNamespace(
        true_cfg_scale=1.0,
        sigmas=None,
        num_outputs_per_prompt=1,
        height=64,
        width=64,
        max_sequence_length=512,
        generator=None,
        seed=5,
        num_inference_steps=4,
        output_type=output_type,
        extra_args=dict(sde_window_size=2, sde_window_range=[0, 3], noise_level=0.7, sde_type="sde", logprobs=True),
    )
    request = SimpleNamespace(request_id="test", prompt={"prompt_ids": [10, 11, 12]}, sampling_params=sampling)
    result = pipeline.forward(request)
    assert result.trajectory_latents.shape == (1, 3, 16, 4)
    assert result.trajectory_log_probs.shape == (1, 2)
    assert result.trajectory_timesteps.shape == (1, 2)
    assert torch.isfinite(result.trajectory_log_probs).all()
    assert result.trajectory_latents.dtype == torch.float32
    assert result.output["metadata"]["prompt_embeddings"]["prompt_embeds"].shape == (1, 3, 8)
    payload = result.output["payload"]["image"][0]
    assert payload["image_latent"].shape == (16, 4)
    if output_type == "pil":
        assert payload["image_preview"].shape == (3, 64, 64)
        assert payload["image_preview"].dtype == torch.uint8
    else:
        pipeline._decode_latents.assert_not_called()


@pytest.mark.parametrize("request_id", ["dummy_req_id", "dummy_req_id/profile/0"])
def test_dummy_profiling_requests_need_no_training_payload(request_id):
    assert _pipeline().forward(SimpleNamespace(request_id=request_id)).output is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_veomni_matches_diffusers_predictions_and_backpropagates(dtype):
    native = pytest.importorskip(
        "veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer"
    )
    from diffusers import QwenImage21Transformer2DModel

    kwargs = dict(
        in_channels=4,
        out_channels=4,
        context_in_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    torch.manual_seed(7)
    reference = QwenImage21Transformer2DModel(**kwargs).to(dtype)
    from veomni.arguments import OpsImplementationConfig

    config = native.QwenImage21Transformer2DModel.config_class(**kwargs)
    config._set_ops_implementation(
        OpsImplementationConfig(
            attn_implementation="eager", rms_norm_implementation="eager", rotary_pos_emb_implementation="eager"
        )
    )
    model = native.QwenImage21Transformer2DModel(config).to(dtype)
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = dict(
        hidden_states=torch.randn(2, 16, 4, dtype=dtype),
        timestep=torch.tensor([0.8, 0.4]),
        encoder_hidden_states=torch.randn(2, 7, 8, dtype=dtype),
        encoder_hidden_states_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]),
        img_shapes=[[(1, 4, 4)], [(1, 4, 4)]],
    )
    expected = QwenImage21FlowGRPO.forward(reference, _config(), inputs)
    actual = QwenImage21FlowGRPO.forward(model, _config(), inputs)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.square().mean().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0
