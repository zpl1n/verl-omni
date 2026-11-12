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

"""Exercise VeOmni's model-runtime API without building a distributed model."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("veomni.models.model_runtime")
import veomni.models.model_runtime as runtime_module

from verl_omni.workers.config import VeOmniDiffusionEngineConfig, VeOmniDiffusionOptimizerConfig
from verl_omni.workers.engine.veomni.diffusion_impl import VeOmniDiffusionEngine
from verl_omni.workers.engine.veomni.model_runtime import VerlDiffusionModelRuntime


def _engine(tmp_path, *, forward_only=False):
    engine = object.__new__(VeOmniDiffusionEngine)
    engine.engine_config = VeOmniDiffusionEngineConfig(forward_only=forward_only)
    engine.optimizer_config = VeOmniDiffusionOptimizerConfig(lr=1e-4, total_training_steps=10)
    engine.model_config = SimpleNamespace(
        local_path=str(tmp_path),
        transformer_subfolder="transformer",
        config_path=None,
        local_tokenizer_path=str(tmp_path / "processor"),
        tokenizer_path=None,
        enable_gradient_checkpointing=True,
    )
    engine.dp_size = engine.dp_shard_size = engine.dp_replicate_size = 1
    engine._is_lora = False
    return engine


def test_model_runtime_arguments_keep_loading_and_training_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    engine = _engine(tmp_path)
    args = engine._build_veomni_dit_args()
    assert args.model.model_path == str(tmp_path / "transformer")
    assert args.model.config_path == args.model.model_path
    assert args.model.tokenizer_path == str(tmp_path / "processor")
    assert args.model.accelerator.init_device == "meta"
    assert args.model.accelerator.gradient_checkpointing.enable
    assert not args.model.accelerator.gradient_checkpointing.enable_reentrant
    assert args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.model.optimizer.lr == pytest.approx(1e-4)
    assert args.model.ops_implementation.attn_implementation == "eager"
    assert args.train.global_batch_size == 1
    assert args.train._train_steps == 10


def test_named_mesh_uses_the_engine_topology(tmp_path, monkeypatch):
    from veomni.distributed import parallel_state

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    seen = []
    state = SimpleNamespace(device_mesh="mesh", sp_enabled=False, dp_rank=2, dp_size=4, dp_group="group")
    monkeypatch.setattr(parallel_state, "init_parallel_state_from_config", lambda args, name: seen.append((args, name)))
    monkeypatch.setattr(parallel_state, "get_parallel_state_by_name", lambda name: state)
    engine = _engine(tmp_path, forward_only=True)
    engine._init_device_mesh()
    accelerator, name = seen[0]
    assert (accelerator.dp_replicate_size, accelerator.dp_shard_size) == (1, 4)
    assert name == "verl_diffusion_ref"
    assert engine.get_data_parallel_rank() == 2
    assert engine.get_data_parallel_size() == 4
    assert engine.get_data_parallel_group() == "group"


def test_runtime_builds_native_transformer_from_diffusers_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("veomni.models.diffusers.qwen_image21.qwen_image21_transformer")
    from diffusers import QwenImage21Transformer2DModel

    monkeypatch.setenv("WORLD_SIZE", "1")
    model = QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        context_in_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    model.save_pretrained(tmp_path / "transformer")
    engine = _engine(tmp_path)
    runtime = object.__new__(VerlDiffusionModelRuntime)
    runtime.args = engine._build_veomni_dit_args().model
    runtime.engine_config = engine.engine_config
    runtime._build_model()
    assert type(runtime.model).__module__.startswith("veomni.models.diffusers.qwen_image21.")
    assert runtime.model.config._attn_implementation == "eager"
    assert all(param.is_meta for param in runtime.model.parameters())
    assert runtime.model.state_dict().keys() == model.state_dict().keys()

    from verl_omni.model_merger.fsdp_model_merger import _transformer_schema

    runtime.model.config.save_pretrained(tmp_path / "actor")
    shapes, _ = _transformer_schema(tmp_path / "transformer", tmp_path / "actor/config.json", "QwenImage21Pipeline")
    assert shapes == {key: tuple(value.shape) for key, value in model.state_dict().items()}


@pytest.mark.parametrize("forward_only", [False, True])
def test_runtime_builds_in_order_and_does_not_load_processors_or_checkpoints(monkeypatch, forward_only):
    events = []
    active = []

    @contextmanager
    def scope(name):
        active.append(name)
        try:
            yield
        finally:
            active.pop()

    def build_model(runtime):
        assert active == ["test_runtime"]
        events.append("model")
        runtime.model = torch.nn.Linear(4, 4)

    def operation(name):
        def run(runtime):
            assert active == ["test_runtime"]
            events.append(name)

        return run

    base = runtime_module.VeOmniModelRuntime
    monkeypatch.setattr(runtime_module, "use_parallel_state", scope)
    monkeypatch.setattr(base, "setup", lambda self: events.append("mesh"))
    monkeypatch.setattr(base, "_build_model", build_model)
    monkeypatch.setattr(base, "_freeze_model_module", operation("freeze"))
    monkeypatch.setattr(base, "_build_parallelized_model", operation("parallelize"))
    monkeypatch.setattr(base, "_build_optimizer", operation("optimizer"))
    monkeypatch.setattr(base, "_build_model_assets", lambda self: pytest.fail("verl owns the processor"))
    monkeypatch.setattr(base, "build_checkpoint", lambda self: pytest.fail("verl owns checkpoints"))
    runtime = VerlDiffusionModelRuntime(
        SimpleNamespace(lora_config={}),
        "test_runtime",
        train=SimpleNamespace(),
        engine_config=SimpleNamespace(forward_only=forward_only, attn_implementation="eager"),
    )
    expected = ["mesh", "model", "freeze", "parallelize"] + ([] if forward_only else ["optimizer"])
    assert events == expected
    assert active == []
    assert runtime.checkpoint is None
    if forward_only:
        assert runtime.optimizer is None
