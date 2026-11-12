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

"""Bridge VeOmni's model-runtime lifecycle to verl's diffusion engine ownership."""

from veomni.models.model_runtime import VeOmniModelRuntime

from .lora_utils import _reject_veomni_moe_expert_lora
from .patch import _apply_attention_backend


class VerlDiffusionModelRuntime(VeOmniModelRuntime):
    """Build the trainable transformer; verl owns processors, rollout and checkpoints."""

    def __init__(self, args, model_name, *, train, engine_config):
        self.engine_config = engine_config
        super().__init__(args, model_name, train=train)

    def _build_model(self):
        super()._build_model()
        _apply_attention_backend(self.model, self.engine_config.attn_implementation)

    def _freeze_model_module(self):
        super()._freeze_model_module()
        if self.args.lora_config:
            _reject_veomni_moe_expert_lora(self.model)

    def _build_optimizer(self):
        if not self.engine_config.forward_only:
            super()._build_optimizer()

    def _build_model_assets(self):
        # The diffusion model config already owns the tokenizer and processor.
        self.model_assets = [self.model_config]

    def build_checkpoint(self):
        """Leave checkpoint ownership to the engine's FSDPCheckpointManager."""
        return
