# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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
"""Thinker NPU patches adapted from https://github.com/verl-project/verl/pull/7960."""

import logging
import os
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)
_experts_call_logged = False


def _with_first_experts_call_log(experts_cls, forward):
    @wraps(forward)
    def first_forward(self, hidden_states, top_k_index, top_k_weights):
        global _experts_call_logged

        output = forward(self, hidden_states, top_k_index, top_k_weights)
        if not _experts_call_logged:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else "uninitialized"
            print(
                f"[Qwen3-Omni NPU] first MoE forward: qwen3_5_moe_experts_forward_npu "
                f"pid={os.getpid()} rank={rank} device={hidden_states.device} "
                f"dtype={hidden_states.dtype} shape={tuple(hidden_states.shape)}",
                flush=True,
            )
            _experts_call_logged = True
        # Remove the logging wrapper from subsequent forwards, including recomputation.
        if experts_cls.forward is first_forward:
            experts_cls.forward = forward
        return output

    return first_forward


def apply_qwen3_omni_thinker_npu_patch(text_config: Any) -> None:
    """Install process-wide Thinker patches before FSDP wrapping on NPU workers.

    Args:
        text_config: The loaded Thinker's text configuration.

    Raises:
        ValueError: The configured activation is incompatible with SwiGLU.
    """
    if text_config.hidden_act != "silu":
        raise ValueError(f"Qwen3-Omni Thinker NPU patches require hidden_act='silu', got {text_config.hidden_act!r}.")

    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling
    from verl.models.transformers.npu_patch import (
        apply_rotary_pos_emb_npu,
        qwen3_5_moe_experts_forward_npu,
        rms_norm_forward_npu,
        silu_forward_npu,
    )

    patches = (
        (modeling.Qwen3OmniMoeThinkerTextExperts, qwen3_5_moe_experts_forward_npu),
        (modeling.Qwen3OmniMoeThinkerTextRMSNorm, rms_norm_forward_npu),
        (modeling.Qwen3OmniMoeTextRMSNorm, rms_norm_forward_npu),
        (modeling.Qwen3OmniMoeThinkerTextMLP, silu_forward_npu),
    )

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        # Transformers' fifth positional argument is unsqueeze_dim, not position_ids.
        return apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    for module_cls, forward in patches:
        module_cls.forward = forward
    if not _experts_call_logged:
        modeling.Qwen3OmniMoeThinkerTextExperts.forward = _with_first_experts_call_log(
            modeling.Qwen3OmniMoeThinkerTextExperts, qwen3_5_moe_experts_forward_npu
        )
    modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("Applied Qwen3-Omni Thinker NPU patches: experts, RMSNorm, MLP and RoPE")
